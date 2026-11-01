//! Zero-copy GPU frame import via DMA-BUF.
//!
//! Supports two backends:
//! - **Vulkan**: Export `VkDeviceMemory` as DMA-BUF fd, import into HIP.
//!   Uses RADV (ACO shader compiler, no LLVM conflict with MIGraphX).
//! - **VA-API** (legacy): Export VA surface as DMA-BUF fd, import into HIP.
//!
//! Both paths use `hipImportExternalMemory` with `hipExternalMemoryHandleTypeOpaqueFd`.
//! (ROCm 7.x lacks `hipExternalMemoryHandleTypeDmaBufFd`, but the opaque fd
//! path goes through the same kernel DMA-BUF import under the hood on AMD.)

use std::ffi::c_void;
use std::ptr;

#[cfg(feature = "vaapi")]
use crate::error::check_va;
use crate::error::{GpuError, Result, check_hip};
use crate::hip_sys;
#[cfg(feature = "vaapi")]
use crate::va_sys;

/// NV12 frame data accessible as HIP device memory (zero-copy).
///
/// Created from a Vulkan or VA-API surface via DMA-BUF export + HIP import.
/// The external memory handle is destroyed on drop.
pub struct GpuFrame {
    /// HIP device pointer to the raw NV12 data.
    device_ptr: *mut c_void,
    /// HIP external memory handle (for cleanup).
    ext_mem: hip_sys::hipExternalMemory_t,
    /// NV12 plane layout info.
    pub layout: Nv12Layout,
}

/// Layout information for an NV12 frame in VRAM.
#[derive(Debug, Clone, Copy)]
pub struct Nv12Layout {
    pub width: u32,
    pub height: u32,
    /// Y plane offset from the base device pointer (bytes).
    pub y_offset: u32,
    /// Y plane row pitch (bytes).
    pub y_pitch: u32,
    /// UV plane offset from the base device pointer (bytes).
    pub uv_offset: u32,
    /// UV plane row pitch (bytes).
    pub uv_pitch: u32,
    /// Total buffer size (bytes).
    pub total_size: u64,
}

impl GpuFrame {
    /// Import a DMA-BUF fd into HIP device memory (zero-copy).
    ///
    /// This is the common path for both Vulkan and VA-API backends.
    /// The fd is consumed (HIP takes ownership on success, closed on failure).
    ///
    /// # Safety
    /// The DMA-BUF fd must be valid and the memory must be fully written
    /// (decode complete). The layout must accurately describe the NV12 data.
    pub unsafe fn from_dma_buf_fd(fd: i32, buf_size: u64, layout: Nv12Layout) -> Result<Self> {
        // SAFETY: The caller guarantees `fd` is a valid DMA-BUF whose lifetime
        // can be transferred to HIP, and that `layout` describes initialized
        // NV12 data within `buf_size`.
        unsafe {
            let mut handle_desc = hip_sys::hipExternalMemoryHandleDesc::default();
            handle_desc.type_ = hip_sys::hipExternalMemoryHandleTypeOpaqueFd;
            handle_desc.handle.fd = fd;
            handle_desc.size = buf_size;

            let mut ext_mem: hip_sys::hipExternalMemory_t = ptr::null_mut();
            let import_result = check_hip(
                hip_sys::hipImportExternalMemory(&mut ext_mem, &handle_desc),
                "hipImportExternalMemory",
            );

            // On failure, close the fd ourselves (on success HIP takes ownership)
            if import_result.is_err() {
                libc::close(fd);
                return import_result.map(|_| unreachable!());
            }

            // Map to a device pointer
            let mut buf_desc = hip_sys::hipExternalMemoryBufferDesc::default();
            buf_desc.size = buf_size;

            let mut device_ptr: *mut c_void = ptr::null_mut();
            check_hip(
                hip_sys::hipExternalMemoryGetMappedBuffer(&mut device_ptr, ext_mem, &buf_desc),
                "hipExternalMemoryGetMappedBuffer",
            )?;

            Ok(Self {
                device_ptr,
                ext_mem,
                layout,
            })
        }
    }

    /// Import NV12 pixel data from a CPU buffer into HIP device memory.
    ///
    /// Allocates HIP device memory and copies the NV12 data from host.
    /// Used when zero-copy DMA-BUF export is not available (e.g., RADV
    /// doesn't support DMA-BUF for Vulkan Video decode images).
    pub fn from_nv12_host(nv12_data: &[u8], width: u32, height: u32, y_pitch: u32) -> Result<Self> {
        let uv_height = height / 2;
        let y_size = (y_pitch * height) as usize;
        let uv_size = (y_pitch * uv_height) as usize;
        let total_size = y_size + uv_size;

        if nv12_data.len() < total_size {
            return Err(GpuError::Unsupported(format!(
                "NV12 buffer too small: {} < {total_size}",
                nv12_data.len()
            )));
        }

        // Allocate HIP device memory
        let mut device_ptr: *mut c_void = ptr::null_mut();
        check_hip(
            // SAFETY: `device_ptr` points to valid writable host storage for
            // HIP to fill with an allocation handle. The allocation is owned by
            // the returned `GpuFrame` and freed in `Drop`.
            unsafe { hip_sys::hipMalloc(&mut device_ptr, total_size) },
            "hipMalloc (NV12)",
        )?;

        // Copy NV12 data from host to device
        check_hip(
            // SAFETY: `device_ptr` is a live HIP allocation of `total_size`
            // bytes, and the length check above proves `nv12_data` has at
            // least that many initialized bytes.
            unsafe {
                hip_sys::hipMemcpy(
                    device_ptr,
                    nv12_data.as_ptr() as *const c_void,
                    total_size,
                    hip_sys::hipMemcpyHostToDevice,
                )
            },
            "hipMemcpy H2D (NV12)",
        )?;

        let layout = Nv12Layout {
            width,
            height,
            y_offset: 0,
            y_pitch,
            uv_offset: y_pitch * height,
            uv_pitch: y_pitch,
            total_size: total_size as u64,
        };

        Ok(Self {
            device_ptr,
            ext_mem: ptr::null_mut(), // No external memory handle — we own this
            layout,
        })
    }

    /// Import a VA-API surface into HIP device memory (zero-copy).
    ///
    /// Exports the surface as a DMA-BUF fd via `vaExportSurfaceHandle`, then
    /// imports it into HIP via `hipImportExternalMemory` using the opaque fd
    /// handle type. No CPU-side data copy occurs.
    ///
    /// # Safety
    /// The VA surface must be fully decoded. The surface must remain valid
    /// for the lifetime of this `GpuFrame`.
    #[cfg(feature = "vaapi")]
    pub unsafe fn from_va_surface(
        va_display: va_sys::VADisplay,
        va_surface: va_sys::VASurfaceID,
    ) -> Result<Self> {
        // SAFETY: The caller guarantees `va_display` and `va_surface` are valid
        // VA-API handles and that the surface remains live while it is exported
        // and imported into HIP below.
        unsafe {
            // Ensure decoding is complete
            check_va(
                va_sys::vaSyncSurface(va_display, va_surface),
                "vaSyncSurface",
            )?;

            // Export as DRM PRIME 2 descriptor (DMA-BUF fd)
            let mut desc = va_sys::VADRMPRIMESurfaceDescriptor::default();
            check_va(
                va_sys::vaExportSurfaceHandle(
                    va_display,
                    va_surface,
                    va_sys::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                    va_sys::VA_EXPORT_SURFACE_READ_ONLY | va_sys::VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                    &mut desc as *mut _ as *mut c_void,
                ),
                "vaExportSurfaceHandle",
            )?;

            if desc.num_objects == 0 {
                return Err(GpuError::Unsupported(
                    "VA surface export returned 0 objects".into(),
                ));
            }

            let fd = desc.objects[0].fd;
            let buf_size = desc.objects[0].size as u64;

            // Extract NV12 plane layout from the DRM PRIME descriptor.
            // AMD mesa exports NV12 as 2 layers: layer 0 = Y, layer 1 = UV
            let layout = if desc.num_layers >= 2 {
                Nv12Layout {
                    width: desc.width,
                    height: desc.height,
                    y_offset: desc.layers[0].offset[0],
                    y_pitch: desc.layers[0].pitch[0],
                    uv_offset: desc.layers[1].offset[0],
                    uv_pitch: desc.layers[1].pitch[0],
                    total_size: buf_size,
                }
            } else {
                let y_pitch = desc.layers[0].pitch[0];
                Nv12Layout {
                    width: desc.width,
                    height: desc.height,
                    y_offset: 0,
                    y_pitch,
                    uv_offset: y_pitch * desc.height,
                    uv_pitch: y_pitch,
                    total_size: buf_size,
                }
            };

            Self::from_dma_buf_fd(fd, buf_size, layout)
        }
    }

    /// Device pointer to the Y plane.
    pub fn y_plane_ptr(&self) -> *const u8 {
        // SAFETY: `device_ptr` is the base of the live imported NV12 allocation
        // and `layout.y_offset` is produced from the decoder/export descriptor.
        unsafe { (self.device_ptr as *const u8).add(self.layout.y_offset as usize) }
    }

    /// Device pointer to the UV plane.
    pub fn uv_plane_ptr(&self) -> *const u8 {
        // SAFETY: `device_ptr` is the base of the live imported NV12 allocation
        // and `layout.uv_offset` is produced from the decoder/export descriptor.
        unsafe { (self.device_ptr as *const u8).add(self.layout.uv_offset as usize) }
    }

    /// Raw device pointer to the start of the buffer.
    pub fn device_ptr(&self) -> *mut c_void {
        self.device_ptr
    }
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        // SAFETY: `GpuFrame` owns either an imported HIP external-memory handle
        // or a HIP allocation created by `from_nv12_host`. Each is released once
        // here, and null handles are skipped.
        unsafe {
            if !self.ext_mem.is_null() {
                // DMA-BUF import path: destroy external memory handle
                hip_sys::hipDestroyExternalMemory(self.ext_mem);
            } else if !self.device_ptr.is_null() {
                // Host-upload path (from_nv12_host): free HIP allocation
                hip_sys::hipFree(self.device_ptr);
            }
        }
    }
}

// SAFETY: `GpuFrame` owns opaque HIP handles and device pointers. Those handles
// can be moved between host threads; callers must synchronize any actual GPU
// work that reads or writes the underlying frame memory.
unsafe impl Send for GpuFrame {}
