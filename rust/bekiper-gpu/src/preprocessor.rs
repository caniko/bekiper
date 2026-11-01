//! GPU preprocessor: launches fused HIP kernels for frame preprocessing.
//!
//! Manages the compiled kernel module and provides high-level methods to
//! preprocess NV12 or RGB frames into model-ready NCHW float32 tensors
//! entirely on the GPU.

use std::ffi::c_void;
use std::ptr;

use crate::device_mem::DeviceBuffer;
use crate::error::{GpuError, Result, check_hip};
use crate::gpu_frame::GpuFrame;
use crate::hip_sys;

// Compiled kernel code object, embedded at build time by build.rs.
include!(concat!(env!("OUT_DIR"), "/preprocess_kernel.rs"));

/// Normalization parameters for the preprocessing kernel.
#[derive(Debug, Clone, Copy)]
pub struct NormParams {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl NormParams {
    /// ImageNet normalization: (pixel/255 - mean) / std.
    pub fn imagenet() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// Zero-to-one normalization: pixel/255. Mean=0, std=1.
    pub fn zero_one() -> Self {
        Self {
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }
}

/// GPU preprocessor: loads the HIP kernel module and dispatches preprocessing.
pub struct GpuPreprocessor {
    module: hip_sys::hipModule_t,
    fn_nv12_to_nchw: hip_sys::hipFunction_t,
    fn_rgb_to_nchw: hip_sys::hipFunction_t,
    fn_nv12_batch: hip_sys::hipFunction_t,
    stream: hip_sys::hipStream_t,
    /// Reusable output buffer for single-frame preprocessing.
    output_buf: Option<DeviceBuffer>,
    /// Model input dimensions.
    dst_w: u32,
    dst_h: u32,
    norm: NormParams,
}

impl GpuPreprocessor {
    /// Create a new preprocessor for the given model input dimensions.
    pub fn new(dst_w: u32, dst_h: u32, norm: NormParams) -> Result<Self> {
        // Load the embedded compiled kernel
        let mut module: hip_sys::hipModule_t = ptr::null_mut();
        check_hip(
            // SAFETY: `module` points to writable storage for the returned HIP
            // module handle, and the embedded code object bytes live for the
            // full program lifetime.
            unsafe {
                hip_sys::hipModuleLoadData(
                    &mut module,
                    PREPROCESS_KERNEL_CO.as_ptr() as *const c_void,
                )
            },
            "hipModuleLoadData",
        )?;

        // Get function handles
        let fn_nv12_to_nchw = get_function(module, b"preprocess_nv12_to_nchw\0")?;
        let fn_rgb_to_nchw = get_function(module, b"preprocess_rgb_to_nchw\0")?;
        let fn_nv12_batch = get_function(module, b"preprocess_nv12_batch_to_nchw\0")?;

        // Create a dedicated stream for preprocessing
        let mut stream: hip_sys::hipStream_t = ptr::null_mut();
        check_hip(
            // SAFETY: `stream` points to writable storage for the created HIP
            // stream handle.
            unsafe { hip_sys::hipStreamCreate(&mut stream) },
            "hipStreamCreate",
        )?;

        // Pre-allocate output buffer for single frames
        let tensor_bytes = (3 * dst_w * dst_h) as usize * std::mem::size_of::<f32>();
        let output_buf = DeviceBuffer::alloc(tensor_bytes)?;

        Ok(Self {
            module,
            fn_nv12_to_nchw,
            fn_rgb_to_nchw,
            fn_nv12_batch,
            stream,
            output_buf: Some(output_buf),
            dst_w,
            dst_h,
            norm,
        })
    }

    /// Preprocess a single NV12 GpuFrame → NCHW float32 tensor on GPU.
    ///
    /// Returns a device buffer containing [1, 3, dst_h, dst_w] float32 data.
    /// The buffer is reusable — calling this again overwrites the previous result.
    pub fn preprocess_nv12(&mut self, frame: &GpuFrame) -> Result<&DeviceBuffer> {
        let output = self
            .output_buf
            .as_ref()
            .ok_or_else(|| GpuError::Kernel("output buffer not initialized".into()))?;

        let block = hip_sys::dim3::new(16, 16, 1);
        let grid = hip_sys::dim3::new(
            self.dst_w.div_ceil(block.x),
            self.dst_h.div_ceil(block.y),
            1,
        );

        let y_ptr = frame.y_plane_ptr();
        let uv_ptr = frame.uv_plane_ptr();
        let y_pitch = frame.layout.y_pitch as i32;
        let uv_pitch = frame.layout.uv_pitch as i32;
        let src_w = frame.layout.width as i32;
        let src_h = frame.layout.height as i32;
        let out_ptr = output.as_ptr();
        let dst_w = self.dst_w as i32;
        let dst_h = self.dst_h as i32;
        let mean_r = self.norm.mean[0];
        let mean_g = self.norm.mean[1];
        let mean_b = self.norm.mean[2];
        let std_r = self.norm.std[0];
        let std_g = self.norm.std[1];
        let std_b = self.norm.std[2];

        let mut args: [*mut c_void; 15] = [
            &y_ptr as *const _ as *mut c_void,
            &uv_ptr as *const _ as *mut c_void,
            &y_pitch as *const _ as *mut c_void,
            &uv_pitch as *const _ as *mut c_void,
            &src_w as *const _ as *mut c_void,
            &src_h as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &dst_w as *const _ as *mut c_void,
            &dst_h as *const _ as *mut c_void,
            &mean_r as *const _ as *mut c_void,
            &mean_g as *const _ as *mut c_void,
            &mean_b as *const _ as *mut c_void,
            &std_r as *const _ as *mut c_void,
            &std_g as *const _ as *mut c_void,
            &std_b as *const _ as *mut c_void,
        ];

        check_hip(
            // SAFETY: The kernel function and stream were resolved from the
            // loaded module, and `args` contains stable pointers to POD kernel
            // arguments for the duration of the launch call.
            unsafe {
                hip_sys::hipModuleLaunchKernel(
                    self.fn_nv12_to_nchw,
                    grid.x,
                    grid.y,
                    grid.z,
                    block.x,
                    block.y,
                    block.z,
                    0, // shared mem
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch preprocess_nv12_to_nchw",
        )?;

        // Synchronize to ensure kernel completion
        check_hip(
            // SAFETY: `self.stream` is a live HIP stream created in `new`.
            unsafe { hip_sys::hipStreamSynchronize(self.stream) },
            "hipStreamSynchronize",
        )?;

        Ok(output)
    }

    /// Preprocess RGB24 data (already on GPU) → NCHW float32 tensor.
    ///
    /// Used for the fallback path: CPU decode → hipMemcpy → GPU preprocess.
    pub fn preprocess_rgb(
        &mut self,
        rgb_device_ptr: *const u8,
        src_w: u32,
        src_h: u32,
        src_pitch: u32,
    ) -> Result<&DeviceBuffer> {
        let output = self
            .output_buf
            .as_ref()
            .ok_or_else(|| GpuError::Kernel("output buffer not initialized".into()))?;

        let block = hip_sys::dim3::new(16, 16, 1);
        let grid = hip_sys::dim3::new(
            self.dst_w.div_ceil(block.x),
            self.dst_h.div_ceil(block.y),
            1,
        );

        let out_ptr = output.as_ptr();
        let src_w_i = src_w as i32;
        let src_h_i = src_h as i32;
        let src_pitch_i = src_pitch as i32;
        let dst_w = self.dst_w as i32;
        let dst_h = self.dst_h as i32;
        let mean_r = self.norm.mean[0];
        let mean_g = self.norm.mean[1];
        let mean_b = self.norm.mean[2];
        let std_r = self.norm.std[0];
        let std_g = self.norm.std[1];
        let std_b = self.norm.std[2];

        let mut args: [*mut c_void; 13] = [
            &rgb_device_ptr as *const _ as *mut c_void,
            &src_w_i as *const _ as *mut c_void,
            &src_h_i as *const _ as *mut c_void,
            &src_pitch_i as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &dst_w as *const _ as *mut c_void,
            &dst_h as *const _ as *mut c_void,
            &mean_r as *const _ as *mut c_void,
            &mean_g as *const _ as *mut c_void,
            &mean_b as *const _ as *mut c_void,
            &std_r as *const _ as *mut c_void,
            &std_g as *const _ as *mut c_void,
            &std_b as *const _ as *mut c_void,
        ];

        check_hip(
            // SAFETY: The stream, function handle, and argument pointers remain
            // valid for the duration of the launch call.
            unsafe {
                hip_sys::hipModuleLaunchKernel(
                    self.fn_rgb_to_nchw,
                    grid.x,
                    grid.y,
                    grid.z,
                    block.x,
                    block.y,
                    block.z,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch preprocess_rgb_to_nchw",
        )?;

        check_hip(
            // SAFETY: `self.stream` is a live HIP stream created in `new`.
            unsafe { hip_sys::hipStreamSynchronize(self.stream) },
            "hipStreamSynchronize",
        )?;

        Ok(output)
    }

    /// Preprocess a batch of NV12 GpuFrames → [N, 3, dst_h, dst_w] float32 tensor.
    ///
    /// Returns a newly allocated device buffer (caller owns it).
    pub fn preprocess_nv12_batch(&self, frames: &[&GpuFrame]) -> Result<DeviceBuffer> {
        let n = frames.len();
        let frame_floats = 3 * self.dst_w as usize * self.dst_h as usize;
        let total_bytes = n * frame_floats * std::mem::size_of::<f32>();
        let output = DeviceBuffer::alloc(total_bytes)?;

        // Build per-frame parameter arrays on the host, then upload to device
        let y_ptrs: Vec<*const u8> = frames.iter().map(|f| f.y_plane_ptr()).collect();
        let uv_ptrs: Vec<*const u8> = frames.iter().map(|f| f.uv_plane_ptr()).collect();
        let y_pitches: Vec<i32> = frames.iter().map(|f| f.layout.y_pitch as i32).collect();
        let uv_pitches: Vec<i32> = frames.iter().map(|f| f.layout.uv_pitch as i32).collect();
        let src_ws: Vec<i32> = frames.iter().map(|f| f.layout.width as i32).collect();
        let src_hs: Vec<i32> = frames.iter().map(|f| f.layout.height as i32).collect();

        // Upload pointer arrays to device
        let d_y_ptrs = upload_to_device(&y_ptrs)?;
        let d_uv_ptrs = upload_to_device(&uv_ptrs)?;
        let d_y_pitches = upload_to_device(&y_pitches)?;
        let d_uv_pitches = upload_to_device(&uv_pitches)?;
        let d_src_ws = upload_to_device(&src_ws)?;
        let d_src_hs = upload_to_device(&src_hs)?;

        let block = hip_sys::dim3::new(16, 16, 1);
        let grid = hip_sys::dim3::new(
            self.dst_w.div_ceil(block.x),
            self.dst_h.div_ceil(block.y),
            n as u32,
        );

        let d_y_ptr = d_y_ptrs.as_ptr();
        let d_uv_ptr = d_uv_ptrs.as_ptr();
        let d_yp = d_y_pitches.as_ptr();
        let d_uvp = d_uv_pitches.as_ptr();
        let d_sw = d_src_ws.as_ptr();
        let d_sh = d_src_hs.as_ptr();
        let out_ptr = output.as_ptr();
        let dst_w = self.dst_w as i32;
        let dst_h = self.dst_h as i32;
        let batch_size = n as i32;
        let mean_r = self.norm.mean[0];
        let mean_g = self.norm.mean[1];
        let mean_b = self.norm.mean[2];
        let std_r = self.norm.std[0];
        let std_g = self.norm.std[1];
        let std_b = self.norm.std[2];

        let mut args: [*mut c_void; 16] = [
            &d_y_ptr as *const _ as *mut c_void,
            &d_uv_ptr as *const _ as *mut c_void,
            &d_yp as *const _ as *mut c_void,
            &d_uvp as *const _ as *mut c_void,
            &d_sw as *const _ as *mut c_void,
            &d_sh as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &dst_w as *const _ as *mut c_void,
            &dst_h as *const _ as *mut c_void,
            &batch_size as *const _ as *mut c_void,
            &mean_r as *const _ as *mut c_void,
            &mean_g as *const _ as *mut c_void,
            &mean_b as *const _ as *mut c_void,
            &std_r as *const _ as *mut c_void,
            &std_g as *const _ as *mut c_void,
            &std_b as *const _ as *mut c_void,
        ];

        check_hip(
            // SAFETY: The batch kernel sees device pointers uploaded above, and
            // `args` points to stable POD launch arguments for this call.
            unsafe {
                hip_sys::hipModuleLaunchKernel(
                    self.fn_nv12_batch,
                    grid.x,
                    grid.y,
                    grid.z,
                    block.x,
                    block.y,
                    block.z,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch preprocess_nv12_batch_to_nchw",
        )?;

        check_hip(
            // SAFETY: `self.stream` is a live HIP stream created in `new`.
            unsafe { hip_sys::hipStreamSynchronize(self.stream) },
            "hipStreamSynchronize",
        )?;

        Ok(output)
    }

    /// The HIP stream used by this preprocessor.
    pub fn stream(&self) -> hip_sys::hipStream_t {
        self.stream
    }

    /// Model input dimensions.
    pub fn model_dims(&self) -> (u32, u32) {
        (self.dst_w, self.dst_h)
    }
}

impl Drop for GpuPreprocessor {
    fn drop(&mut self) {
        // SAFETY: We own the HIP stream and module handles and only destroy
        // them once during drop after null checks.
        unsafe {
            if !self.stream.is_null() {
                hip_sys::hipStreamDestroy(self.stream);
            }
            if !self.module.is_null() {
                hip_sys::hipModuleUnload(self.module);
            }
        }
    }
}

// SAFETY: The struct owns opaque HIP handles whose thread-safety contract is
// delegated to the HIP runtime; moving ownership between threads does not
// create aliased mutable Rust references.
unsafe impl Send for GpuPreprocessor {}

/// Helper: upload a slice of POD data to a device buffer.
fn upload_to_device<T>(data: &[T]) -> Result<DeviceBuffer> {
    let bytes = std::mem::size_of_val(data);
    let buf = DeviceBuffer::alloc(bytes)?;
    check_hip(
        // SAFETY: `buf` points at writable device memory, `data` is a valid
        // host slice for `bytes`, and HIP performs the host-to-device copy.
        unsafe {
            hip_sys::hipMemcpy(
                buf.as_ptr(),
                data.as_ptr() as *const c_void,
                bytes,
                hip_sys::hipMemcpyHostToDevice,
            )
        },
        "upload_to_device",
    )?;
    Ok(buf)
}

fn get_function(module: hip_sys::hipModule_t, name: &[u8]) -> Result<hip_sys::hipFunction_t> {
    let mut func: hip_sys::hipFunction_t = ptr::null_mut();
    check_hip(
        // SAFETY: `func` points to writable storage and `name` is expected to
        // be a trailing-NUL kernel symbol emitted by build-time codegen.
        unsafe { hip_sys::hipModuleGetFunction(&mut func, module, name.as_ptr()) },
        &format!(
            "hipModuleGetFunction({})",
            std::str::from_utf8(&name[..name.len() - 1]).unwrap_or("?")
        ),
    )?;
    Ok(func)
}
