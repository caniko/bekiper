//! Safe RAII wrappers for HIP device memory allocations.

use std::ffi::c_void;
use std::ptr;

use crate::error::{Result, check_hip};
use crate::hip_sys;

/// Owned HIP device memory allocation. Freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    size: usize,
}

impl DeviceBuffer {
    /// Allocate `size` bytes of device memory.
    pub fn alloc(size: usize) -> Result<Self> {
        let mut ptr: *mut c_void = ptr::null_mut();
        check_hip(
            // SAFETY: `ptr` points to valid writable host storage for HIP to
            // fill with an allocation handle. The returned pointer is owned by
            // the resulting `DeviceBuffer` and freed in `Drop`.
            unsafe { hip_sys::hipMalloc(&mut ptr, size) },
            "hipMalloc",
        )?;
        Ok(Self { ptr, size })
    }

    /// Allocate device memory for `count` elements of type `T`.
    pub fn alloc_typed<T>(count: usize) -> Result<Self> {
        Self::alloc(count * std::mem::size_of::<T>())
    }

    /// Raw device pointer.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy data from host to this device buffer.
    pub fn copy_from_host(&self, src: &[u8]) -> Result<()> {
        assert!(src.len() <= self.size, "source exceeds device buffer size");
        check_hip(
            // SAFETY: `self.ptr` is a live HIP allocation of at least
            // `self.size` bytes, and the assertion above proves the source
            // slice is no larger than the destination allocation.
            unsafe {
                hip_sys::hipMemcpy(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    hip_sys::hipMemcpyHostToDevice,
                )
            },
            "hipMemcpy H2D",
        )
    }

    /// Copy data from this device buffer to host.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        let len = dst.len().min(self.size);
        check_hip(
            // SAFETY: `self.ptr` is a live HIP allocation of at least `len`
            // bytes, and `dst` is a valid mutable host slice of at least `len`
            // bytes.
            unsafe {
                hip_sys::hipMemcpy(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    len,
                    hip_sys::hipMemcpyDeviceToHost,
                )
            },
            "hipMemcpy D2H",
        )
    }

    /// Copy `count` elements of type `T` from this device buffer to a host Vec.
    pub fn copy_to_host_typed<T: Clone + Default>(&self, count: usize) -> Result<Vec<T>> {
        let byte_len = count * std::mem::size_of::<T>();
        assert!(byte_len <= self.size, "requested count exceeds buffer size");
        let mut out = vec![T::default(); count];
        check_hip(
            // SAFETY: `out` owns initialized contiguous storage for `count`
            // `T` values, `byte_len` matches that storage size, and the
            // assertion above proves the device allocation is large enough.
            unsafe {
                hip_sys::hipMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    byte_len,
                    hip_sys::hipMemcpyDeviceToHost,
                )
            },
            "hipMemcpy D2H typed",
        )?;
        Ok(out)
    }

    /// Zero-fill this device buffer.
    pub fn zero(&self) -> Result<()> {
        check_hip(
            // SAFETY: `self.ptr` is a live HIP allocation of `self.size`
            // bytes owned by this `DeviceBuffer`.
            unsafe { hip_sys::hipMemset(self.ptr, 0, self.size) },
            "hipMemset",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: non-null `self.ptr` came from `hipMalloc` in `alloc`,
            // has not been freed elsewhere, and `Drop` runs exactly once for
            // this owning wrapper.
            unsafe {
                hip_sys::hipFree(self.ptr);
            }
        }
    }
}

// SAFETY: `DeviceBuffer` only owns an opaque HIP device allocation. HIP device
// pointers may be passed across host threads; mutation still requires external
// synchronization by callers sharing the same allocation.
unsafe impl Send for DeviceBuffer {}
// SAFETY: Shared references expose only the raw device pointer and copy methods;
// HIP handles cross-thread device access, and callers must synchronize mutable
// GPU work using streams/events at a higher layer.
unsafe impl Sync for DeviceBuffer {}
