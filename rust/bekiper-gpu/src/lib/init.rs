/// Initialize the HIP runtime on the given device. Call once before using
/// any GPU functions.
///
/// Breaking change over the zero-arg `init()` helper: every caller must
/// select a device explicitly. Resolve `device_id` from config +
/// `BEKIPER_GPU_DEVICE` in the application, not here — this crate stays
/// side-effect free.
pub fn init(device_id: u32) -> crate::error::Result<()> {
    crate::error::check_hip(unsafe { crate::hip_sys::hipInit(0) }, "hipInit")?;
    crate::error::check_hip(
        unsafe { crate::hip_sys::hipSetDevice(device_id as i32) },
        "hipSetDevice",
    )?;
    tracing::info!(device_id, "HIP runtime initialized");
    Ok(())
}

/// Device-to-device `hipMemcpy`. Both `dst` and `src` must be HIP device pointers.
///
/// # Safety
/// Caller must ensure both pointers are valid HIP device allocations with at
/// least `size` bytes.
pub unsafe fn hip_memcpy_d2d(
    dst: *mut std::ffi::c_void,
    src: *mut std::ffi::c_void,
    size: usize,
) -> crate::error::Result<()> {
    use crate::{error, hip_sys};
    unsafe {
        error::check_hip(
            hip_sys::hipMemcpy(dst, src as *const _, size, hip_sys::hipMemcpyDeviceToDevice),
            "hipMemcpy (D2D)",
        )
    }
}
