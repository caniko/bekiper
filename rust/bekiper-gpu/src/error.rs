//! GPU pipeline error types.

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("HIP error: {0} (code {1})")]
    Hip(String, i32),

    #[error("VA-API error: {0} (status {1})")]
    VaApi(String, i32),

    #[error("Vulkan error: {0} (code {1})")]
    Vulkan(String, i32),

    #[error("kernel error: {0}")]
    Kernel(String),

    #[error("unsupported: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, GpuError>;

/// Check a HIP return code, returning an error on failure.
pub(crate) fn check_hip(code: i32, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(GpuError::Hip(context.into(), code))
    }
}

/// Check a VA-API return code, returning an error on failure.
#[cfg(feature = "vaapi")]
pub(crate) fn check_va(status: i32, context: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(GpuError::VaApi(context.into(), status))
    }
}
