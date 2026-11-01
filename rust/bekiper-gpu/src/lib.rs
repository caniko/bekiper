//! Zero-copy GPU pipeline for video pose estimation.
//!
//! Provides a complete GPU-resident pipeline:
//!
//! 1. **Vulkan Video decode** -> NV12 frame in VRAM (via VCN hardware decoder)
//! 2. **DMA-BUF export** -> HIP device pointer (zero-copy, same physical pages)
//! 3. **Fused HIP kernel** -> NV12->RGB + bilinear resize + normalize -> NCHW float32
//! 4. **ONNX Runtime MIGraphX EP** -> inference with no H2D/D2H copies
//!
//! Uses RADV (Vulkan) instead of VA-API to avoid LLVM symbol conflicts
//! between Mesa's radeonsi and MIGraphX's bundled LLVM.

pub mod device_mem;
pub mod error;
pub mod gpu_frame;
pub mod hip_sys;
pub mod preprocessor;

#[cfg(feature = "vulkan")]
pub mod ffmpeg_vk_sys;
#[cfg(feature = "vulkan")]
pub mod vk_sys;

#[cfg(feature = "vaapi")]
pub mod va_sys;

#[path = "lib/init.rs"]
mod init;

#[cfg(test)]
#[path = "lib/tests.rs"]
mod tests;

pub use device_mem::DeviceBuffer;
pub use error::GpuError;
pub use gpu_frame::{GpuFrame, Nv12Layout};
pub use init::{hip_memcpy_d2d, init};
pub use preprocessor::{GpuPreprocessor, NormParams};
#[cfg(feature = "vulkan")]
pub use vk_sys::VulkanContext;
