#[path = "test_gpu_pipeline/find_test_video.rs"]
mod find_test_video;

use bekiper_gpu::{GpuFrame, GpuPreprocessor, NormParams};
use bekiper_pose::video::{HwBackend, VideoDecoder};
use find_test_video::*;

#[path = "test_gpu_pipeline/test_no_llvm_conflict_vulkan_plus_hip.rs"]
mod test_no_llvm_conflict_vulkan_plus_hip;
