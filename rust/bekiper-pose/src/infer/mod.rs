#[cfg(feature = "infer")]
mod model;
#[cfg(feature = "infer")]
mod preprocess;
#[cfg(feature = "infer")]
mod spec;

#[cfg(feature = "infer")]
pub use model::{
    DEFAULT_FALLBACK_BATCH_SIZE, DEFAULT_GPU_DEVICE_ID, DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_VRAM_HEADROOM_FRACTION, GpuOptions, OnnxPoseModel,
};
#[cfg(feature = "infer")]
pub use preprocess::{CropRegion, preprocess_batch, preprocess_frame, rescale_keypoints};
#[cfg(feature = "infer")]
pub use spec::{
    DlcSuperAnimalDefinition, Keypoint, LabelAlias, LightningPoseDefinition, ModelSpec,
    Normalization, OutputLayout, PoseModelDefinition, SleapSingleInstanceDefinition,
};
