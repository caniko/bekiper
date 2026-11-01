use std::ffi::c_void;
#[cfg(feature = "gpu")]
use std::ptr;

use image::DynamicImage;

#[cfg(feature = "gpu")]
use bekiper_gpu::vk_sys::VulkanContext;

/// Which hardware decode backend is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwBackend {
    Vulkan,
    VaApi,
    None,
}

/// Metadata extracted from a video file.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    /// Output dimensions (may differ from source if resize is active).
    pub output_width: u32,
    pub output_height: u32,
    pub total_frames: u64,
    pub fps: f64,
    pub codec: String,
    pub hw_accel: bool,
    /// Stream time base (numerator/denominator) for PTS conversion.
    pub time_base: (i32, i32),
}

/// A decoded frame with its presentation timestamp.
#[derive(Debug)]
pub struct DecodedFrame {
    pub index: u64,
    pub image: DynamicImage,
    /// Presentation timestamp in seconds (from PTS × time_base).
    pub timestamp_seconds: f64,
}

/// A decoded frame still resident on the GPU as a VA-API surface.
///
/// Holds the ffmpeg frame reference to prevent VA-API surface reuse.
/// The caller must import this into HIP (via `GpuFrame::from_va_surface`) and
/// preprocess it before dropping this struct or decoding more frames.
pub struct RawVaFrame {
    /// VA-API surface ID (`VASurfaceID`, the `u32` handle).
    pub surface_id: u32,
    /// VA display handle (valid for the lifetime of the `VideoDecoder`).
    pub va_display: *mut c_void,
    /// Source frame width.
    pub width: u32,
    /// Source frame height.
    pub height: u32,
    /// Frame index in the video sequence.
    pub index: u64,
    /// Presentation timestamp in seconds.
    pub timestamp_seconds: f64,
    /// Keep the ffmpeg frame alive to prevent VA surface reuse.
    pub(super) _frame: ffmpeg_next::frame::Video,
}

// SAFETY: The VA surface ID is an opaque integer handle, the VA display is
// valid for the decoder lifetime, and the retained ffmpeg frame prevents
// surface reuse while the value is moved between host threads.
unsafe impl Send for RawVaFrame {}

/// A decoded frame still resident on the GPU as a Vulkan Video surface.
///
/// Holds the ffmpeg frame reference to prevent Vulkan image reuse.
/// Prefer `gpu_next_nv12_host` for the complete decode → host transfer path.
/// This struct is used internally and for low-level Vulkan frame access.
#[cfg(feature = "gpu")]
pub struct RawVkFrame {
    /// Pointer to the AVVkFrame (from `AVFrame.data[0]`).
    /// Valid for the lifetime of this struct (held by _frame).
    pub vk_frame: *const bekiper_gpu::ffmpeg_vk_sys::AVVkFrame,
    /// Source frame width.
    pub width: u32,
    /// Source frame height.
    pub height: u32,
    /// Frame index in the video sequence.
    pub index: u64,
    /// Presentation timestamp in seconds.
    pub timestamp_seconds: f64,
    /// Keep the ffmpeg frame alive to prevent Vulkan image reuse.
    pub(super) _frame: ffmpeg_next::frame::Video,
}

#[cfg(feature = "gpu")]
// SAFETY: The pointer targets data owned by `_frame`, and the retained ffmpeg
// frame prevents Vulkan image reuse while the value is moved between host
// threads.
unsafe impl Send for RawVkFrame {}

/// A decoded NV12 frame transferred to host memory.
///
/// Created by `VideoDecoder::gpu_next_nv12_host`, which decodes via Vulkan Video
/// then transfers the frame to CPU via `av_hwframe_transfer_data`. The NV12 data
/// can then be uploaded to HIP via `GpuFrame::from_nv12_host`.
#[cfg(feature = "gpu")]
pub struct RawNv12Frame {
    /// Raw NV12 pixel data (Y plane followed by interleaved UV plane).
    pub data: Vec<u8>,
    /// Source frame width.
    pub width: u32,
    /// Source frame height.
    pub height: u32,
    /// Y plane row pitch (bytes per row, may include padding).
    pub y_pitch: u32,
    /// Frame index in the video sequence.
    pub index: u64,
    /// Presentation timestamp in seconds.
    pub timestamp_seconds: f64,
}

/// Frame-by-frame video decoder with hardware acceleration support.
///
/// Tries Vulkan Video first (no LLVM conflict with MIGraphX), then VA-API,
/// then falls back to CPU decoding.
/// Yields frames as `DynamicImage` for direct use with `OnnxPoseModel::infer_frame`.
pub struct VideoDecoder {
    pub(super) decoder: ffmpeg_next::codec::decoder::Video,
    pub(super) input: ffmpeg_next::format::context::Input,
    pub(super) stream_index: usize,
    pub(super) scaler: ffmpeg_next::software::scaling::Context,
    pub(super) info: VideoInfo,
    pub(super) frame_index: u64,
    /// Cached frame from seek overshoot — returned by next `next_frame()` call.
    pub(super) cached_frame: Option<(u64, DynamicImage)>,
    /// Cached timestamp from seek overshoot.
    pub(super) cached_timestamp: Option<f64>,
    /// PTS of the last decoded frame (for timestamp tracking).
    pub(super) last_pts_seconds: f64,
    // Keep the hw device context alive for the lifetime of the decoder.
    // The AVCodecContext holds a ref, but we own the original.
    pub(super) _hw_device_ctx: Option<HwDeviceCtx>,
    /// Which hardware decode backend is active.
    pub(super) hw_backend: HwBackend,
    /// Vulkan context (owned, kept alive for the lifetime of the decoder).
    #[cfg(feature = "gpu")]
    pub(super) _vk_ctx: Option<std::sync::Arc<VulkanContext>>,
}

/// RAII wrapper for AVBufferRef* from av_hwdevice_ctx_create.
pub(super) struct HwDeviceCtx {
    pub(super) ptr: *mut ffmpeg_next::ffi::AVBufferRef,
}

impl Drop for HwDeviceCtx {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffmpeg_next::ffi::av_buffer_unref(&mut self.ptr);
            }
        }
    }
}

// SAFETY: `AVBufferRef` is ffmpeg's ref-counted hardware-device context handle;
// this wrapper only transfers ownership of one reference and releases it once in
// `Drop`.
unsafe impl Send for HwDeviceCtx {}

/// Result of opening a hardware-accelerated decoder.
pub(super) struct OpenDecoderResult {
    pub(super) decoder: ffmpeg_next::codec::decoder::Video,
    pub(super) hw_backend: HwBackend,
    pub(super) hw_ctx: Option<HwDeviceCtx>,
    #[cfg(feature = "gpu")]
    pub(super) vk_ctx: Option<std::sync::Arc<VulkanContext>>,
}

#[cfg(feature = "gpu")]
pub(super) fn should_try_vulkan_decoder(codec_id: ffmpeg_next::codec::Id) -> bool {
    // The local gfx1100 / 7900 XTX ROCm path still uses MIGraphX for inference,
    // but RADV/FFmpeg's Vulkan Video decoder rejects H.264 on this device with
    // "Device does not support decoding h264!". Avoid selecting that path so
    // H.264 videos use CPU decode directly instead of failing once per stream.
    codec_id != ffmpeg_next::codec::Id::H264
}

/// `get_format` callback for Vulkan Video decoding.
///
/// Called by FFmpeg when negotiating the pixel format. When Vulkan is
/// available, we intercept frames context creation to add
/// `VkExportMemoryAllocateInfo` with `DMA_BUF_BIT_EXT` so that decoded
/// frames can be exported as DMA-BUF fds for zero-copy HIP import.
///
/// The `AVCodecContext.opaque` field must point to a `VkExportMemoryAllocateInfo`
/// allocated by `alloc_export_memory_info_dma_buf`.
#[cfg(feature = "gpu")]
/// Find the last (lowest-priority) software pixel format in the format list.
/// FFmpeg lists formats in priority order: hw formats first, sw formats last.
/// Returns AV_PIX_FMT_NONE if no software format found.
unsafe fn find_sw_format(
    fmt: *const ffmpeg_next::ffi::AVPixelFormat,
) -> ffmpeg_next::ffi::AVPixelFormat {
    use ffmpeg_next::ffi::AVPixelFormat;
    // Hardware formats to skip
    const HW_FMTS: &[AVPixelFormat] = &[
        AVPixelFormat::AV_PIX_FMT_VULKAN,
        AVPixelFormat::AV_PIX_FMT_CUDA,
        AVPixelFormat::AV_PIX_FMT_VAAPI,
        AVPixelFormat::AV_PIX_FMT_VDPAU,
        AVPixelFormat::AV_PIX_FMT_D3D11,
        AVPixelFormat::AV_PIX_FMT_DXVA2_VLD,
        AVPixelFormat::AV_PIX_FMT_QSV,
        AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX,
        AVPixelFormat::AV_PIX_FMT_DRM_PRIME,
        AVPixelFormat::AV_PIX_FMT_MEDIACODEC,
        AVPixelFormat::AV_PIX_FMT_OPENCL,
    ];
    let mut p = fmt;
    let mut last_sw = AVPixelFormat::AV_PIX_FMT_NONE;
    unsafe {
        while *p != AVPixelFormat::AV_PIX_FMT_NONE {
            if !HW_FMTS.contains(&*p) {
                last_sw = *p;
            }
            p = p.add(1);
        }
    }
    last_sw
}

#[cfg(feature = "gpu")]
pub(super) unsafe extern "C" fn vulkan_get_format(
    s: *mut ffmpeg_next::ffi::AVCodecContext,
    fmt: *const ffmpeg_next::ffi::AVPixelFormat,
) -> ffmpeg_next::ffi::AVPixelFormat {
    use bekiper_gpu::ffmpeg_vk_sys::AVVulkanFramesContext;

    // Find AV_PIX_FMT_VULKAN in the format list
    let mut p = fmt;
    let mut found_vulkan = false;
    unsafe {
        while *p != ffmpeg_next::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *p == ffmpeg_next::ffi::AVPixelFormat::AV_PIX_FMT_VULKAN {
                found_vulkan = true;
            }
            p = p.add(1);
        }
    }

    let sw_fallback = unsafe { find_sw_format(fmt) };

    if !found_vulkan {
        return sw_fallback;
    }

    // Use avcodec_get_hw_frames_parameters to get a properly configured
    // frames context from the decoder.
    let mut frames_ref: *mut ffmpeg_next::ffi::AVBufferRef = ptr::null_mut();
    let ret = unsafe {
        ffmpeg_next::ffi::avcodec_get_hw_frames_parameters(
            s,
            (*s).hw_device_ctx,
            ffmpeg_next::ffi::AVPixelFormat::AV_PIX_FMT_VULKAN,
            &mut frames_ref,
        )
    };
    if ret < 0 || frames_ref.is_null() {
        // Vulkan hw frames not available for this codec — fall through to sw format
        tracing::warn!(
            "Vulkan Video decode not supported for this stream, falling back to CPU decode"
        );
        return sw_fallback;
    }

    // Configure export and multiplane settings on the frames context
    unsafe {
        let frames_ctx = (*frames_ref).data as *mut ffmpeg_next::ffi::AVHWFramesContext;
        let vk_frames = (*frames_ctx).hwctx as *mut AVVulkanFramesContext;

        // Set alloc_pnext to the VkExportMemoryAllocateInfo stored in opaque
        let export_info = (*s).opaque;
        if !export_info.is_null() {
            (*vk_frames).alloc_pnext[0] = export_info;
        }

        // Disable multiplane for simpler memory layout (single NV12 image)
        (*vk_frames).flags = bekiper_gpu::ffmpeg_vk_sys::AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE;

        // Initialize the configured frames context
        let ret = ffmpeg_next::ffi::av_hwframe_ctx_init(frames_ref);
        if ret < 0 {
            tracing::warn!("Vulkan frames context init failed ({ret}), falling back to CPU decode");
            ffmpeg_next::ffi::av_buffer_unref(&mut frames_ref);
            return sw_fallback;
        }

        // Set the frames context on the codec context
        (*s).hw_frames_ctx = frames_ref;
    }

    ffmpeg_next::ffi::AVPixelFormat::AV_PIX_FMT_VULKAN
}
