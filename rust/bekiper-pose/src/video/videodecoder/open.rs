use super::*;

impl VideoDecoder {
    /// Open a video file, outputting frames at original resolution.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_inner(path, None)
    }

    /// Open a video file, resizing frames to the given dimensions via ffmpeg's
    /// SIMD-optimized swscale. This eliminates the need for a separate CPU resize
    /// step, which is the single biggest preprocessing bottleneck.
    pub fn open_with_resize(path: &Path, output_width: u32, output_height: u32) -> Result<Self> {
        Self::open_inner(path, Some((output_width, output_height)))
    }

    fn open_inner(path: &Path, output_size: Option<(u32, u32)>) -> Result<Self> {
        ffmpeg_next::init().map_err(|e| BekiperError::Data(format!("ffmpeg init: {e}")))?;

        let input = ffmpeg_next::format::input(path).map_err(|e| {
            BekiperError::Data(format!("cannot open video {}: {e}", path.display()))
        })?;

        let stream = input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| BekiperError::Data("no video stream found".into()))?;

        let stream_index = stream.index();
        let codec_params = stream.parameters();

        let fps = f64::from(stream.avg_frame_rate());
        let time_base = stream.time_base();
        let total_frames = derive_total_frames(
            stream.frames(),
            stream.duration(),
            (time_base.numerator(), time_base.denominator()),
            fps,
        );

        let result = Self::open_decoder(&codec_params)?;
        let decoder = result.decoder;
        let hw_backend = result.hw_backend;
        let hw_ctx = result.hw_ctx;
        #[cfg(feature = "gpu")]
        let vk_ctx = result.vk_ctx;

        let hw_accel = hw_backend != HwBackend::None;
        let width = decoder.width();
        let height = decoder.height();
        let (out_w, out_h) = output_size.unwrap_or((width, height));
        let codec_name = decoder
            .codec()
            .map(|c| c.name().to_string())
            .unwrap_or_else(|| "unknown".into());

        // For hw-accel, decoded frames are in NV12 format.
        // We need to handle hw->sw transfer before scaling.
        let src_format = if hw_accel {
            // After hw transfer, frames will be NV12
            ffmpeg_next::format::Pixel::NV12
        } else {
            decoder.format()
        };

        let scaler = ffmpeg_next::software::scaling::Context::get(
            src_format,
            width,
            height,
            ffmpeg_next::format::Pixel::RGB24,
            out_w,
            out_h,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| BekiperError::Data(format!("scaler init: {e}")))?;

        let info = VideoInfo {
            width,
            height,
            output_width: out_w,
            output_height: out_h,
            total_frames,
            fps,
            codec: codec_name,
            hw_accel,
            time_base: (time_base.numerator(), time_base.denominator()),
        };

        tracing::info!(
            width,
            height,
            output_width = out_w,
            output_height = out_h,
            total_frames,
            fps,
            codec = %info.codec,
            hw_accel,
            backend = ?hw_backend,
            "opened video"
        );

        Ok(Self {
            decoder,
            input,
            stream_index,
            scaler,
            info,
            frame_index: 0,
            cached_frame: None,
            cached_timestamp: None,
            last_pts_seconds: 0.0,
            _hw_device_ctx: hw_ctx,
            hw_backend,
            #[cfg(feature = "gpu")]
            _vk_ctx: vk_ctx,
        })
    }

    fn open_decoder(codec_params: &ffmpeg_next::codec::Parameters) -> Result<OpenDecoderResult> {
        // Try Vulkan Video first (no LLVM conflict with MIGraphX)
        #[cfg(feature = "gpu")]
        {
            let codec_id = codec_params.id();
            if should_try_vulkan_decoder(codec_id) {
                match Self::open_vulkan_decoder(codec_params) {
                    Ok((decoder, hw_ctx, vk_ctx)) => {
                        tracing::info!("using Vulkan Video hardware decoder");
                        return Ok(OpenDecoderResult {
                            decoder,
                            hw_backend: HwBackend::Vulkan,
                            hw_ctx: Some(hw_ctx),
                            #[cfg(feature = "gpu")]
                            vk_ctx: Some(vk_ctx),
                        });
                    }
                    Err(e) => {
                        tracing::debug!("Vulkan Video unavailable: {e}");
                    }
                }
            } else {
                tracing::debug!(
                    codec = ?codec_id,
                    "skipping Vulkan Video decoder for unsupported local ROCm codec"
                );
            }
        }

        // Try VA-API (but NOT when gpu feature is enabled — loading VA-API's radeonsi
        // driver pulls in Mesa's LLVM, which conflicts with MIGraphX's bundled LLVM)
        #[cfg(not(feature = "gpu"))]
        match Self::open_vaapi_decoder(codec_params) {
            Ok((decoder, hw_ctx)) => {
                tracing::info!("using VA-API hardware decoder");
                return Ok(OpenDecoderResult {
                    decoder,
                    hw_backend: HwBackend::VaApi,
                    hw_ctx: Some(hw_ctx),
                });
            }
            Err(e) => {
                tracing::debug!("VA-API unavailable: {e}");
            }
        }
        #[cfg(feature = "gpu")]
        tracing::debug!("skipping VA-API (would conflict with MIGraphX LLVM)");

        // Fallback: CPU decoder with multithreading
        let mut ctx = ffmpeg_next::codec::Context::from_parameters(codec_params.clone())
            .map_err(|e| BekiperError::Data(format!("codec context: {e}")))?;
        ctx.set_threading(ffmpeg_next::threading::Config {
            kind: ffmpeg_next::threading::Type::Frame,
            count: 0,
        });
        let decoder = ctx
            .decoder()
            .video()
            .map_err(|e| BekiperError::Data(format!("video decoder: {e}")))?;

        tracing::info!(
            codec = ?codec_params.id(),
            "using multithreaded CPU decoder"
        );

        Ok(OpenDecoderResult {
            decoder,
            hw_backend: HwBackend::None,
            hw_ctx: None,
            #[cfg(feature = "gpu")]
            vk_ctx: None,
        })
    }

    #[cfg(not(feature = "gpu"))]
    fn open_vaapi_decoder(
        codec_params: &ffmpeg_next::codec::Parameters,
    ) -> std::result::Result<(ffmpeg_next::codec::decoder::Video, HwDeviceCtx), String> {
        // Create VA-API hardware device context
        let mut hw_device_ptr: *mut ffmpeg_next::ffi::AVBufferRef = std::ptr::null_mut();
        let ret = unsafe {
            ffmpeg_next::ffi::av_hwdevice_ctx_create(
                &mut hw_device_ptr,
                ffmpeg_next::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                std::ptr::null(), // default device (/dev/dri/renderD128)
                std::ptr::null_mut(),
                0,
            )
        };
        if ret < 0 || hw_device_ptr.is_null() {
            return Err(format!("av_hwdevice_ctx_create failed: {ret}"));
        }
        let hw_ctx = HwDeviceCtx { ptr: hw_device_ptr };

        // Create codec context and set hw_device_ctx before opening
        let mut ctx = ffmpeg_next::codec::Context::from_parameters(codec_params.clone())
            .map_err(|e| format!("codec context: {e}"))?;

        unsafe {
            // av_buffer_ref increments refcount; the codec context will own this ref
            let ref_ptr = ffmpeg_next::ffi::av_buffer_ref(hw_ctx.ptr);
            if ref_ptr.is_null() {
                return Err("av_buffer_ref failed".into());
            }
            (*ctx.as_mut_ptr()).hw_device_ctx = ref_ptr;
        }

        let decoder = ctx
            .decoder()
            .video()
            .map_err(|e| format!("vaapi decoder: {e}"))?;

        Ok((decoder, hw_ctx))
    }

    /// Open a Vulkan Video hardware decoder.
    ///
    /// Creates a `VulkanContext` with video decode extensions, then configures
    /// FFmpeg to use it via `AV_HWDEVICE_TYPE_VULKAN`.
    ///
    /// Uses a `get_format` callback to configure the frames context with
    /// export memory hints and single-plane NV12 layout.
    #[cfg(feature = "gpu")]
    fn open_vulkan_decoder(
        codec_params: &ffmpeg_next::codec::Parameters,
    ) -> std::result::Result<
        (
            ffmpeg_next::codec::decoder::Video,
            HwDeviceCtx,
            std::sync::Arc<VulkanContext>,
        ),
        String,
    > {
        use bekiper_gpu::ffmpeg_vk_sys::AVVulkanDeviceContext;

        // Create our Vulkan context with video decode + DMA-BUF extensions
        let vk_ctx = VulkanContext::new().map_err(|e| format!("VulkanContext::new: {e}"))?;
        let vk_ctx = std::sync::Arc::new(vk_ctx);

        // Allocate FFmpeg's Vulkan hardware device context
        let hw_device_buf = unsafe {
            ffmpeg_next::ffi::av_hwdevice_ctx_alloc(
                ffmpeg_next::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
            )
        };
        if hw_device_buf.is_null() {
            return Err("av_hwdevice_ctx_alloc(VULKAN) returned null".into());
        }

        // Fill in AVVulkanDeviceContext with our Vulkan handles
        unsafe {
            let hw_dev_ctx = (*hw_device_buf).data as *mut ffmpeg_next::ffi::AVHWDeviceContext;
            let vk_dev_ctx = (*hw_dev_ctx).hwctx as *mut AVVulkanDeviceContext;

            (*vk_dev_ctx).get_proc_addr = Some(vk_ctx.get_instance_proc_addr());
            (*vk_dev_ctx).inst = vk_ctx.raw_instance();
            (*vk_dev_ctx).phys_dev = vk_ctx.physical_device;
            (*vk_dev_ctx).act_dev = vk_ctx.raw_device();

            // Set up queue families
            (*vk_dev_ctx).queue_family_index = vk_ctx.graphics_queue_family as i32;
            (*vk_dev_ctx).nb_graphics_queues = 1;
            (*vk_dev_ctx).queue_family_tx_index = vk_ctx.transfer_queue_family as i32;
            (*vk_dev_ctx).nb_tx_queues = 1;
            (*vk_dev_ctx).queue_family_comp_index = vk_ctx.graphics_queue_family as i32;
            (*vk_dev_ctx).nb_comp_queues = 1;
            (*vk_dev_ctx).queue_family_decode_index = vk_ctx.decode_queue_family as i32;
            (*vk_dev_ctx).nb_decode_queues = 1;
            (*vk_dev_ctx).queue_family_encode_index = -1;
            (*vk_dev_ctx).nb_encode_queues = 0;

            // Set enabled extensions so FFmpeg knows what's available
            // (FFmpeg checks these to decide which features to use)
            let ext_ptrs: Vec<*const i8> = vk_ctx
                .enabled_device_extensions
                .iter()
                .map(|e: &std::ffi::CString| e.as_ptr())
                .collect();
            (*vk_dev_ctx).enabled_dev_extensions = ext_ptrs.as_ptr();
            (*vk_dev_ctx).nb_enabled_dev_extensions = ext_ptrs.len() as i32;

            // Initialize the device context
            let ret = ffmpeg_next::ffi::av_hwdevice_ctx_init(hw_device_buf);
            if ret < 0 {
                ffmpeg_next::ffi::av_buffer_unref(&mut (hw_device_buf as *mut _));
                return Err(format!("av_hwdevice_ctx_init(VULKAN) failed: {ret}"));
            }

            // Leak ext_ptrs intentionally - FFmpeg holds a reference to them.
            // They point into vk_ctx.enabled_device_extensions which lives
            // as long as the Arc<VulkanContext>.
            std::mem::forget(ext_ptrs);
        }

        let hw_ctx = HwDeviceCtx { ptr: hw_device_buf };

        // Create codec context and set hw_device_ctx
        let mut ctx = ffmpeg_next::codec::Context::from_parameters(codec_params.clone())
            .map_err(|e| format!("codec context: {e}"))?;

        unsafe {
            let ref_ptr = ffmpeg_next::ffi::av_buffer_ref(hw_ctx.ptr);
            if ref_ptr.is_null() {
                return Err("av_buffer_ref failed".into());
            }
            (*ctx.as_mut_ptr()).hw_device_ctx = ref_ptr;

            // Set get_format callback to configure DMA-BUF export on the
            // frames context created by the decoder. Without this,
            // vkGetMemoryFdKHR fails because memory wasn't allocated with
            // VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT.
            (*ctx.as_mut_ptr()).get_format = Some(vulkan_get_format);

            // Store the DMA-BUF export alloc info in opaque so get_format
            // can access it. This is leaked intentionally (lives as long
            // as the decoder).
            let export_info_ptr = bekiper_gpu::ffmpeg_vk_sys::alloc_export_memory_info_dma_buf();
            (*ctx.as_mut_ptr()).opaque = export_info_ptr;
        }

        let decoder = ctx
            .decoder()
            .video()
            .map_err(|e| format!("vulkan decoder: {e}"))?;

        Ok((decoder, hw_ctx, vk_ctx))
    }

    /// Video metadata.
    pub fn info(&self) -> &VideoInfo {
        &self.info
    }

    /// Convert a raw PTS value to seconds using the stream time base.
    pub(super) fn pts_to_seconds(&self, pts: i64) -> f64 {
        let (num, den) = self.info.time_base;
        pts as f64 * num as f64 / den as f64
    }

    /// Decode the next frame as a `DynamicImage`.
    /// Returns `None` at end of stream.
    pub fn next_frame(&mut self) -> Result<Option<(u64, DynamicImage)>> {
        Ok(self.next_frame_timed()?.map(|f| (f.index, f.image)))
    }
}
