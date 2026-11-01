use super::*;

impl VideoDecoder {
    /// Decode the next frame, keeping it on the GPU as a Vulkan Video surface.
    ///
    /// Returns a `RawVkFrame` containing a pointer to the AVVkFrame.
    /// Prefer `gpu_next_nv12_host` for the complete decode → host transfer path.
    ///
    /// Returns `None` at end of stream. Returns an error if the decoder is not
    /// using Vulkan Video hardware acceleration.
    #[cfg(feature = "gpu")]
    pub fn gpu_next_frame_vk(&mut self) -> Result<Option<RawVkFrame>> {
        if self.hw_backend != HwBackend::Vulkan {
            return Err(BekiperError::Data(
                "gpu_next_frame_vk requires Vulkan Video hardware decoding".into(),
            ));
        }

        let Some(decoded) = self.receive_decoded_frame()? else {
            return Ok(None);
        };
        let ts = self.frame_timestamp(&decoded);
        // AVVkFrame* is stored at frame->data[0] for Vulkan hwaccel.
        let vk_frame =
            unsafe { (*decoded.as_ptr()).data[0] as *const bekiper_gpu::ffmpeg_vk_sys::AVVkFrame };
        let idx = self.frame_index;
        self.frame_index += 1;
        Ok(Some(RawVkFrame {
            vk_frame,
            width: self.info.width,
            height: self.info.height,
            index: idx,
            timestamp_seconds: ts,
            _frame: decoded,
        }))
    }

    /// Decode Vulkan GPU frames in batches. Returns up to `batch_size` frames.
    /// Empty vec means end of stream.
    #[cfg(feature = "gpu")]
    pub fn gpu_next_batch_vk(&mut self, batch_size: usize) -> Result<Vec<RawVkFrame>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match self.gpu_next_frame_vk()? {
                Some(frame) => batch.push(frame),
                None => break,
            }
        }
        Ok(batch)
    }

    /// Re-open the decoder as CPU-only (no hw acceleration).
    ///
    /// Called when Vulkan Video decoding fails for a specific stream (e.g., the
    /// H.264 profile isn't supported by the hardware). Preserves the current
    /// frame position and re-creates the decoder with multithreaded CPU decoding.
    #[cfg(feature = "gpu")]
    pub(super) fn reopen_as_cpu(&mut self) -> Result<()> {
        let current_frame = self.frame_index;

        // Get codec parameters from the video stream
        let stream = self
            .input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| BekiperError::Data("no video stream".into()))?;
        let codec_params = stream.parameters();

        // Create a new CPU decoder with multithreading
        let mut ctx = ffmpeg_next::codec::Context::from_parameters(codec_params)
            .map_err(|e| BekiperError::Data(format!("cpu codec context: {e}")))?;
        ctx.set_threading(ffmpeg_next::threading::Config {
            kind: ffmpeg_next::threading::Type::Frame,
            count: 0,
        });
        let new_decoder = ctx
            .decoder()
            .video()
            .map_err(|e| BekiperError::Data(format!("cpu video decoder: {e}")))?;

        // Update scaler for the new pixel format (CPU decoder outputs YUV420P typically)
        let src_format = new_decoder.format();
        let (out_w, out_h) = (self.info.output_width, self.info.output_height);
        self.scaler = ffmpeg_next::software::scaling::Context::get(
            src_format,
            self.info.width,
            self.info.height,
            ffmpeg_next::format::Pixel::RGB24,
            out_w,
            out_h,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| BekiperError::Data(format!("scaler reinit: {e}")))?;

        // Replace decoder and drop hw resources
        self.decoder = new_decoder;
        self.hw_backend = HwBackend::None;
        self.info.hw_accel = false;
        self._hw_device_ctx = None;
        self._vk_ctx = None;

        tracing::warn!(
            frame = current_frame,
            "reopened decoder as CPU (Vulkan Video failed for this stream)"
        );

        // Seek back to where we were. This is required even for frame 0:
        // the failed hardware decode path has already consumed packets from
        // the demuxer, so a fresh CPU decoder must rewind to a stable start.
        self.frame_index = 0;
        self.cached_frame = None;
        self.cached_timestamp = None;

        let stream = self
            .input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| BekiperError::Data("no video stream".into()))?;
        let time_base = stream.time_base();
        let fps = f64::from(stream.avg_frame_rate());
        let timestamp = (current_frame as f64 / fps * time_base.denominator() as f64
            / time_base.numerator() as f64) as i64;

        self.input
            .seek(timestamp, ..timestamp)
            .map_err(|e| BekiperError::Data(format!("seek after cpu reopen: {e}")))?;
        self.decoder.flush();

        // Discard frames until we reach current_frame.
        let tb_num = time_base.numerator() as f64;
        let tb_den = time_base.denominator() as f64;
        let mut decoded = ffmpeg_next::frame::Video::empty();

        loop {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let frame_num = decoded
                        .pts()
                        .map(|pts| (pts as f64 * tb_num / tb_den * fps).round() as u64)
                        .unwrap_or(0);
                    if frame_num >= current_frame {
                        self.frame_index = current_frame;
                        break;
                    }
                }
                Err(ffmpeg_next::Error::Other { errno }) if errno == ffmpeg_next::error::EAGAIN => {
                    if !self.send_next_packet()? {
                        self.frame_index = current_frame;
                        break;
                    }
                }
                Err(_) => {
                    self.frame_index = current_frame;
                    break;
                }
            }
        }

        Ok(())
    }

    /// Decode the next frame via Vulkan Video (or CPU fallback), then transfer to host as NV12.
    ///
    /// Uses `av_hwframe_transfer_data` to copy the Vulkan-decoded frame to CPU
    /// memory. The resulting `RawNv12Frame` can be uploaded to HIP via
    /// `GpuFrame::from_nv12_host`.
    ///
    /// If Vulkan Video fails for a specific stream, automatically re-opens the
    /// decoder as CPU and continues decoding.
    #[cfg(feature = "gpu")]
    pub fn gpu_next_nv12_host(&mut self) -> Result<Option<RawNv12Frame>> {
        if self.hw_backend != HwBackend::Vulkan && self.hw_backend != HwBackend::None {
            return Err(BekiperError::Data(
                "gpu_next_nv12_host requires Vulkan Video or CPU decoding".into(),
            ));
        }

        let mut decoded = ffmpeg_next::frame::Video::empty();

        loop {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    match self.transfer_vk_frame_to_nv12(&decoded)? {
                        Some(frame) => return Ok(Some(frame)),
                        None => {
                            // transfer_vk_frame_to_nv12 returned None after reopening
                            // as CPU — the decoder has been seeked back, continue the
                            // loop to decode with the new CPU decoder.
                            decoded = ffmpeg_next::frame::Video::empty();
                            continue;
                        }
                    }
                }
                Err(ffmpeg_next::Error::Other { errno }) if errno == ffmpeg_next::error::EAGAIN => {
                    if !self.send_next_packet()? {
                        self.decoder
                            .send_eof()
                            .map_err(|e| BekiperError::Data(format!("send eof: {e}")))?;
                        match self.decoder.receive_frame(&mut decoded) {
                            Ok(()) => match self.transfer_vk_frame_to_nv12(&decoded)? {
                                Some(frame) => return Ok(Some(frame)),
                                None => {
                                    decoded = ffmpeg_next::frame::Video::empty();
                                    continue;
                                }
                            },
                            Err(_) => return Ok(None),
                        }
                    }
                }
                Err(ffmpeg_next::Error::Eof) => return Ok(None),
                Err(e) => return Err(BekiperError::Data(format!("decode frame: {e}"))),
            }
        }
    }

    /// Transfer a decoded frame (hw or sw) to CPU NV12 and package as RawNv12Frame.
    ///
    /// If the frame is a Vulkan hardware frame, uses `av_hwframe_transfer_data`.
    /// If the frame is already a software frame (e.g. FFmpeg fell back to CPU decode
    /// when Vulkan Video doesn't support this codec profile), extracts planes directly.
    #[cfg(feature = "gpu")]
    fn transfer_vk_frame_to_nv12(
        &mut self,
        decoded: &ffmpeg_next::frame::Video,
    ) -> Result<Option<RawNv12Frame>> {
        use ffmpeg_next::ffi::AVPixelFormat;

        let ts = decoded
            .pts()
            .map(|pts| self.pts_to_seconds(pts))
            .unwrap_or(self.frame_index as f64 / self.info.fps);
        self.last_pts_seconds = ts;

        let format = unsafe { (*decoded.as_ptr()).format };
        let is_hw_frame = format == AVPixelFormat::AV_PIX_FMT_VULKAN as i32;

        let (width, height, y_pitch, nv12) = if is_hw_frame {
            // Vulkan hw frame → transfer to CPU via av_hwframe_transfer_data
            let mut sw_frame = ffmpeg_next::frame::Video::empty();
            let ret = unsafe {
                ffmpeg_next::ffi::av_hwframe_transfer_data(
                    sw_frame.as_mut_ptr(),
                    decoded.as_ptr(),
                    0,
                )
            };
            if ret < 0 {
                // Vulkan Video can't decode this stream (unsupported profile).
                // Re-open as CPU decoder and signal caller to retry.
                tracing::warn!(ret, "hw frame transfer failed, switching to CPU decode");
                self.reopen_as_cpu()?;
                // Return None to signal the caller to retry with the new decoder
                return Ok(None);
            }
            let w = sw_frame.width();
            let h = sw_frame.height();
            let pitch = sw_frame.stride(0) as u32;
            let y_data = sw_frame.data(0);
            let uv_data = sw_frame.data(1);
            let y_size = (pitch * h) as usize;
            let uv_size = (pitch * h / 2) as usize;
            let mut nv12 = Vec::with_capacity(y_size + uv_size);
            nv12.extend_from_slice(&y_data[..y_size]);
            nv12.extend_from_slice(&uv_data[..uv_size]);
            (w, h, pitch, nv12)
        } else if format == AVPixelFormat::AV_PIX_FMT_NV12 as i32 {
            // Already NV12 software frame — extract planes directly
            let w = decoded.width();
            let h = decoded.height();
            let pitch = decoded.stride(0) as u32;
            let y_data = decoded.data(0);
            let uv_data = decoded.data(1);
            let y_size = (pitch * h) as usize;
            let uv_size = (pitch * h / 2) as usize;
            let mut nv12 = Vec::with_capacity(y_size + uv_size);
            nv12.extend_from_slice(&y_data[..y_size]);
            nv12.extend_from_slice(&uv_data[..uv_size]);
            (w, h, pitch, nv12)
        } else if format == AVPixelFormat::AV_PIX_FMT_YUV420P as i32 {
            // YUV420P (planar) → convert to NV12 (interleave U+V into UV plane)
            let w = decoded.width();
            let h = decoded.height();
            let y_pitch = decoded.stride(0) as u32;
            let u_pitch = decoded.stride(1) as u32;
            let y_data = decoded.data(0);
            let u_data = decoded.data(1);
            let v_data = decoded.data(2);
            let y_size = (y_pitch * h) as usize;
            let uv_h = h / 2;
            let uv_w = w / 2;
            // NV12 UV plane has same pitch as Y
            let uv_size = (y_pitch * uv_h) as usize;
            let mut nv12 = Vec::with_capacity(y_size + uv_size);
            // Y plane
            nv12.extend_from_slice(&y_data[..y_size]);
            // Interleave U and V into NV12 UV plane
            for row in 0..uv_h {
                let u_row_start = (row * u_pitch) as usize;
                let v_row_start = (row * u_pitch) as usize;
                for col in 0..uv_w {
                    nv12.push(u_data[u_row_start + col as usize]);
                    nv12.push(v_data[v_row_start + col as usize]);
                }
                // Pad to y_pitch if needed
                let written = (uv_w * 2) as usize;
                let pad_bytes = (y_pitch as usize).saturating_sub(written);
                nv12.extend(std::iter::repeat_n(0u8, pad_bytes));
            }
            (w, h, y_pitch, nv12)
        } else {
            return Err(BekiperError::Data(format!(
                "unsupported pixel format for NV12 conversion: {format}"
            )));
        };

        let idx = self.frame_index;
        self.frame_index += 1;

        Ok(Some(RawNv12Frame {
            data: nv12,
            width,
            height,
            y_pitch,
            index: idx,
            timestamp_seconds: ts,
        }))
    }

    /// Decode a batch of frames via Vulkan Video, transferred to host as NV12.
    /// Returns up to `batch_size` frames. Empty vec means end of stream.
    #[cfg(feature = "gpu")]
    pub fn gpu_next_nv12_host_batch(&mut self, batch_size: usize) -> Result<Vec<RawNv12Frame>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match self.gpu_next_nv12_host()? {
                Some(frame) => batch.push(frame),
                None => break,
            }
        }
        Ok(batch)
    }

    pub(super) fn send_next_packet(&mut self) -> Result<bool> {
        for (stream, packet) in self.input.packets() {
            if stream.index() == self.stream_index {
                self.decoder
                    .send_packet(&packet)
                    .map_err(|e| BekiperError::Data(format!("send packet: {e}")))?;
                return Ok(true);
            }
        }
        Ok(false)
    }
}
