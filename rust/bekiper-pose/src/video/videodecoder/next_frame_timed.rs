use super::*;

impl VideoDecoder {
    #[cfg(not(feature = "gpu"))]
    pub(super) fn reopen_as_cpu(&mut self) -> Result<()> {
        Err(BekiperError::Data(
            "hardware decode fallback is only available with the gpu feature".into(),
        ))
    }

    /// Decode the next frame with its presentation timestamp.
    /// Returns `None` at end of stream.
    pub fn next_frame_timed(&mut self) -> Result<Option<DecodedFrame>> {
        // Return cached frame from seek overshoot before decoding new ones
        if let Some(cached_img) = self.cached_frame.take() {
            let ts = self
                .cached_timestamp
                .take()
                .unwrap_or(self.last_pts_seconds);
            return Ok(Some(DecodedFrame {
                index: cached_img.0,
                image: cached_img.1,
                timestamp_seconds: ts,
            }));
        }

        loop {
            let Some(decoded) = self.receive_decoded_frame()? else {
                return Ok(None);
            };
            let ts = self.frame_timestamp(&decoded);
            let image = match self.frame_to_image(&decoded) {
                Ok(image) => image,
                Err(_) if self.hw_backend != HwBackend::None => {
                    self.reopen_as_cpu()?;
                    continue;
                }
                Err(err) => return Err(err),
            };
            let idx = self.frame_index;
            self.frame_index += 1;
            return Ok(Some(DecodedFrame {
                index: idx,
                image,
                timestamp_seconds: ts,
            }));
        }
    }

    pub(super) fn frame_timestamp(&mut self, decoded: &ffmpeg_next::frame::Video) -> f64 {
        let ts = decoded
            .pts()
            .map(|pts| self.pts_to_seconds(pts))
            .unwrap_or(self.frame_index as f64 / self.info.fps);
        self.last_pts_seconds = ts;
        ts
    }

    /// Feed packets until a decoded frame is available, including the final
    /// decoder drain. Image conversion and hardware surface ownership remain
    /// with the caller.
    pub(super) fn receive_decoded_frame(&mut self) -> Result<Option<ffmpeg_next::frame::Video>> {
        let mut decoded = ffmpeg_next::frame::Video::empty();
        loop {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => return Ok(Some(decoded)),
                Err(ffmpeg_next::Error::Other { errno }) if errno == ffmpeg_next::error::EAGAIN => {
                    if !self.send_next_packet()? {
                        self.decoder
                            .send_eof()
                            .map_err(|e| BekiperError::Data(format!("send eof: {e}")))?;
                        match self.decoder.receive_frame(&mut decoded) {
                            Ok(()) => return Ok(Some(decoded)),
                            Err(_) => return Ok(None),
                        }
                    }
                }
                Err(ffmpeg_next::Error::Eof) => return Ok(None),
                Err(e) => return Err(BekiperError::Data(format!("decode frame: {e}"))),
            }
        }
    }

    /// Decode frames in batches for efficient inference.
    /// Returns up to `batch_size` frames. Empty vec means end of stream.
    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<(u64, DynamicImage)>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match self.next_frame()? {
                Some(frame) => batch.push(frame),
                None => break,
            }
        }
        Ok(batch)
    }

    /// Decode frames in batches with timestamps.
    pub fn next_batch_timed(&mut self, batch_size: usize) -> Result<Vec<DecodedFrame>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match self.next_frame_timed()? {
                Some(frame) => batch.push(frame),
                None => break,
            }
        }
        Ok(batch)
    }

    /// Seek to a frame index so that the next `next_frame()` / `next_batch()`
    /// call returns frames starting from `target_frame`.
    ///
    /// ffmpeg seek lands on the nearest **keyframe before** the target (up to
    /// one GOP interval, typically 30-250 frames). This method decodes and
    /// discards the gap frames (raw decode only, no swscale conversion) so the
    /// next call to `next_frame()` returns exactly `target_frame` with correct
    /// pixel data.
    pub fn seek_to(&mut self, target_frame: u64) -> Result<()> {
        if target_frame == 0 {
            return Ok(());
        }

        let stream = self
            .input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| BekiperError::Data("no video stream".into()))?;

        let time_base = stream.time_base();
        let fps = f64::from(stream.avg_frame_rate());
        let timestamp = (target_frame as f64 / fps * time_base.denominator() as f64
            / time_base.numerator() as f64) as i64;

        self.input
            .seek(timestamp, ..timestamp)
            .map_err(|e| BekiperError::Data(format!("seek: {e}")))?;

        self.decoder.flush();
        self.cached_frame = None;
        self.cached_timestamp = None;

        // Decode frames until we reach target_frame. We use PTS to determine
        // the actual frame number of each decoded frame.
        let tb_num = time_base.numerator() as f64;
        let tb_den = time_base.denominator() as f64;
        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut skipped = 0u64;

        loop {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let frame_num = decoded
                        .pts()
                        .map(|pts| (pts as f64 * tb_num / tb_den * fps).round() as u64)
                        .unwrap_or(0);

                    if frame_num >= target_frame {
                        // We've reached or passed the target. Convert this frame
                        // to an image and cache it so next_frame() returns it.
                        let ts = decoded
                            .pts()
                            .map(|pts| self.pts_to_seconds(pts))
                            .unwrap_or(frame_num as f64 / self.info.fps);
                        match self.frame_to_image(&decoded) {
                            Ok(image) => {
                                self.frame_index = frame_num + 1;
                                self.cached_frame = Some((frame_num, image));
                                self.cached_timestamp = Some(ts);
                                self.last_pts_seconds = ts;
                            }
                            Err(_) if self.hw_backend != HwBackend::None => {
                                // Hw transfer failed (e.g., Vulkan Video doesn't
                                // support this codec profile). Reopen as CPU and
                                // redo the seek.
                                #[cfg(feature = "gpu")]
                                {
                                    self.reopen_as_cpu()?;
                                    // reopen_as_cpu already seeked back and positioned
                                    // at current_frame. We're done.
                                    return Ok(());
                                }
                                #[cfg(not(feature = "gpu"))]
                                {
                                    return Err(BekiperError::Data(
                                        "hw frame transfer failed during seek".into(),
                                    ));
                                }
                            }
                            Err(e) => return Err(e),
                        }

                        tracing::debug!(
                            target_frame,
                            landed_at = frame_num,
                            skipped,
                            "seek complete"
                        );
                        return Ok(());
                    }

                    // Before target — discard (no swscale conversion)
                    skipped += 1;
                }
                Err(ffmpeg_next::Error::Other { errno }) if errno == ffmpeg_next::error::EAGAIN => {
                    if !self.send_next_packet()? {
                        // EOF before reaching target
                        self.frame_index = target_frame;
                        return Ok(());
                    }
                }
                Err(ffmpeg_next::Error::Eof) => {
                    self.frame_index = target_frame;
                    return Ok(());
                }
                Err(e) => {
                    return Err(BekiperError::Data(format!("decode during seek: {e}")));
                }
            }
        }
    }

    /// Extract a single frame at a specific index (by seeking).
    pub fn frame_at(&mut self, frame_index: u64) -> Result<DynamicImage> {
        self.seek_to(frame_index)?;
        self.next_frame()?
            .map(|(_, img)| img)
            .ok_or_else(|| BekiperError::Data(format!("no frame at index {frame_index}")))
    }

    /// Whether this decoder is using hardware acceleration.
    pub fn is_hw_accel(&self) -> bool {
        self.info.hw_accel
    }

    /// Which hardware decode backend is active.
    pub fn hw_backend(&self) -> HwBackend {
        self.hw_backend
    }

    /// Get the Vulkan context (if using Vulkan Video decode).
    #[cfg(feature = "gpu")]
    pub fn vk_context(&self) -> Option<&std::sync::Arc<VulkanContext>> {
        self._vk_ctx.as_ref()
    }

    /// Get the VA display handle for zero-copy GPU import.
    ///
    /// Returns `None` if not using VA-API hardware decoding.
    /// The returned pointer is valid for the lifetime of this `VideoDecoder`.
    pub fn va_display(&self) -> Option<*mut c_void> {
        if self.hw_backend != HwBackend::VaApi {
            return None;
        }
        let hw_ctx = self._hw_device_ctx.as_ref()?;
        unsafe {
            // AVBufferRef.data → AVHWDeviceContext.hwctx → AVVAAPIDeviceContext
            // AVVAAPIDeviceContext's first field is `VADisplay display` (*mut c_void).
            let buf_ref = &*hw_ctx.ptr;
            let hw_dev_ctx = buf_ref.data as *mut ffmpeg_next::ffi::AVHWDeviceContext;
            let va_display = *((*hw_dev_ctx).hwctx as *const *mut c_void);
            Some(va_display)
        }
    }

    /// Decode the next frame, keeping it on the GPU as a VA-API surface.
    ///
    /// Returns a `RawVaFrame` containing the VA surface ID and display handle,
    /// ready for zero-copy import into HIP via `GpuFrame::from_va_surface`.
    ///
    /// Returns `None` at end of stream. Returns an error if the decoder is not
    /// using VA-API hardware acceleration.
    ///
    /// **Important**: The VA surface is only valid while the returned `RawVaFrame`
    /// is alive. Import it into HIP and preprocess before dropping.
    pub fn gpu_next_frame(&mut self) -> Result<Option<RawVaFrame>> {
        if !self.info.hw_accel {
            return Err(BekiperError::Data(
                "gpu_next_frame requires VA-API hardware decoding".into(),
            ));
        }

        let va_display = self
            .va_display()
            .ok_or_else(|| BekiperError::Data("VA display not available".into()))?;

        let mut decoded = ffmpeg_next::frame::Video::empty();

        loop {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let ts = decoded
                        .pts()
                        .map(|pts| self.pts_to_seconds(pts))
                        .unwrap_or(self.frame_index as f64 / self.info.fps);
                    self.last_pts_seconds = ts;

                    // Extract VASurfaceID from frame->data[3]
                    let surface_id = unsafe { (*decoded.as_ptr()).data[3] as usize as u32 };

                    let idx = self.frame_index;
                    self.frame_index += 1;

                    return Ok(Some(RawVaFrame {
                        surface_id,
                        va_display,
                        width: self.info.width,
                        height: self.info.height,
                        index: idx,
                        timestamp_seconds: ts,
                        _frame: decoded,
                    }));
                }
                Err(ffmpeg_next::Error::Other { errno }) if errno == ffmpeg_next::error::EAGAIN => {
                    if !self.send_next_packet()? {
                        self.decoder
                            .send_eof()
                            .map_err(|e| BekiperError::Data(format!("send eof: {e}")))?;
                        match self.decoder.receive_frame(&mut decoded) {
                            Ok(()) => {
                                let ts = decoded
                                    .pts()
                                    .map(|pts| self.pts_to_seconds(pts))
                                    .unwrap_or(self.frame_index as f64 / self.info.fps);
                                self.last_pts_seconds = ts;

                                let surface_id =
                                    unsafe { (*decoded.as_ptr()).data[3] as usize as u32 };

                                let idx = self.frame_index;
                                self.frame_index += 1;

                                return Ok(Some(RawVaFrame {
                                    surface_id,
                                    va_display,
                                    width: self.info.width,
                                    height: self.info.height,
                                    index: idx,
                                    timestamp_seconds: ts,
                                    _frame: decoded,
                                }));
                            }
                            Err(_) => return Ok(None),
                        }
                    }
                }
                Err(ffmpeg_next::Error::Eof) => return Ok(None),
                Err(e) => return Err(BekiperError::Data(format!("decode frame: {e}"))),
            }
        }
    }

    /// Decode GPU frames in batches. Returns up to `batch_size` frames.
    /// Empty vec means end of stream.
    pub fn gpu_next_batch(&mut self, batch_size: usize) -> Result<Vec<RawVaFrame>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match self.gpu_next_frame()? {
                Some(frame) => batch.push(frame),
                None => break,
            }
        }
        Ok(batch)
    }
}
