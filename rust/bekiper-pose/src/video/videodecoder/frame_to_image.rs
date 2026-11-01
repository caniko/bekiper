use super::*;

impl VideoDecoder {
    pub(super) fn frame_to_image(
        &mut self,
        frame: &ffmpeg_next::frame::Video,
    ) -> Result<DynamicImage> {
        // If VA-API, transfer from GPU to CPU first
        let cpu_frame = if self.info.hw_accel {
            let mut sw_frame = ffmpeg_next::frame::Video::empty();
            let ret = unsafe {
                ffmpeg_next::ffi::av_hwframe_transfer_data(sw_frame.as_mut_ptr(), frame.as_ptr(), 0)
            };
            if ret < 0 {
                return Err(BekiperError::Data(format!(
                    "hw frame transfer failed: {ret}"
                )));
            }
            sw_frame
        } else {
            // Clone not needed for CPU frames — but we need an owned frame for the scaler
            frame.clone()
        };

        let mut rgb_frame = ffmpeg_next::frame::Video::empty();
        self.scaler
            .run(&cpu_frame, &mut rgb_frame)
            .map_err(|e| BekiperError::Data(format!("scale frame: {e}")))?;

        let width = rgb_frame.width();
        let height = rgb_frame.height();
        let data = rgb_frame.data(0);
        let stride = rgb_frame.stride(0);

        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height as usize {
            let row_start = y * stride;
            let row_end = row_start + (width as usize * 3);
            pixels.extend_from_slice(&data[row_start..row_end]);
        }

        let img = RgbImage::from_raw(width, height, pixels)
            .ok_or_else(|| BekiperError::Data("failed to create image from frame data".into()))?;

        Ok(DynamicImage::ImageRgb8(img))
    }
}
