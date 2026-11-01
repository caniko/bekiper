//! Tests for VideoDecoder seek correctness.
//!
//! Generates a tiny test video using ffmpeg's testsrc2 filter, where each frame
//! has a unique frame counter burned in. Tests that seek_to() lands on the
//! correct frame and that sequential decoding after seeking produces correct
//! frame indices.

#[cfg(feature = "video")]
mod video_seek_tests {
    use std::path::PathBuf;
    use std::process::Command;

    use bekiper_pose::video::VideoDecoder;

    /// Generate a short test video (100 frames at 25fps = 4 seconds).
    /// Uses ffmpeg's testsrc2 which renders a unique pattern per frame.
    fn generate_test_video(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bekiper_test_video_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.mp4");

        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=25:duration=4",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                // Short GOP for predictable keyframe placement
                "-g",
                "10",
                path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .expect("ffmpeg must be on PATH to run video tests");

        assert!(status.success(), "ffmpeg failed to generate test video");
        path
    }

    #[test]
    fn sequential_decode_produces_sequential_indices() {
        let path = generate_test_video("seq");
        let mut decoder = VideoDecoder::open(&path).unwrap();

        let info = decoder.info();
        assert_eq!(info.width, 320);
        assert_eq!(info.height, 240);
        assert!(
            info.total_frames >= 90,
            "expected ~100 frames, got {}",
            info.total_frames
        );

        // Decode first 20 frames and check indices
        for expected in 0..20u64 {
            let (idx, _img) = decoder.next_frame().unwrap().unwrap();
            assert_eq!(idx, expected, "frame index mismatch at position {expected}");
        }

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn seek_to_zero_is_noop() {
        let path = generate_test_video("seek0");
        let mut decoder = VideoDecoder::open(&path).unwrap();

        decoder.seek_to(0).unwrap();
        let (idx, _) = decoder.next_frame().unwrap().unwrap();
        assert_eq!(idx, 0);

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn seek_to_exact_frame() {
        let path = generate_test_video("seek_exact");
        let mut decoder = VideoDecoder::open(&path).unwrap();

        // Seek to frame 50 (should skip past keyframes at 0, 10, 20, 30, 40)
        decoder.seek_to(50).unwrap();
        let (idx, _) = decoder.next_frame().unwrap().unwrap();
        assert_eq!(idx, 50, "seek_to(50) should land on frame 50");

        // Next frame should be 51
        let (idx, _) = decoder.next_frame().unwrap().unwrap();
        assert_eq!(idx, 51);

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn seek_to_keyframe_boundary() {
        let path = generate_test_video("seek_kf");
        let mut decoder = VideoDecoder::open(&path).unwrap();

        // Seek to frame 30 (exactly on a keyframe with -g 10)
        decoder.seek_to(30).unwrap();
        let (idx, _) = decoder.next_frame().unwrap().unwrap();
        assert_eq!(idx, 30, "seek to keyframe boundary should be exact");

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn seek_then_batch_has_correct_indices() {
        let path = generate_test_video("seek_batch");
        let mut decoder = VideoDecoder::open(&path).unwrap();

        decoder.seek_to(40).unwrap();
        let batch = decoder.next_batch(5).unwrap();
        assert_eq!(batch.len(), 5);

        let indices: Vec<u64> = batch.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(indices, vec![40, 41, 42, 43, 44]);

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn seek_multiple_times() {
        let path = generate_test_video("seek_multi");
        let mut decoder = VideoDecoder::open(&path).unwrap();

        // First seek: frame 20
        decoder.seek_to(20).unwrap();
        let (idx, _) = decoder.next_frame().unwrap().unwrap();
        assert_eq!(idx, 20);

        // Second seek: frame 60 (forward)
        decoder.seek_to(60).unwrap();
        let (idx, _) = decoder.next_frame().unwrap().unwrap();
        assert_eq!(idx, 60);

        // Third seek: frame 10 (backward)
        decoder.seek_to(10).unwrap();
        let (idx, _) = decoder.next_frame().unwrap().unwrap();
        assert_eq!(idx, 10);

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn seek_with_resize_has_correct_frame() {
        let path = generate_test_video("seek_resize");
        let mut decoder = VideoDecoder::open_with_resize(&path, 64, 64).unwrap();

        decoder.seek_to(25).unwrap();
        let (idx, img) = decoder.next_frame().unwrap().unwrap();
        assert_eq!(idx, 25);
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn frame_at_returns_correct_frame() {
        let path = generate_test_video("frame_at");
        let mut decoder = VideoDecoder::open(&path).unwrap();

        // frame_at delegates to seek_to + next_frame
        let img = decoder.frame_at(35).unwrap();
        assert_eq!(img.width(), 320);
        assert_eq!(img.height(), 240);

        // After frame_at, sequential decoding should continue from 36
        let (idx, _) = decoder.next_frame().unwrap().unwrap();
        assert_eq!(idx, 36);

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn seek_pixel_data_differs_between_frames() {
        let path = generate_test_video("seek_pixels");
        let mut decoder = VideoDecoder::open(&path).unwrap();

        // Get frame 10 pixels
        decoder.seek_to(10).unwrap();
        let (_, img_10) = decoder.next_frame().unwrap().unwrap();
        let pixels_10: Vec<u8> = img_10.to_rgb8().into_raw();

        // Get frame 50 pixels — should be visually different (testsrc2 changes each frame)
        decoder.seek_to(50).unwrap();
        let (_, img_50) = decoder.next_frame().unwrap().unwrap();
        let pixels_50: Vec<u8> = img_50.to_rgb8().into_raw();

        assert_eq!(pixels_10.len(), pixels_50.len());
        assert_ne!(
            pixels_10, pixels_50,
            "frames 10 and 50 should have different pixel data"
        );

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
