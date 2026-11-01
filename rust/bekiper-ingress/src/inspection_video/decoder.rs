/// Decode through a bounded raw-RGB pipe. Nonblocking reads make cancellation
/// effective even when FFmpeg is stalled inside demuxing or decoding. The child
/// is killed/reaped before the caller releases its admitted workspace.
struct FfmpegDecoder {
    child: Child,
    stdout: std::process::ChildStdout,
    width: u32,
    height: u32,
    fps: f64,
    /// Container frame count (or duration × FPS) when the probe reports one.
    frame_count: Option<u64>,
    stderr_path: PathBuf,
}

impl Drop for FfmpegDecoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl FfmpegDecoder {
    fn start(source: &Path, output: &Path, cancelled: &dyn Fn() -> bool) -> Result<Self> {
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let metadata = output.with_extension("ffprobe.json");
        let stderr_path = output.with_extension("decode.stderr.log");
        let status = bekiper_post_analysis::pipeline::run_child(
            Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_frames,duration",
                    "-of",
                    "json",
                ])
                .arg(source),
            &metadata,
            &stderr_path,
            cancelled,
        )
        .map_err(|error| BekiperError::Data(error.to_string()))?;
        if !status.success() {
            return Err(BekiperError::Data(format!(
                "inspection video probe failed: {}",
                fs::read_to_string(&stderr_path)?
            )));
        }
        let spec: serde_json::Value = serde_json::from_slice(&fs::read(metadata)?)
            .map_err(|error| BekiperError::Data(error.to_string()))?;
        let stream = &spec["streams"][0];
        let dimension = |name| {
            stream[name]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .filter(|n| *n > 0)
                .ok_or_else(|| BekiperError::Data(format!("invalid source video {name}")))
        };
        let width = dimension("width")?;
        let height = dimension("height")?;
        let fps = stream["avg_frame_rate"]
            .as_str()
            .and_then(|rate| rate.split_once('/'))
            .and_then(|(n, d)| Some(n.parse::<f64>().ok()? / d.parse::<f64>().ok()?))
            .filter(|fps| fps.is_finite() && *fps > 0.)
            .ok_or_else(|| BekiperError::Data("source video needs finite positive FPS".into()))?;
        let probed = |name| stream[name].as_str().and_then(|v| v.parse::<f64>().ok());
        let frame_count = probed("nb_frames")
            .or_else(|| probed("duration").map(|seconds| (seconds * fps).round()))
            .filter(|n| n.is_finite() && *n >= 1.)
            .map(|n| n as u64);
        let mut child = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(source)
            .args([
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(fs::File::create(&stderr_path)?))
            .spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| BekiperError::Data("decoder stdout unavailable".into()))?;
        let decoder = Self {
            child,
            stdout,
            width,
            height,
            fps,
            frame_count,
            stderr_path,
        };
        #[cfg(unix)]
        {
            let flags = rustix::fs::fcntl_getfl(&decoder.stdout).map_err(std::io::Error::from)?;
            rustix::fs::fcntl_setfl(&decoder.stdout, flags | rustix::fs::OFlags::NONBLOCK)
                .map_err(std::io::Error::from)?;
        }
        Ok(decoder)
    }

    fn next_frame(&mut self, cancelled: &dyn Fn() -> bool) -> Result<Option<RgbImage>> {
        use std::io::Read;
        let size = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| BekiperError::Data("source video frame size overflow".into()))?;
        let mut bytes = vec![0; size];
        let mut offset = 0;
        while offset < size {
            if cancelled() {
                return Err(BekiperError::Other("inspection video cancelled".into()));
            }
            match self.stdout.read(&mut bytes[offset..]) {
                Ok(0) if offset != 0 => {
                    return Err(BekiperError::Data(
                        "decoder returned a truncated source frame".into(),
                    ));
                }
                Ok(0) => {
                    let status = loop {
                        if cancelled() {
                            return Err(BekiperError::Other("inspection video cancelled".into()));
                        }
                        if let Some(status) = self.child.try_wait()? {
                            break status;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    };
                    if !status.success() {
                        return Err(BekiperError::Data(format!(
                            "inspection video decode failed: {}",
                            fs::read_to_string(&self.stderr_path)?
                        )));
                    }
                    return Ok(None);
                }
                Ok(read) => offset += read,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    #[cfg(unix)]
                    wait_for_pipe(&self.stdout, rustix::event::PollFlags::IN)?;
                    #[cfg(not(unix))]
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(Some(
            RgbImage::from_raw(self.width, self.height, bytes)
                .ok_or_else(|| BekiperError::Data("invalid RGB frame".into()))?,
        ))
    }
}

#[cfg(all(test, unix))]
mod decoder_tests {
    use super::*;
    #[test]
    fn stalled_decoder_read_is_cancellable_and_reaped() {
        let mut child = Command::new("sleep")
            .arg("60")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let pid = child.id();
        let stdout = child.stdout.take().unwrap();
        let flags = rustix::fs::fcntl_getfl(&stdout).unwrap();
        rustix::fs::fcntl_setfl(&stdout, flags | rustix::fs::OFlags::NONBLOCK).unwrap();
        let mut decoder = FfmpegDecoder {
            child,
            stdout,
            width: 16,
            height: 16,
            fps: 30.,
            frame_count: None,
            stderr_path: PathBuf::new(),
        };
        let started = std::time::Instant::now();
        assert!(
            decoder
                .next_frame(&|| started.elapsed() >= std::time::Duration::from_millis(100))
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
        drop(decoder);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        #[cfg(target_os = "linux")]
        assert!(!Path::new(&format!("/proc/{pid}")).exists());
        #[cfg(not(target_os = "linux"))]
        let _ = pid;
    }
}
