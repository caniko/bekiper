/// Wake immediately when a raw-RGB pipe can progress. The bounded wait lets
/// callers check cancellation even when FFmpeg stops consuming/producing data.
#[cfg(unix)]
fn wait_for_pipe(fd: &impl std::os::fd::AsFd, events: rustix::event::PollFlags) -> Result<()> {
    let timeout = rustix::event::Timespec {
        tv_sec: 0,
        tv_nsec: 10_000_000,
    };
    match rustix::event::poll(
        &mut [rustix::event::PollFd::new(fd, events)],
        Some(&timeout),
    ) {
        Ok(_) | Err(rustix::io::Errno::INTR) => Ok(()),
        Err(error) => Err(std::io::Error::from(error).into()),
    }
}

fn faststart_container(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "mp4" | "m4v" | "mov"))
}

struct FfmpegEncoder {
    child: Child,
    stdin: Option<ChildStdin>,
    output_path: PathBuf,
    stderr_path: PathBuf,
}

impl Drop for FfmpegEncoder {
    fn drop(&mut self) {
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl FfmpegEncoder {
    fn start(
        output_path: &Path,
        codec: &str,
        preset: &str,
        crf: Option<u8>,
        width: u32,
        height: u32,
        fps: f64,
    ) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let encoder = map_codec(codec);
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y")
            .arg("-loglevel")
            .arg("error")
            .arg("-f")
            .arg("rawvideo")
            .arg("-pix_fmt")
            .arg("rgb24")
            .arg("-s:v")
            .arg(format!("{width}x{height}"))
            .arg("-r")
            .arg(format!("{fps:.6}"))
            .arg("-i")
            .arg("pipe:0")
            .arg("-an")
            .arg("-c:v")
            .arg(encoder);

        if encoder_supports_preset(encoder) {
            cmd.arg("-preset").arg(preset);
        }
        if let Some(crf) = crf {
            cmd.arg("-crf").arg(crf.to_string());
        }
        if faststart_container(output_path) {
            // Browsers can start playback before the whole file has downloaded.
            cmd.arg("-movflags").arg("+faststart");
        }

        let stderr_path = output_path.with_extension("ffmpeg.stderr.log");
        cmd.arg("-pix_fmt")
            .arg("yuv420p")
            .arg(output_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::from(fs::File::create(&stderr_path)?));

        let mut child = cmd.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| BekiperError::Other("failed to open ffmpeg stdin".into()))?;

        #[cfg(unix)]
        {
            let flags = rustix::fs::fcntl_getfl(&stdin).map_err(std::io::Error::from)?;
            rustix::fs::fcntl_setfl(&stdin, flags | rustix::fs::OFlags::NONBLOCK)
                .map_err(std::io::Error::from)?;
        }

        Ok(Self {
            child,
            stdin: Some(stdin),
            output_path: output_path.to_path_buf(),
            stderr_path,
        })
    }

    fn write_frame(&mut self, frame: &RgbImage, cancelled: &dyn Fn() -> bool) -> Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| BekiperError::Other("ffmpeg stdin already closed".into()))?;
        let mut bytes = frame.as_raw().as_slice();
        while !bytes.is_empty() {
            if cancelled() {
                return Err(BekiperError::Other("inspection video cancelled".into()));
            }
            match stdin.write(bytes) {
                Ok(0) => return Err(std::io::Error::from(std::io::ErrorKind::WriteZero).into()),
                Ok(written) => bytes = &bytes[written..],
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    #[cfg(unix)]
                    wait_for_pipe(stdin, rustix::event::PollFlags::OUT)?;
                    #[cfg(not(unix))]
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    fn finish(mut self, cancelled: &dyn Fn() -> bool) -> Result<()> {
        self.stdin.take();
        let status = loop {
            if cancelled() {
                return Err(BekiperError::Other("inspection video cancelled".into()));
            }
            if let Some(status) = self.child.try_wait()? {
                break status;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        };
        if status.success() {
            return Ok(());
        }
        let stderr = fs::read_to_string(&self.stderr_path)?;
        Err(BekiperError::Other(format!(
            "ffmpeg failed for {}: {}",
            self.output_path.display(),
            stderr.trim()
        )))
    }
}
