//! Cancellable child lifetime: logs never block a pipe and exit is reaped before
//! the caller releases its workspace lease or reports terminal status.
use anyhow::{Context, Result, bail};
use std::{
    fs::File,
    path::Path,
    process::{Child, Command, ExitStatus, Stdio},
    time::Duration,
};

struct Reap {
    child: Child,
    reaped: bool,
}
impl Drop for Reap {
    fn drop(&mut self) {
        if !self.reaped {
            #[cfg(unix)]
            if let Some(pid) = rustix::process::Pid::from_raw(self.child.id() as i32) {
                let _ = rustix::process::kill_process_group(pid, rustix::process::Signal::KILL);
            }
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

pub fn run_child(
    command: &mut Command,
    stdout: &Path,
    stderr: &Path,
    cancelled: &dyn Fn() -> bool,
) -> Result<ExitStatus> {
    if cancelled() {
        bail!("rendering cancelled");
    }
    command
        .stdout(Stdio::from(File::create(stdout)?))
        .stderr(Stdio::from(File::create(stderr)?));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = Reap {
        child: command.spawn().context("failed to start renderer")?,
        reaped: false,
    };
    loop {
        if cancelled() {
            bail!("rendering cancelled");
        }
        if let Some(status) = child.child.try_wait()? {
            child.reaped = true;
            return Ok(status);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn cancellation_reaps_the_child_before_returning() {
        let dir = tempfile::tempdir().unwrap();
        let stdout = dir.path().join("stdout");
        let calls = AtomicUsize::new(0);
        let result = run_child(
            Command::new("sh").args(["-c", "echo $$; exec sleep 60"]),
            &stdout,
            &dir.path().join("stderr"),
            &|| calls.fetch_add(1, Ordering::SeqCst) >= 4,
        );
        assert!(result.unwrap_err().to_string().contains("cancelled"));
        let pid = std::fs::read_to_string(stdout).unwrap();
        assert!(!Path::new(&format!("/proc/{}", pid.trim())).exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cancellation_terminates_renderer_descendants() {
        let dir = tempfile::tempdir().unwrap();
        let stdout = dir.path().join("stdout");
        let calls = AtomicUsize::new(0);
        let result = run_child(
            Command::new("sh").args(["-c", "sleep 60 & echo $!; wait"]),
            &stdout,
            &dir.path().join("stderr"),
            &|| calls.fetch_add(1, Ordering::SeqCst) >= 4,
        );
        assert!(result.is_err());
        let pid = std::fs::read_to_string(stdout).unwrap();
        // Group SIGKILL is asynchronous: reaping the direct child need not
        // schedule its descendant first. Bound the signal-delivery observation
        // rather than racing a single /proc read against kernel termination.
        let running = |pid: &str| {
            std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
                stat.split_once(") ")
                    .is_some_and(|(_, fields)| !fields.starts_with('Z'))
            })
        };
        assert!(running(&std::process::id().to_string()));
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let mut alive = true;
        while alive && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
            alive = running(pid.trim());
        }
        assert!(!alive, "renderer descendant survived cancellation");
    }
}
