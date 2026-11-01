use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use bekiper_core::config::RuntimeSettings;
use bekiper_core::error::{BekiperError, Result};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

const DEFAULT_MAX_RAM_USAGE_FRACTION: f64 = 0.80;
const RAM_USAGE_RESUME_HYSTERESIS: f64 = 0.05;
const DEFAULT_MEMORY_POLL_INTERVAL: Duration = Duration::from_millis(500);
const MEMINFO_PATH: &str = "/proc/meminfo";

type MemoryUsageProvider = Arc<dyn Fn() -> std::result::Result<f64, String> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemorySource {
    Disabled,
    ProcMeminfo,
    ThreadCapOnly,
}

impl MemorySource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::ProcMeminfo => "proc_meminfo",
            Self::ThreadCapOnly => "thread_cap_only",
        }
    }
}

#[derive(Debug)]
struct GateState {
    active_tasks: usize,
    throttled: bool,
    throttle_logged: bool,
    memory_scheduler_active: bool,
    memory_source: MemorySource,
    provider_failure_logged: bool,
}

struct AdmissionGateInner {
    state: Mutex<GateState>,
    condvar: Condvar,
    memory_provider: MemoryUsageProvider,
    max_ram_usage_fraction: f64,
    resume_ram_usage_fraction: f64,
    poll_interval: Duration,
}

#[derive(Clone)]
struct AdmissionGate {
    inner: Arc<AdmissionGateInner>,
}

impl AdmissionGate {
    fn new(
        memory_scheduler_enabled: bool,
        max_ram_usage_fraction: f64,
        memory_provider: MemoryUsageProvider,
    ) -> Self {
        let mut memory_source = if memory_scheduler_enabled {
            MemorySource::ProcMeminfo
        } else {
            MemorySource::Disabled
        };
        let mut scheduler_active = memory_scheduler_enabled;
        let mut provider_failure_logged = false;

        if scheduler_active && let Err(error) = memory_provider() {
            tracing::warn!(
                path = MEMINFO_PATH,
                error = %error,
                "could not initialize RAM scheduler from /proc/meminfo; falling back to thread-cap-only scheduling"
            );
            scheduler_active = false;
            memory_source = MemorySource::ThreadCapOnly;
            provider_failure_logged = true;
        }

        Self {
            inner: Arc::new(AdmissionGateInner {
                state: Mutex::new(GateState {
                    active_tasks: 0,
                    throttled: false,
                    throttle_logged: false,
                    memory_scheduler_active: scheduler_active,
                    memory_source,
                    provider_failure_logged,
                }),
                condvar: Condvar::new(),
                memory_provider,
                max_ram_usage_fraction,
                resume_ram_usage_fraction: (max_ram_usage_fraction - RAM_USAGE_RESUME_HYSTERESIS)
                    .max(0.0),
                poll_interval: DEFAULT_MEMORY_POLL_INTERVAL,
            }),
        }
    }

    fn acquire(&self) -> AdmissionPermit {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("admission gate mutex poisoned");
        loop {
            if !state.memory_scheduler_active {
                state.active_tasks += 1;
                return AdmissionPermit { gate: self.clone() };
            }

            match (self.inner.memory_provider)() {
                Ok(usage) => {
                    if state.throttled {
                        if usage < self.inner.resume_ram_usage_fraction {
                            state.throttled = false;
                            state.throttle_logged = false;
                            tracing::info!(
                                current_ram_usage_fraction = usage,
                                resume_ram_usage_fraction = self.inner.resume_ram_usage_fraction,
                                active_tasks = state.active_tasks,
                                "RAM usage back below the resume threshold; resuming new admissions"
                            );
                            state.active_tasks += 1;
                            return AdmissionPermit { gate: self.clone() };
                        }

                        self.log_throttle(&mut state, usage);
                    } else if usage >= self.inner.max_ram_usage_fraction {
                        state.throttled = true;
                        state.throttle_logged = false;
                        self.log_throttle(&mut state, usage);
                    } else {
                        state.active_tasks += 1;
                        return AdmissionPermit { gate: self.clone() };
                    }
                }
                Err(error) => {
                    self.disable_memory_scheduler(&mut state, &error);
                    state.active_tasks += 1;
                    return AdmissionPermit { gate: self.clone() };
                }
            }

            let (next_state, _) = self
                .inner
                .condvar
                .wait_timeout(state, self.inner.poll_interval)
                .expect("admission gate mutex poisoned while waiting");
            state = next_state;
        }
    }

    fn release(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("admission gate mutex poisoned");
        if state.active_tasks > 0 {
            state.active_tasks -= 1;
        }
        self.inner.condvar.notify_all();
    }

    fn memory_scheduler_enabled(&self) -> bool {
        self.inner
            .state
            .lock()
            .expect("admission gate mutex poisoned")
            .memory_scheduler_active
    }

    fn memory_source_name(&self) -> &'static str {
        self.inner
            .state
            .lock()
            .expect("admission gate mutex poisoned")
            .memory_source
            .as_str()
    }

    fn log_throttle(&self, state: &mut GateState, usage: f64) {
        if state.throttle_logged {
            return;
        }

        if state.active_tasks > 0 {
            tracing::warn!(
                current_ram_usage_fraction = usage,
                max_ram_usage_fraction = self.inner.max_ram_usage_fraction,
                active_tasks = state.active_tasks,
                "RAM usage is above the configured threshold while experiments are in flight; throttling new admissions"
            );
        } else {
            tracing::info!(
                current_ram_usage_fraction = usage,
                max_ram_usage_fraction = self.inner.max_ram_usage_fraction,
                active_tasks = state.active_tasks,
                "RAM usage is above the configured threshold; waiting before admitting more work"
            );
        }

        state.throttle_logged = true;
    }

    fn disable_memory_scheduler(&self, state: &mut GateState, error: &str) {
        if !state.provider_failure_logged {
            tracing::warn!(
                path = MEMINFO_PATH,
                error = %error,
                "could not read RAM usage from /proc/meminfo; falling back to thread-cap-only scheduling"
            );
            state.provider_failure_logged = true;
        }
        state.memory_scheduler_active = false;
        state.throttled = false;
        state.throttle_logged = false;
        state.memory_source = MemorySource::ThreadCapOnly;
    }
}

struct AdmissionPermit {
    gate: AdmissionGate,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

/// Local experiment scheduler for `bekiper run`.
///
/// This owns a dedicated Rayon thread pool so `project.runtime.num_threads`
/// becomes an actual hard cap, and it wraps experiment dispatch in a
/// host-RAM-aware admission gate.
pub struct ParallelScheduler {
    pool: rayon::ThreadPool,
    gate: AdmissionGate,
    num_threads: usize,
    max_ram_usage_fraction: f64,
}

impl ParallelScheduler {
    pub fn new(settings: &RuntimeSettings) -> Result<Self> {
        Self::with_memory_provider(settings, Arc::new(read_system_ram_usage_from_proc_meminfo))
    }

    fn with_memory_provider(
        settings: &RuntimeSettings,
        memory_provider: MemoryUsageProvider,
    ) -> Result<Self> {
        let num_threads = resolve_thread_cap(settings.num_threads);
        let max_ram_usage_fraction =
            normalize_max_ram_usage_fraction(settings.max_ram_usage_fraction);
        let gate = AdmissionGate::new(
            settings.memory_scheduler_enabled,
            max_ram_usage_fraction,
            memory_provider,
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| BekiperError::Other(format!("build rayon thread pool: {e}")))?;

        Ok(Self {
            pool,
            gate,
            num_threads,
            max_ram_usage_fraction,
        })
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn memory_scheduler_enabled(&self) -> bool {
        self.gate.memory_scheduler_enabled()
    }

    pub fn max_ram_usage_fraction(&self) -> f64 {
        self.max_ram_usage_fraction
    }

    pub fn memory_source_name(&self) -> &'static str {
        self.gate.memory_source_name()
    }

    pub fn for_each<T, F>(&self, items: &[T], f: F)
    where
        T: Sync,
        F: Fn(&T) + Sync + Send,
    {
        self.dispatch(items.iter(), |item| {
            f(item);
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap_or_else(|never| match never {});
    }

    pub fn try_for_each<T, F, E>(&self, items: &[T], f: F) -> std::result::Result<(), E>
    where
        T: Sync,
        F: Fn(&T) -> std::result::Result<(), E> + Sync + Send,
        E: Send,
    {
        self.dispatch(items.iter(), f)
    }

    pub fn try_for_each_mut<T, F, E>(&self, items: &mut [T], f: F) -> std::result::Result<(), E>
    where
        T: Send,
        F: Fn(&mut T) -> std::result::Result<(), E> + Sync + Send,
        E: Send,
    {
        self.dispatch(items.iter_mut(), f)
    }

    fn dispatch<I, F, E>(&self, items: I, f: F) -> std::result::Result<(), E>
    where
        I: Iterator + Send,
        I::Item: Send,
        F: Fn(I::Item) -> std::result::Result<(), E> + Sync,
        E: Send,
    {
        let pending = Mutex::new(items);
        let failure = Mutex::new(None);
        let stopped = AtomicBool::new(false);
        // One pinned dispatcher per worker, without queued per-trial Rayon jobs.
        // Waiting on Polars' separate pool can run local Rayon work; a queued
        // outer trial would re-enter this pipeline on the same stack and defeat
        // the configured in-flight cap. Broadcast jobs cannot be stolen.
        self.pool.broadcast(|_| {
            while !stopped.load(Ordering::Acquire) {
                let item = pending.lock().expect("scheduler iterator poisoned").next();
                let Some(item) = item else { break };
                let _permit = self.gate.acquire();
                if stopped.load(Ordering::Acquire) {
                    break;
                }
                if let Err(error) = f(item) {
                    let mut first = failure.lock().expect("scheduler failure mutex poisoned");
                    if first.is_none() {
                        *first = Some(error);
                    }
                    stopped.store(true, Ordering::Release);
                    break;
                }
            }
        });
        match failure
            .into_inner()
            .expect("scheduler failure mutex poisoned")
        {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

fn resolve_thread_cap(configured: Option<usize>) -> usize {
    match configured {
        Some(0) => {
            let fallback = available_parallelism();
            tracing::warn!(
                fallback_num_threads = fallback,
                "project.runtime.num_threads must be > 0; falling back to available parallelism"
            );
            fallback
        }
        Some(v) => v,
        None => available_parallelism(),
    }
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
}

fn normalize_max_ram_usage_fraction(raw: f64) -> f64 {
    if raw > 0.0 && raw <= 1.0 {
        raw
    } else {
        tracing::warn!(
            configured_max_ram_usage_fraction = raw,
            default_max_ram_usage_fraction = DEFAULT_MAX_RAM_USAGE_FRACTION,
            "project.runtime.max_ram_usage_fraction must be in the interval (0.0, 1.0]; using the default"
        );
        DEFAULT_MAX_RAM_USAGE_FRACTION
    }
}

fn read_system_ram_usage_from_proc_meminfo() -> std::result::Result<f64, String> {
    let contents =
        std::fs::read_to_string(MEMINFO_PATH).map_err(|e| format!("read {MEMINFO_PATH}: {e}"))?;
    parse_proc_meminfo_ram_usage(&contents)
}

/// Serial staged rendering adds the estimated child/raster workspace to current
/// host usage before admission. Native stages have already drained at this point.
pub fn wait_for_renderer_workspace(bytes: u64, settings: &RuntimeSettings) -> Result<()> {
    if bytes == 0 || !settings.memory_scheduler_enabled {
        return Ok(());
    }
    let fraction = normalize_max_ram_usage_fraction(settings.max_ram_usage_fraction);
    let mut logged = false;
    loop {
        let contents = std::fs::read_to_string(MEMINFO_PATH)?;
        let value = |key| {
            contents
                .lines()
                .find_map(|line| parse_meminfo_line_kib(line, key))
                .and_then(|v| v.checked_mul(1024))
                .ok_or_else(|| BekiperError::Data(format!("missing renderer admission {key}")))
        };
        let total = value("MemTotal")?;
        let available = value("MemAvailable")?;
        let limit = (total as f64 * fraction) as u64;
        if bytes > limit {
            return Err(BekiperError::Config(
                "renderer workspace exceeds RAM admission limit".into(),
            ));
        }
        if total.saturating_sub(available).saturating_add(bytes) <= limit {
            return Ok(());
        }
        if !logged {
            tracing::info!(bytes, available, limit, "waiting for renderer workspace");
            logged = true;
        }
        std::thread::sleep(DEFAULT_MEMORY_POLL_INTERVAL);
    }
}

fn parse_proc_meminfo_ram_usage(contents: &str) -> std::result::Result<f64, String> {
    let mut mem_total_kib = None;
    let mut mem_available_kib = None;

    for line in contents.lines() {
        if let Some(value) = parse_meminfo_line_kib(line, "MemTotal") {
            mem_total_kib = Some(value);
        } else if let Some(value) = parse_meminfo_line_kib(line, "MemAvailable") {
            mem_available_kib = Some(value);
        }
    }

    let mem_total_kib = mem_total_kib.ok_or_else(|| "MemTotal not found".to_string())?;
    let mem_available_kib =
        mem_available_kib.ok_or_else(|| "MemAvailable not found".to_string())?;

    if mem_total_kib == 0 {
        return Err("MemTotal must be > 0".to_string());
    }
    if mem_available_kib > mem_total_kib {
        return Err("MemAvailable cannot exceed MemTotal".to_string());
    }

    Ok(1.0 - (mem_available_kib as f64 / mem_total_kib as f64))
}

fn parse_meminfo_line_kib(line: &str, key: &str) -> Option<u64> {
    let (name, rest) = line.split_once(':')?;
    if name.trim() != key {
        return None;
    }

    let value = rest.split_whitespace().next()?;
    value.parse::<u64>().ok()
}
