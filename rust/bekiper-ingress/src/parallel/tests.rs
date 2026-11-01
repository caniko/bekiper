#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    use bekiper_core::config::RuntimeSettings;

    use super::super::{MemoryUsageProvider, ParallelScheduler, parse_proc_meminfo_ram_usage};

    fn settings() -> RuntimeSettings {
        RuntimeSettings {
            num_threads: Some(2),
            memory_scheduler_enabled: true,
            max_ram_usage_fraction: 0.80,
            ..RuntimeSettings::default()
        }
    }

    fn provider_from_atomic(usage_percent: Arc<AtomicUsize>) -> MemoryUsageProvider {
        Arc::new(move || Ok(usage_percent.load(Ordering::Relaxed) as f64 / 100.0))
    }

    #[test]
    fn parse_proc_meminfo_usage_reads_total_and_available() {
        let usage = parse_proc_meminfo_ram_usage(
            "MemTotal:       100000 kB\nMemAvailable:    25000 kB\nSwapTotal:           0 kB\n",
        )
        .unwrap();

        assert!((usage - 0.75).abs() < 1e-12);
    }

    #[test]
    fn scheduler_blocks_admission_until_memory_drops() {
        let usage_percent = Arc::new(AtomicUsize::new(90));
        let scheduler = ParallelScheduler::with_memory_provider(
            &settings(),
            provider_from_atomic(usage_percent.clone()),
        )
        .unwrap();
        let started = Arc::new(AtomicUsize::new(0));
        let started_for_worker = started.clone();

        let handle = thread::spawn(move || {
            scheduler.for_each(&[1usize], |_| {
                started_for_worker.fetch_add(1, Ordering::SeqCst);
            });
        });

        thread::sleep(Duration::from_millis(150));
        assert_eq!(started.load(Ordering::SeqCst), 0);

        usage_percent.store(30, Ordering::SeqCst);
        handle.join().unwrap();
        assert_eq!(started.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scheduler_respects_hysteresis_before_resuming() {
        let usage_percent = Arc::new(AtomicUsize::new(79));
        let scheduler = ParallelScheduler::with_memory_provider(
            &RuntimeSettings {
                num_threads: Some(1),
                ..settings()
            },
            provider_from_atomic(usage_percent.clone()),
        )
        .unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let order_for_worker = order.clone();
        let usage_for_worker = usage_percent.clone();
        let releaser = Arc::new(Mutex::new(None));
        let releaser_for_worker = releaser.clone();

        scheduler.for_each(&[0usize, 1usize], move |item| {
            order_for_worker.lock().unwrap().push(*item);
            if *item == 0 {
                usage_for_worker.store(82, Ordering::SeqCst);
                let usage_for_thread = usage_for_worker.clone();
                let handle = thread::spawn(move || {
                    thread::sleep(Duration::from_millis(150));
                    usage_for_thread.store(78, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(150));
                    usage_for_thread.store(74, Ordering::SeqCst);
                });
                *releaser_for_worker.lock().unwrap() = Some(handle);
            }
        });

        if let Some(handle) = releaser.lock().unwrap().take() {
            handle.join().unwrap();
        }

        let observed = order.lock().unwrap().clone();
        assert_eq!(observed, vec![0, 1]);
    }

    #[test]
    fn scheduler_honors_thread_cap_even_when_memory_scheduler_is_disabled() {
        let scheduler = ParallelScheduler::with_memory_provider(
            &RuntimeSettings {
                num_threads: Some(2),
                memory_scheduler_enabled: false,
                max_ram_usage_fraction: 0.80,
                ..RuntimeSettings::default()
            },
            Arc::new(|| Ok(0.1)),
        )
        .unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        scheduler.for_each(&[0usize, 1, 2, 3], {
            let active = active.clone();
            let peak = peak.clone();
            move |_| {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(50));
                active.fetch_sub(1, Ordering::SeqCst);
            }
        });

        assert!(peak.load(Ordering::SeqCst) <= 2);
    }

    #[test]
    fn nested_parallel_work_cannot_reenter_other_experiments() {
        for threads in [1, 2] {
            for mode in ["for_each", "try_for_each", "try_for_each_mut"] {
                let scheduler = ParallelScheduler::with_memory_provider(
                    &RuntimeSettings {
                        num_threads: Some(threads),
                        memory_scheduler_enabled: false,
                        ..settings()
                    },
                    Arc::new(|| Ok(0.1)),
                )
                .unwrap();
                let nested_pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap();
                let active = AtomicUsize::new(0);
                let peak = AtomicUsize::new(0);
                let completed = AtomicUsize::new(0);
                let work = || {
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(current, Ordering::SeqCst);
                    // Polars also waits on a separate Rayon pool. Rayon can
                    // execute queued outer jobs while waiting across pools.
                    nested_pool.install(|| thread::sleep(Duration::from_millis(2)));
                    active.fetch_sub(1, Ordering::SeqCst);
                    completed.fetch_add(1, Ordering::SeqCst);
                };
                let mut values = [0usize; 12];
                match mode {
                    "for_each" => scheduler.for_each(&values, |_| work()),
                    "try_for_each" => scheduler
                        .try_for_each(&values, |_| {
                            work();
                            Ok::<_, ()>(())
                        })
                        .unwrap(),
                    _ => scheduler
                        .try_for_each_mut(&mut values, |value| {
                            work();
                            *value += 1;
                            Ok::<_, ()>(())
                        })
                        .unwrap(),
                }
                assert_eq!(completed.load(Ordering::SeqCst), values.len());
                assert_eq!(active.load(Ordering::SeqCst), 0);
                assert!(
                    peak.load(Ordering::SeqCst) <= threads,
                    "{mode}: active experiments exceeded {threads}"
                );
                if mode == "try_for_each_mut" {
                    assert!(values.iter().all(|value| *value == 1));
                }
            }
        }
    }

    #[test]
    fn fallible_dispatch_drains_in_flight_work_before_returning() {
        let scheduler = ParallelScheduler::with_memory_provider(
            &RuntimeSettings {
                num_threads: Some(2),
                memory_scheduler_enabled: false,
                ..settings()
            },
            Arc::new(|| Ok(0.1)),
        )
        .unwrap();
        let started = std::sync::Barrier::new(2);
        let completed = AtomicUsize::new(0);
        let result = scheduler.try_for_each(&[0, 1], |item| {
            started.wait();
            if *item == 0 {
                return Err("trial failed");
            }
            thread::sleep(Duration::from_millis(20));
            completed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        assert_eq!(result, Err("trial failed"));
        assert_eq!(completed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scheduler_warn_path_is_non_fatal_when_inflight_work_overshoots_threshold() {
        let usage_percent = Arc::new(AtomicUsize::new(20));
        let scheduler = ParallelScheduler::with_memory_provider(
            &settings(),
            provider_from_atomic(usage_percent.clone()),
        )
        .unwrap();
        let completed = Arc::new(AtomicUsize::new(0));
        let usage_for_worker = usage_percent.clone();

        scheduler.for_each(&[0usize, 1usize], {
            let completed = completed.clone();
            move |item| {
                if *item == 0 {
                    usage_for_worker.store(95, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(150));
                    usage_for_worker.store(30, Ordering::SeqCst);
                }
                completed.fetch_add(1, Ordering::SeqCst);
            }
        });

        assert_eq!(completed.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn scheduler_falls_back_cleanly_when_memory_source_is_unavailable() {
        let scheduler = ParallelScheduler::with_memory_provider(
            &settings(),
            Arc::new(|| Err("boom".to_string())),
        )
        .unwrap();
        let started = Arc::new(AtomicUsize::new(0));

        scheduler.for_each(&[0usize, 1usize], {
            let started = started.clone();
            move |_| {
                started.fetch_add(1, Ordering::SeqCst);
            }
        });

        assert_eq!(scheduler.memory_source_name(), "thread_cap_only");
        assert!(!scheduler.memory_scheduler_enabled());
        assert_eq!(started.load(Ordering::SeqCst), 2);
    }
}
