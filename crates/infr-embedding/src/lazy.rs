//! Standalone embedding lifecycle. Shared-LLM Vulkan clients retain their unified-arena path.
use crate::{EmbeddingBatch, EmbeddingConfig, EmbeddingEngine};
use anyhow::{anyhow, ensure, Context, Result};
use infr_core::{MemoryTier, ResourceKind, ResourceSnapshot, ResourceTracker};
use std::{
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    path::Path,
    sync::{Arc, Condvar, Mutex, Weak},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

type Factory = dyn Fn() -> Result<Box<dyn EmbeddingEngine>> + Send + Sync;

struct Residency {
    model: Option<Box<dyn EmbeddingEngine>>,
    last_use: Instant,
}

struct Inner {
    factory: Box<Factory>,
    state: Mutex<Residency>,
    resource: ResourceTracker,
}

/// A metadata-only service until the first request. Idle eviction drops the entire standalone
/// backend, including weight banks, repack caches and runtime scratch, so reloads cannot append
/// allocations to an old bump arena. A zero timeout keeps the first loaded model resident.
pub struct LazyNativeEmbeddingEngine {
    cfg: EmbeddingConfig,
    inner: Arc<Inner>,
    _reaper: Option<IdleReaper>,
}

impl LazyNativeEmbeddingEngine {
    pub(crate) fn new(
        path: &Path,
        idle_timeout: Duration,
        factory: impl Fn() -> Result<Box<dyn EmbeddingEngine>> + Send + Sync + 'static,
    ) -> Result<Self> {
        let (cfg, bytes) = crate::native::standalone_metadata(path)?;
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&cfg.name);
        Self::with_factory(
            cfg.clone(),
            format!("embedding:{id}"),
            bytes,
            idle_timeout,
            factory,
        )
    }

    fn with_factory(
        cfg: EmbeddingConfig,
        id: String,
        bytes: u64,
        idle_timeout: Duration,
        factory: impl Fn() -> Result<Box<dyn EmbeddingEngine>> + Send + Sync + 'static,
    ) -> Result<Self> {
        let inner = Arc::new(Inner {
            factory: Box::new(factory),
            state: Mutex::new(Residency {
                model: None,
                last_use: Instant::now(),
            }),
            resource: ResourceTracker::new(
                id,
                ResourceKind::EmbeddingWeights,
                bytes,
                0,
                MemoryTier::Ssd,
                bytes,
            ),
        });
        let reaper = if idle_timeout.is_zero() {
            None
        } else {
            Some(IdleReaper::spawn(Arc::downgrade(&inner), idle_timeout)?)
        };
        tracing::info!(model = %cfg.name, weights_mib = bytes as f64 / 1048576.0,
            idle_timeout_secs = idle_timeout.as_secs(), "standalone embedding ready; backend loads on first request");
        Ok(Self {
            cfg,
            inner,
            _reaper: reaper,
        })
    }
}

impl EmbeddingEngine for LazyNativeEmbeddingEngine {
    fn config(&self) -> &EmbeddingConfig {
        &self.cfg
    }

    fn resource_snapshot(&self) -> ResourceSnapshot {
        self.inner.resource.snapshot()
    }

    fn embed(&self, inputs: &[String]) -> Result<EmbeddingBatch> {
        ensure!(
            !inputs.is_empty(),
            "input array must contain at least one string"
        );
        let _lease = self.inner.resource.acquire();
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow!("embedding service lock poisoned"))?;
        // Catch while the guard remains outside the unwind boundary. A failed backend is never
        // reused, and one panicking request cannot poison every later request.
        let result = catch_unwind(AssertUnwindSafe(|| {
            if state.model.is_none() {
                let model = (self.inner.factory)()?;
                ensure!(
                    model.config() == &self.cfg,
                    "embedding metadata changed since service startup"
                );
                let snapshot = model.resource_snapshot();
                self.inner
                    .resource
                    .set_residency(snapshot.tier, snapshot.resident_bytes);
                state.model = Some(model);
            }
            state
                .model
                .as_ref()
                .expect("model loaded above")
                .embed(inputs)
        }));
        state.last_use = Instant::now();
        match result {
            Ok(result) => result,
            Err(panic) => {
                state.model = None;
                self.inner.resource.set_residency(MemoryTier::Ssd, 0);
                drop(state);
                resume_unwind(panic)
            }
        }
    }
}

impl Inner {
    fn evict_if_idle(&self, idle_timeout: Duration) {
        let Ok(mut state) = self.state.try_lock() else {
            return;
        };
        if self.resource.snapshot().active_requests != 0 || state.last_use.elapsed() < idle_timeout
        {
            return;
        }
        if state.model.take().is_some() {
            self.resource.set_residency(MemoryTier::Ssd, 0);
            tracing::info!("idle standalone embedding backend and weights released");
        }
    }
}

struct IdleReaper {
    stop: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

impl IdleReaper {
    fn spawn(inner: Weak<Inner>, timeout: Duration) -> Result<Self> {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = Arc::clone(&stop);
        let interval = timeout
            .min(Duration::from_secs(1))
            .max(Duration::from_millis(10));
        let thread = thread::Builder::new()
            .name("embedding-idle".into())
            .spawn(move || loop {
                let (lock, changed) = &*signal;
                let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
                let (guard, _) = changed
                    .wait_timeout_while(guard, interval, |stop| !*stop)
                    .unwrap_or_else(|e| e.into_inner());
                if *guard {
                    break;
                }
                drop(guard);
                let Some(inner) = inner.upgrade() else { break };
                inner.evict_if_idle(timeout);
            })
            .context("spawn standalone embedding idle reaper")?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for IdleReaper {
    fn drop(&mut self) {
        let (lock, changed) = &*self.stop;
        *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
        changed.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn cfg() -> EmbeddingConfig {
        EmbeddingConfig {
            name: "test".into(),
            architecture: "bert".into(),
            max_context: 512,
            dimensions: 2,
        }
    }

    struct Fake {
        cfg: EmbeddingConfig,
        dropped: Arc<AtomicUsize>,
        panic: bool,
    }
    impl Drop for Fake {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl EmbeddingEngine for Fake {
        fn config(&self) -> &EmbeddingConfig {
            &self.cfg
        }
        fn resource_snapshot(&self) -> ResourceSnapshot {
            ResourceTracker::new(
                "fake",
                ResourceKind::EmbeddingWeights,
                100,
                100,
                MemoryTier::Ram,
                100,
            )
            .snapshot()
        }
        fn embed(&self, _: &[String]) -> Result<EmbeddingBatch> {
            assert!(!self.panic, "fake backend panic");
            Ok(EmbeddingBatch {
                embeddings: vec![vec![1.0, 0.0]],
                prompt_tokens: 1,
            })
        }
    }

    fn service(
        timeout: Duration,
    ) -> (
        LazyNativeEmbeddingEngine,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let loaded = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&loaded);
        let deaths = Arc::clone(&dropped);
        let engine = LazyNativeEmbeddingEngine::with_factory(
            cfg(),
            "embedding:test".into(),
            100,
            timeout,
            move || {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(Box::new(Fake {
                    cfg: cfg(),
                    dropped: Arc::clone(&deaths),
                    panic: false,
                }))
            },
        )
        .unwrap();
        (engine, loaded, dropped)
    }

    #[test]
    fn lazy_reuse_and_full_backend_release_then_reload() {
        let (engine, loaded, dropped) = service(Duration::ZERO);
        assert_eq!(loaded.load(Ordering::SeqCst), 0);
        assert_eq!(engine.resource_snapshot().resident_bytes, 0);
        for _ in 0..2 {
            engine.embed(&["hello".into()]).unwrap();
        }
        assert_eq!(loaded.load(Ordering::SeqCst), 1);
        assert_eq!(engine.resource_snapshot().tier, MemoryTier::Ram);
        engine.inner.evict_if_idle(Duration::ZERO);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(engine.resource_snapshot().resident_bytes, 0);
        assert_eq!(engine.resource_snapshot().tier, MemoryTier::Ssd);
        engine.embed(&["hello".into()]).unwrap();
        assert_eq!(loaded.load(Ordering::SeqCst), 2);
        drop(engine);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn active_lease_and_idle_deadline_protect_model() {
        let (engine, _, dropped) = service(Duration::ZERO);
        engine.embed(&["hello".into()]).unwrap();
        engine.inner.evict_if_idle(Duration::from_secs(3600));
        let lease = engine.inner.resource.acquire();
        engine.inner.evict_if_idle(Duration::ZERO);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(lease);
        engine.inner.evict_if_idle(Duration::ZERO);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn empty_input_does_not_load_and_zero_timeout_has_no_reaper() {
        let (engine, loaded, _) = service(Duration::ZERO);
        assert!(engine.embed(&[]).is_err());
        assert_eq!(loaded.load(Ordering::SeqCst), 0);
        assert!(engine._reaper.is_none());
    }

    #[test]
    fn load_failure_and_panic_do_not_poison_later_requests() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&attempts);
        let dropped = Arc::new(AtomicUsize::new(0));
        let deaths = Arc::clone(&dropped);
        let engine = LazyNativeEmbeddingEngine::with_factory(
            cfg(),
            "test".into(),
            100,
            Duration::ZERO,
            move || {
                let attempt = count.fetch_add(1, Ordering::SeqCst);
                ensure!(attempt > 0, "first load fails");
                Ok(Box::new(Fake {
                    cfg: cfg(),
                    dropped: Arc::clone(&deaths),
                    panic: attempt == 1,
                }))
            },
        )
        .unwrap();
        assert!(engine.embed(&["x".into()]).is_err());
        assert_eq!(engine.resource_snapshot().active_requests, 0);
        assert!(catch_unwind(AssertUnwindSafe(|| engine.embed(&["x".into()]))).is_err());
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(engine.resource_snapshot().resident_bytes, 0);
        engine.embed(&["x".into()]).unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn reaper_evicts_and_drop_wakes_long_idle_wait() {
        let (engine, _, dropped) = service(Duration::from_millis(10));
        engine.embed(&["x".into()]).unwrap();
        let limit = Instant::now() + Duration::from_secs(2);
        while dropped.load(Ordering::SeqCst) == 0 && Instant::now() < limit {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        let (engine, _, _) = service(Duration::from_secs(3600));
        let start = Instant::now();
        drop(engine);
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}
