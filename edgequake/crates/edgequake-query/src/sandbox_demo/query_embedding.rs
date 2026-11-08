//! Question embeddings for the long-lived demo engine.
//!
//! The only network call on the demo hot path is embedding the question,
//! and presenters repeat questions. This memoizes the *vector* — never a
//! result — keyed on everything that determines it: provider, model,
//! dimension, workspace, and the exact text sent to the provider.
//!
//! - **Bounded.** LRU by entry count and by approximate bytes.
//! - **Coalesced.** Identical cold questions arriving together share one
//!   provider request. No lock is held across that await; waiters share a
//!   future, and the future removes itself from the in-flight map when it
//!   settles. Errors reach every waiter and are never cached.
//! - **Batched, optionally.** Distinct cold questions arriving within
//!   `EDGEQUAKE_SANDBOX_EMBED_BATCH_WINDOW_US` share one provider call of
//!   up to `EDGEQUAKE_SANDBOX_EMBED_BATCH_MAX` texts. Off by default:
//!   against local Ollama the batched vectors are bit-identical but one
//!   call of N texts was no faster than N concurrent single calls.
//! - **Rollback.** `EDGEQUAKE_SANDBOX_QUERY_EMBED_CACHE=0` restores the
//!   original one-provider-call-per-query path exactly.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use edgequake_llm::traits::EmbeddingProvider;
use futures::future::{BoxFuture, FutureExt, Shared};
use tokio::sync::oneshot;

use super::error::{Result, SandboxDemoError};
use super::types::QueryEmbeddingSource;

/// Entry cap of the query-embedding cache; `0` disables cache and
/// coalescing.
pub const QUERY_EMBED_CACHE_ENV: &str = "EDGEQUAKE_SANDBOX_QUERY_EMBED_CACHE";
/// Approximate byte cap of the query-embedding cache.
pub const QUERY_EMBED_CACHE_BYTES_ENV: &str = "EDGEQUAKE_SANDBOX_QUERY_EMBED_CACHE_BYTES";
/// Default entry cap.
pub const DEFAULT_QUERY_EMBED_CACHE_ENTRIES: usize = 4096;
/// Default byte cap: ~20k 768-d vectors, far above the entry cap.
pub const DEFAULT_QUERY_EMBED_CACHE_BYTES: usize = 64 << 20;
/// How long a cold question waits for others to share its provider call,
/// in microseconds; `0` (the default) disables batching.
pub const EMBED_BATCH_WINDOW_ENV: &str = "EDGEQUAKE_SANDBOX_EMBED_BATCH_WINDOW_US";
/// Most questions one batched provider call carries.
pub const EMBED_BATCH_MAX_ENV: &str = "EDGEQUAKE_SANDBOX_EMBED_BATCH_MAX";
/// Default batch size cap.
pub const DEFAULT_EMBED_BATCH_MAX: usize = 32;
/// Bookkeeping per entry beyond the text and the vector.
const ENTRY_OVERHEAD_BYTES: usize = 128;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// Query-embedding tuning, read once at engine load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryEmbedConfig {
    /// Entry cap; `0` disables the cache and coalescing.
    pub cache_entries: usize,
    /// Approximate byte cap.
    pub cache_bytes: usize,
    /// Gathering window for distinct cold questions; zero disables.
    pub batch_window: Duration,
    /// Most questions per batched provider call; `<= 1` disables.
    pub batch_max: usize,
}

impl Default for QueryEmbedConfig {
    fn default() -> Self {
        Self {
            cache_entries: DEFAULT_QUERY_EMBED_CACHE_ENTRIES,
            cache_bytes: DEFAULT_QUERY_EMBED_CACHE_BYTES,
            batch_window: Duration::ZERO,
            batch_max: DEFAULT_EMBED_BATCH_MAX,
        }
    }
}

impl QueryEmbedConfig {
    /// Defaults overridden by the environment.
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            cache_entries: env_usize(QUERY_EMBED_CACHE_ENV, defaults.cache_entries),
            cache_bytes: env_usize(QUERY_EMBED_CACHE_BYTES_ENV, defaults.cache_bytes),
            batch_window: Duration::from_micros(env_usize(EMBED_BATCH_WINDOW_ENV, 0) as u64),
            batch_max: env_usize(EMBED_BATCH_MAX_ENV, defaults.batch_max),
        }
    }

    fn batching(&self) -> bool {
        self.cache_entries > 0 && !self.batch_window.is_zero() && self.batch_max > 1
    }

    /// The cache off: one provider call per query, as before it existed.
    pub fn disabled() -> Self {
        Self {
            cache_entries: 0,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EmbedKey {
    provider: Arc<str>,
    model: Arc<str>,
    dimension: usize,
    workspace: Arc<str>,
    text: Arc<str>,
}

type Outcome = std::result::Result<Arc<[f32]>, String>;
/// A vector and whether its provider call carried other questions.
type Fetched = std::result::Result<(Arc<[f32]>, bool), String>;
type InFlight = Shared<BoxFuture<'static, Fetched>>;

struct Pending {
    id: u64,
    texts: Vec<String>,
    waiters: Vec<oneshot::Sender<Fetched>>,
}

struct Entry {
    vector: Arc<[f32]>,
    tick: u64,
    bytes: usize,
}

#[derive(Default)]
struct State {
    entries: HashMap<EmbedKey, Entry>,
    recency: BTreeMap<u64, EmbedKey>,
    tick: u64,
    bytes: usize,
    in_flight: HashMap<EmbedKey, InFlight>,
}

impl State {
    fn hit(&mut self, key: &EmbedKey) -> Option<Arc<[f32]>> {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.entries.get_mut(key)?;
        self.recency.remove(&entry.tick);
        entry.tick = tick;
        self.recency.insert(tick, key.clone());
        Some(entry.vector.clone())
    }

    fn insert(&mut self, key: EmbedKey, vector: Arc<[f32]>, config: &QueryEmbedConfig) {
        let bytes =
            key.text.len() + vector.len() * std::mem::size_of::<f32>() + ENTRY_OVERHEAD_BYTES;
        if bytes > config.cache_bytes || self.entries.contains_key(&key) {
            return;
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= config.cache_entries
                || self.bytes + bytes > config.cache_bytes)
        {
            let Some((_, oldest)) = self.recency.pop_first() else {
                break;
            };
            if let Some(evicted) = self.entries.remove(&oldest) {
                self.bytes -= evicted.bytes;
            }
        }
        self.tick += 1;
        self.recency.insert(self.tick, key.clone());
        self.bytes += bytes;
        self.entries.insert(
            key,
            Entry {
                vector,
                tick: self.tick,
                bytes,
            },
        );
    }
}

struct Inner {
    provider: Arc<dyn EmbeddingProvider>,
    dim: usize,
    config: QueryEmbedConfig,
    scope: (Arc<str>, Arc<str>, usize, Arc<str>),
    state: Mutex<State>,
    pending: Mutex<Option<Pending>>,
    next_batch: AtomicU64,
}

impl Inner {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn key(&self, text: &str) -> EmbedKey {
        let (provider, model, dimension, workspace) = &self.scope;
        EmbedKey {
            provider: provider.clone(),
            model: model.clone(),
            dimension: *dimension,
            workspace: workspace.clone(),
            text: text.into(),
        }
    }

    async fn fetch(&self, text: &str) -> Outcome {
        let vector = self
            .provider
            .embed_batched(&[text.to_string()])
            .await
            .map_err(|e| e.to_string())?
            .into_iter()
            .next()
            .ok_or_else(|| "provider returned no vector".to_string())?;
        finish(vector, self.dim)
    }

    /// Join (or open) the gathering batch and wait for its vector.
    async fn fetch_batched(self: &Arc<Self>, text: &str) -> Fetched {
        let (sender, receiver) = oneshot::channel();
        let full = {
            let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            let batch = match pending.as_mut() {
                Some(batch) => batch,
                None => {
                    let id = self.next_batch.fetch_add(1, Ordering::Relaxed);
                    tokio::spawn(self.clone().flush_after_window(id));
                    pending.insert(Pending {
                        id,
                        texts: Vec::new(),
                        waiters: Vec::new(),
                    })
                }
            };
            batch.texts.push(text.to_string());
            batch.waiters.push(sender);
            if batch.texts.len() >= self.config.batch_max {
                pending.take()
            } else {
                None
            }
        };
        if let Some(batch) = full {
            tokio::spawn(self.clone().run_batch(batch));
        }
        receiver
            .await
            .unwrap_or_else(|_| Err("embedding batch was abandoned".to_string()))
    }

    async fn flush_after_window(self: Arc<Self>, id: u64) {
        tokio::time::sleep(self.config.batch_window).await;
        let batch = {
            let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            match pending.as_ref() {
                Some(batch) if batch.id == id => pending.take(),
                _ => None,
            }
        };
        if let Some(batch) = batch {
            self.run_batch(batch).await;
        }
    }

    /// One provider call for the whole batch, scattered back in order.
    async fn run_batch(self: Arc<Self>, batch: Pending) {
        let Pending { texts, waiters, .. } = batch;
        let batched = texts.len() > 1;
        let vectors = match self.provider.embed_batched(&texts).await {
            Ok(vectors) if vectors.len() == texts.len() => vectors,
            Ok(vectors) => {
                let message = if vectors.is_empty() && texts.len() == 1 {
                    "provider returned no vector".to_string()
                } else {
                    format!(
                        "provider returned {} vectors for {} questions",
                        vectors.len(),
                        texts.len()
                    )
                };
                for waiter in waiters {
                    let _ = waiter.send(Err(message.clone()));
                }
                return;
            }
            Err(e) => {
                let message = e.to_string();
                for waiter in waiters {
                    let _ = waiter.send(Err(message.clone()));
                }
                return;
            }
        };
        for (vector, waiter) in vectors.into_iter().zip(waiters) {
            let _ = waiter.send(finish(vector, self.dim).map(|v| (v, batched)));
        }
    }
}

/// Check the dimension and L2-normalize, exactly as the uncached path did.
fn finish(mut vector: Vec<f32>, dim: usize) -> Outcome {
    if vector.len() != dim {
        return Err(format!(
            "question embedded to {} dimensions, the index is {dim}",
            vector.len()
        ));
    }
    super::engine::normalize(&mut vector);
    Ok(vector.into())
}

/// The engine's question embedder.
pub(crate) struct QueryEmbedder {
    inner: Arc<Inner>,
}

impl QueryEmbedder {
    pub(crate) fn new(
        provider: Arc<dyn EmbeddingProvider>,
        dim: usize,
        workspace: &str,
        config: QueryEmbedConfig,
    ) -> Self {
        let scope = (
            Arc::from(provider.name()),
            Arc::from(provider.model()),
            provider.dimension(),
            Arc::from(workspace),
        );
        Self {
            inner: Arc::new(Inner {
                provider,
                dim,
                config,
                scope,
                state: Mutex::new(State::default()),
                pending: Mutex::new(None),
                next_batch: AtomicU64::new(0),
            }),
        }
    }

    /// The normalized question vector and where it came from.
    pub(crate) async fn embed(&self, text: &str) -> Result<(Arc<[f32]>, QueryEmbeddingSource)> {
        if self.inner.config.cache_entries == 0 {
            let vector = self
                .inner
                .fetch(text)
                .await
                .map_err(SandboxDemoError::Embedding)?;
            return Ok((vector, QueryEmbeddingSource::Miss));
        }

        let key = self.inner.key(text);
        let (shared, source) = {
            let mut state = self.inner.state();
            if let Some(vector) = state.hit(&key) {
                return Ok((vector, QueryEmbeddingSource::Hit));
            }
            match state.in_flight.get(&key) {
                Some(shared) => (shared.clone(), QueryEmbeddingSource::Coalesced),
                None => {
                    let shared = self.call(key.clone());
                    state.in_flight.insert(key, shared.clone());
                    (shared, QueryEmbeddingSource::Miss)
                }
            }
        };
        let (vector, batched) = shared.await.map_err(SandboxDemoError::Embedding)?;
        let source = match source {
            QueryEmbeddingSource::Miss if batched => QueryEmbeddingSource::Batched,
            other => other,
        };
        Ok((vector, source))
    }

    /// One provider request for `key`, settling its own in-flight slot.
    fn call(&self, key: EmbedKey) -> InFlight {
        let inner = self.inner.clone();
        async move {
            let outcome = if inner.config.batching() {
                inner.fetch_batched(&key.text).await
            } else {
                inner.fetch(&key.text).await.map(|v| (v, false))
            };
            let mut state = inner.state();
            state.in_flight.remove(&key);
            if let Ok((vector, _)) = &outcome {
                state.insert(key, vector.clone(), &inner.config);
            }
            outcome
        }
        .boxed()
        .shared()
    }

    #[cfg(test)]
    pub(crate) fn cached_entries(&self) -> (usize, usize) {
        let state = self.inner.state();
        (state.entries.len(), state.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::sandbox_demo::engine::tests::CountingEmbedder;

    fn embedder(provider: &Arc<CountingEmbedder>, config: QueryEmbedConfig) -> QueryEmbedder {
        let provider: Arc<dyn EmbeddingProvider> = provider.clone();
        let dim = provider.dimension();
        QueryEmbedder::new(provider, dim, "demo-workspace", config)
    }

    async fn source(embedder: &QueryEmbedder, text: &str) -> QueryEmbeddingSource {
        embedder.embed(text).await.unwrap().1
    }

    #[tokio::test]
    async fn the_entry_cap_evicts_the_least_recently_used() {
        let provider = CountingEmbedder::stalling(0);
        let cache = embedder(
            &provider,
            QueryEmbedConfig {
                cache_entries: 2,
                ..QueryEmbedConfig::default()
            },
        );
        for text in ["risk a", "risk b"] {
            assert_eq!(source(&cache, text).await, QueryEmbeddingSource::Miss);
        }
        // Touch "a" so "b" is the oldest, then overflow.
        assert_eq!(source(&cache, "risk a").await, QueryEmbeddingSource::Hit);
        assert_eq!(source(&cache, "risk c").await, QueryEmbeddingSource::Miss);
        assert_eq!(cache.cached_entries().0, 2);
        assert_eq!(source(&cache, "risk a").await, QueryEmbeddingSource::Hit);
        assert_eq!(source(&cache, "risk b").await, QueryEmbeddingSource::Miss);
        assert_eq!(provider.calls(), 4);
    }

    #[tokio::test]
    async fn the_byte_cap_bounds_the_cache_independently_of_the_entry_cap() {
        let provider = CountingEmbedder::stalling(0);
        let one_entry = "risk a".len() + provider.dimension() * 4 + ENTRY_OVERHEAD_BYTES;
        let cache = embedder(
            &provider,
            QueryEmbedConfig {
                cache_entries: 100,
                cache_bytes: one_entry,
                ..QueryEmbedConfig::default()
            },
        );
        for text in ["risk a", "risk b", "risk c"] {
            assert_eq!(source(&cache, text).await, QueryEmbeddingSource::Miss);
            let (entries, bytes) = cache.cached_entries();
            assert_eq!(entries, 1);
            assert!(bytes <= one_entry);
        }
        assert_eq!(source(&cache, "risk c").await, QueryEmbeddingSource::Hit);

        // An entry larger than the whole budget is served but not kept.
        let long = "risk ".repeat(64);
        assert_eq!(source(&cache, &long).await, QueryEmbeddingSource::Miss);
        assert_eq!(source(&cache, &long).await, QueryEmbeddingSource::Miss);
        assert_eq!(source(&cache, "risk c").await, QueryEmbeddingSource::Hit);
    }

    fn batching(window_ms: u64, batch_max: usize) -> QueryEmbedConfig {
        QueryEmbedConfig {
            batch_window: Duration::from_millis(window_ms),
            batch_max,
            ..QueryEmbedConfig::default()
        }
    }

    #[tokio::test]
    async fn distinct_cold_questions_share_one_batched_call() {
        let provider = CountingEmbedder::stalling(0);
        let batched = embedder(&provider, batching(30, 32));
        let texts = ["refused", "risk risk", "permitted routine", "process_exec"];
        let got = futures::future::join_all(texts.iter().map(|t| batched.embed(t))).await;
        assert_eq!(provider.calls(), 1);
        assert_eq!(*provider.batch_sizes.lock().unwrap(), vec![4]);

        let reference = CountingEmbedder::stalling(0);
        let single = embedder(&reference, QueryEmbedConfig::disabled());
        for (text, got) in texts.iter().zip(got) {
            let (vector, source) = got.unwrap();
            assert_eq!(source, QueryEmbeddingSource::Batched);
            assert_eq!(&*vector, &*single.embed(text).await.unwrap().0);
        }
        for text in texts {
            assert_eq!(source(&batched, text).await, QueryEmbeddingSource::Hit);
        }
    }

    #[tokio::test]
    async fn a_full_batch_is_sent_without_waiting_out_the_window() {
        let provider = CountingEmbedder::stalling(0);
        let batched = embedder(&provider, batching(60_000, 2));
        let texts = ["a risk", "b risk", "c risk", "d risk"];
        let got = futures::future::join_all(texts.iter().map(|t| batched.embed(t))).await;
        assert!(got.iter().all(|g| g.is_ok()));
        assert_eq!(*provider.batch_sizes.lock().unwrap(), vec![2, 2]);
    }

    #[tokio::test]
    async fn a_lone_question_in_a_window_is_a_plain_miss() {
        let provider = CountingEmbedder::stalling(0);
        let batched = embedder(&provider, batching(1, 32));
        assert_eq!(source(&batched, "risk").await, QueryEmbeddingSource::Miss);
        assert_eq!(*provider.batch_sizes.lock().unwrap(), vec![1]);
    }

    #[tokio::test]
    async fn a_failed_batch_fails_every_member_and_caches_nothing() {
        let provider = CountingEmbedder::stalling(0);
        provider.fail_boom.store(true, Ordering::SeqCst);
        let batched = embedder(&provider, batching(30, 32));
        let texts = ["boom risk", "refused", "boom risk"];
        let got = futures::future::join_all(texts.iter().map(|t| batched.embed(t))).await;
        assert_eq!(
            provider.calls(),
            1,
            "identical texts coalesce before batching"
        );
        for result in got {
            assert!(matches!(
                result,
                Err(SandboxDemoError::Embedding(m)) if m.contains("provider is down")
            ));
        }
        assert_eq!(batched.cached_entries().0, 0);

        provider.fail_boom.store(false, Ordering::SeqCst);
        assert_eq!(
            source(&batched, "refused").await,
            QueryEmbeddingSource::Miss
        );
        assert_eq!(provider.calls(), 2);
    }

    #[tokio::test]
    async fn a_cached_vector_is_the_normalized_provider_vector() {
        let provider = CountingEmbedder::stalling(0);
        let cache = embedder(&provider, QueryEmbedConfig::default());
        let off = embedder(&provider, QueryEmbedConfig::disabled());
        let (cold, _) = cache.embed("refused risk risk").await.unwrap();
        let (warm, _) = cache.embed("refused risk risk").await.unwrap();
        let (uncached, _) = off.embed("refused risk risk").await.unwrap();
        assert_eq!(&*cold, &*warm);
        assert_eq!(&*cold, &*uncached);
        let norm: f32 = cold.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
