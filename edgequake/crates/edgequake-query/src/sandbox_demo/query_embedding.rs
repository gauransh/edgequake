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
//! - **Rollback.** `EDGEQUAKE_SANDBOX_QUERY_EMBED_CACHE=0` restores the
//!   original one-provider-call-per-query path exactly.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};

use edgequake_llm::traits::EmbeddingProvider;
use futures::future::{BoxFuture, FutureExt, Shared};

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
}

impl Default for QueryEmbedConfig {
    fn default() -> Self {
        Self {
            cache_entries: DEFAULT_QUERY_EMBED_CACHE_ENTRIES,
            cache_bytes: DEFAULT_QUERY_EMBED_CACHE_BYTES,
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
        }
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
type InFlight = Shared<BoxFuture<'static, Outcome>>;

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
        let vector = shared.await.map_err(SandboxDemoError::Embedding)?;
        Ok((vector, source))
    }

    /// One provider request for `key`, settling its own in-flight slot.
    fn call(&self, key: EmbedKey) -> InFlight {
        let inner = self.inner.clone();
        async move {
            let outcome = inner.fetch(&key.text).await;
            let mut state = inner.state();
            state.in_flight.remove(&key);
            if let Ok(vector) = &outcome {
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
