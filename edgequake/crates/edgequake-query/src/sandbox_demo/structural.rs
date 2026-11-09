//! Selection of the structural component and the weights of the sandbox
//! blend.
//!
//! By default the blend is `semantic=1.0, structural=0.0`. On a
//! pre-registered gold relevance set, semantic-only ranking beat the
//! bundle's frozen `0.6 / 0.4` blend by +0.113 nDCG@5 on holdout (CI
//! `[+0.030, +0.214]`), and no structural arm tried beat semantic-only
//! (crystalos-ai/itsm-graph-core #226/#227). graph-core still loads,
//! validates, and reranks every query: the structural score is computed and
//! returned for transparency, it just carries no weight.
//!
//! - [`BLEND_WEIGHTS_ENV`]: unset → `semantic=1,structural=0` (source
//!   `default`); `bundle` → the bundle's frozen weights from
//!   `activation.json`, byte-identical to the runtime's own rerank (source
//!   `bundle`, the rollback switch); `semantic=<w>,structural=<w>` → those
//!   weights (source `env`).
//! - [`STRUCTURAL_SCORER_ENV`]: `cosine` (default) | `popularity` |
//!   `agent_frequency` | `knn` | `blend:<name>=<w>,...`.
//!
//! Anything other than `bundle` with `cosine` re-blends through
//! graph-core's `rerank_with_scores`, so the clamp, weight validation, and
//! tie-break are the production ones.

use super::error::{Result, SandboxDemoError};

/// Selects the structural scorer. Unset means `cosine`.
pub const STRUCTURAL_SCORER_ENV: &str = "EDGEQUAKE_SANDBOX_STRUCTURAL_SCORER";
/// Selects the blend weights. Unset means [`BlendWeightsChoice::Default`].
pub const BLEND_WEIGHTS_ENV: &str = "EDGEQUAKE_SANDBOX_BLEND_WEIGHTS";
/// The [`BLEND_WEIGHTS_ENV`] value that restores the bundle's weights.
pub const BUNDLE_WEIGHTS: &str = "bundle";

const WEIGHT_SUM_TOLERANCE: f32 = 1e-3;

/// One structural evidence source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScorerKind {
    /// The runtime's anchor-to-candidate cosine, mapped to `[0, 1]`.
    Cosine,
    /// Distinct runs touching the resource, min-max over every resource.
    Popularity,
    /// Distinct runs *of the anchor* touching the resource, min-max over
    /// every resource.
    AgentFrequency,
    /// Mean runtime structural score of the candidate against every
    /// resource the anchor's runs touched (the candidate itself included
    /// when it is one of them); `0` when the anchor touched nothing.
    Knn,
}

impl ScorerKind {
    fn parse(name: &str) -> Result<Self> {
        match name {
            "cosine" => Ok(Self::Cosine),
            "popularity" => Ok(Self::Popularity),
            "agent_frequency" => Ok(Self::AgentFrequency),
            "knn" => Ok(Self::Knn),
            other => Err(config(format!(
                "{STRUCTURAL_SCORER_ENV}: unknown scorer {other:?}; expected cosine, \
                 popularity, agent_frequency, knn, or blend:<name>=<w>,..."
            ))),
        }
    }

    /// The name accepted by [`STRUCTURAL_SCORER_ENV`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::Popularity => "popularity",
            Self::AgentFrequency => "agent_frequency",
            Self::Knn => "knn",
        }
    }
}

/// The structural scorer in force.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuralScorer {
    /// One source, used as-is (each is already in `[0, 1]`).
    Single(ScorerKind),
    /// `Σ w · minmax(source)`, min-max over the candidate set (a constant
    /// source maps to `0`), weights `>= 0` summing to ~1.
    Blend(Vec<(ScorerKind, f32)>),
}

impl Default for StructuralScorer {
    fn default() -> Self {
        Self::Single(ScorerKind::Cosine)
    }
}

impl StructuralScorer {
    /// Parse a [`STRUCTURAL_SCORER_ENV`] value.
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        let Some(spec) = value.strip_prefix("blend:") else {
            return ScorerKind::parse(value).map(Self::Single);
        };
        let mut parts: Vec<(ScorerKind, f32)> = Vec::new();
        for (name, weight) in parse_pairs(STRUCTURAL_SCORER_ENV, spec)? {
            let kind = ScorerKind::parse(&name)?;
            if parts.iter().any(|(k, _)| *k == kind) {
                return Err(config(format!(
                    "{STRUCTURAL_SCORER_ENV}: {name} appears twice in the blend"
                )));
            }
            parts.push((kind, weight));
        }
        require_unit_sum(STRUCTURAL_SCORER_ENV, parts.iter().map(|(_, w)| *w))?;
        Ok(Self::Blend(parts))
    }

    /// Whether `kind` feeds this scorer, so callers can skip the rest.
    pub fn uses(&self, kind: ScorerKind) -> bool {
        match self {
            Self::Single(k) => *k == kind,
            Self::Blend(parts) => parts.iter().any(|(k, _)| *k == kind),
        }
    }

    /// The canonical spelling, as echoed in responses and logs.
    pub fn label(&self) -> String {
        match self {
            Self::Single(kind) => kind.name().to_string(),
            Self::Blend(parts) => format!(
                "blend:{}",
                parts
                    .iter()
                    .map(|(k, w)| format!("{}={w}", k.name()))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }

    /// The structural score of every candidate, given each source's
    /// per-candidate values.
    pub fn combine(
        &self,
        candidates: usize,
        mut source: impl FnMut(ScorerKind) -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Single(kind) => source(*kind),
            Self::Blend(parts) => {
                let mut out = vec![0.0f32; candidates];
                for (kind, weight) in parts {
                    for (slot, value) in out.iter_mut().zip(min_max(&source(*kind)?)) {
                        *slot += weight * value;
                    }
                }
                Ok(out)
            }
        }
    }
}

/// `semantic` / `structural` weights from [`BLEND_WEIGHTS_ENV`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightOverride {
    /// Weight on the semantic score.
    pub semantic: f32,
    /// Weight on the structural score.
    pub structural: f32,
}

impl WeightOverride {
    /// Parse `semantic=<w>,structural=<w>`; both keys are required.
    pub fn parse(value: &str) -> Result<Self> {
        let (mut semantic, mut structural) = (None, None);
        for (name, weight) in parse_pairs(BLEND_WEIGHTS_ENV, value)? {
            let slot = match name.as_str() {
                "semantic" => &mut semantic,
                "structural" => &mut structural,
                other => {
                    return Err(config(format!(
                        "{BLEND_WEIGHTS_ENV}: unknown key {other:?}; expected semantic and \
                         structural"
                    )))
                }
            };
            if slot.replace(weight).is_some() {
                return Err(config(format!("{BLEND_WEIGHTS_ENV}: {name} appears twice")));
            }
        }
        let (Some(semantic), Some(structural)) = (semantic, structural) else {
            return Err(config(format!(
                "{BLEND_WEIGHTS_ENV}: both semantic=<w> and structural=<w> are required"
            )));
        };
        require_unit_sum(BLEND_WEIGHTS_ENV, [semantic, structural])?;
        Ok(Self {
            semantic,
            structural,
        })
    }
}

/// Which blend weights are in force.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BlendWeightsChoice {
    /// `semantic=1.0, structural=0.0`; [`BLEND_WEIGHTS_ENV`] unset.
    #[default]
    Default,
    /// The bundle's frozen weights; [`BLEND_WEIGHTS_ENV`] = `bundle`.
    Bundle,
    /// Explicit `semantic=<w>,structural=<w>`.
    Env(WeightOverride),
}

impl BlendWeightsChoice {
    /// The weights the default applies.
    pub const SEMANTIC_ONLY: WeightOverride = WeightOverride {
        semantic: 1.0,
        structural: 0.0,
    };

    /// Parse a [`BLEND_WEIGHTS_ENV`] value; blank counts as unset.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim).filter(|v| !v.is_empty()) {
            None => Ok(Self::Default),
            Some(BUNDLE_WEIGHTS) => Ok(Self::Bundle),
            Some(v) => WeightOverride::parse(v).map(Self::Env),
        }
    }

    /// The semantic/structural pair, or `None` for the bundle's.
    pub fn weights(self) -> Option<WeightOverride> {
        match self {
            Self::Default => Some(Self::SEMANTIC_ONLY),
            Self::Bundle => None,
            Self::Env(w) => Some(w),
        }
    }

    /// `default` | `bundle` | `env`, as logged at startup.
    pub fn source(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Bundle => "bundle",
            Self::Env(_) => "env",
        }
    }
}

/// The scoring knobs a generation was loaded with.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScoringConfig {
    /// The structural scorer.
    pub scorer: StructuralScorer,
    /// The blend weights.
    pub weights: BlendWeightsChoice,
    /// True when the scorer or explicit weights were set, so responses echo
    /// the scorer. `bundle` alone does not count: the rollback reproduces
    /// the pre-default response byte for byte.
    pub explicit: bool,
}

impl ScoringConfig {
    /// Build from the two variables' values; blank counts as unset.
    pub fn parse(scorer: Option<&str>, weights: Option<&str>) -> Result<Self> {
        let scorer = scorer.filter(|v| !v.trim().is_empty());
        let weights = BlendWeightsChoice::parse(weights)?;
        Ok(Self {
            explicit: scorer.is_some() || matches!(weights, BlendWeightsChoice::Env(_)),
            scorer: scorer
                .map(StructuralScorer::parse)
                .transpose()?
                .unwrap_or_default(),
            weights,
        })
    }

    /// Read [`STRUCTURAL_SCORER_ENV`] and [`BLEND_WEIGHTS_ENV`].
    pub fn from_env() -> Result<Self> {
        let scorer = std::env::var(STRUCTURAL_SCORER_ENV).ok();
        let weights = std::env::var(BLEND_WEIGHTS_ENV).ok();
        Self::parse(scorer.as_deref(), weights.as_deref())
    }

    /// True when the runtime's own rerank is the answer, untouched: the
    /// cosine scorer under the bundle's weights.
    pub fn is_runtime_rerank(&self) -> bool {
        self.scorer == StructuralScorer::default() && self.weights == BlendWeightsChoice::Bundle
    }
}

/// Min-max to `[0, 1]`; a constant (or empty) input maps to all `0`,
/// matching the offline evaluation's normalization.
pub fn min_max(values: &[f32]) -> Vec<f32> {
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    values
        .iter()
        .map(|v| if hi > lo { (v - lo) / (hi - lo) } else { 0.0 })
        .collect()
}

fn config(message: String) -> SandboxDemoError {
    SandboxDemoError::Config(message)
}

fn parse_pairs(var: &str, spec: &str) -> Result<Vec<(String, f32)>> {
    spec.split(',')
        .map(|pair| {
            let (name, weight) = pair
                .split_once('=')
                .ok_or_else(|| config(format!("{var}: expected <name>=<weight>, got {pair:?}")))?;
            let weight: f32 = weight.trim().parse().map_err(|_| {
                config(format!(
                    "{var}: weight {weight:?} for {name:?} is not a number"
                ))
            })?;
            if !weight.is_finite() || weight < 0.0 {
                return Err(config(format!(
                    "{var}: weight for {name:?} must be finite and >= 0, got {weight}"
                )));
            }
            Ok((name.trim().to_string(), weight))
        })
        .collect()
}

fn require_unit_sum(var: &str, weights: impl IntoIterator<Item = f32>) -> Result<()> {
    let sum: f32 = weights.into_iter().sum();
    if (sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
        return Err(config(format!(
            "{var}: weights must sum to ~1.0, got {sum}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_and_blank_are_semantic_only() {
        for (scorer, weights) in [(None, None), (Some(""), Some("  "))] {
            let config = ScoringConfig::parse(scorer, weights).unwrap();
            assert_eq!(config, ScoringConfig::default());
            assert_eq!(config.weights, BlendWeightsChoice::Default);
            assert_eq!(config.weights.source(), "default");
            assert_eq!(
                config.weights.weights(),
                Some(WeightOverride {
                    semantic: 1.0,
                    structural: 0.0
                })
            );
            assert!(!config.is_runtime_rerank());
            assert!(!config.explicit);
        }
    }

    #[test]
    fn bundle_is_the_runtime_rerank_and_not_echoed() {
        for value in ["bundle", " bundle "] {
            let config = ScoringConfig::parse(None, Some(value)).unwrap();
            assert_eq!(config.weights, BlendWeightsChoice::Bundle);
            assert_eq!(config.weights.source(), "bundle");
            assert_eq!(config.weights.weights(), None);
            assert!(config.is_runtime_rerank());
            assert!(!config.explicit);
        }
        let echoed = ScoringConfig::parse(Some("cosine"), Some("bundle")).unwrap();
        assert!(echoed.is_runtime_rerank());
        assert!(echoed.explicit);
        let popular = ScoringConfig::parse(Some("popularity"), Some("bundle")).unwrap();
        assert!(!popular.is_runtime_rerank());
    }

    #[test]
    fn explicit_cosine_is_semantic_only_but_echoed() {
        let config = ScoringConfig::parse(Some("cosine"), None).unwrap();
        assert_eq!(config.weights, BlendWeightsChoice::Default);
        assert!(!config.is_runtime_rerank());
        assert!(config.explicit);
    }

    #[test]
    fn every_single_scorer_parses() {
        for name in ["cosine", "popularity", "agent_frequency", "knn"] {
            let scorer = StructuralScorer::parse(name).unwrap();
            assert_eq!(scorer.label(), name);
        }
    }

    #[test]
    fn blends_parse_and_round_trip() {
        let scorer = StructuralScorer::parse("blend:knn=0.5, popularity=0.5").unwrap();
        assert_eq!(
            scorer,
            StructuralScorer::Blend(vec![(ScorerKind::Knn, 0.5), (ScorerKind::Popularity, 0.5)])
        );
        assert_eq!(scorer.label(), "blend:knn=0.5,popularity=0.5");
        assert!(scorer.uses(ScorerKind::Knn));
        assert!(!scorer.uses(ScorerKind::Cosine));
    }

    #[test]
    fn bad_scorers_are_refused() {
        for bad in [
            "pagerank",
            "blend:",
            "blend:knn",
            "blend:knn=0.5,popularity=0.6",
            "blend:knn=-0.5,popularity=1.5",
            "blend:knn=0.5,knn=0.5",
            "blend:knn=x,popularity=1",
            "blend:blend=1",
        ] {
            let err = StructuralScorer::parse(bad).unwrap_err();
            assert!(matches!(err, SandboxDemoError::Config(_)), "{bad}: {err:?}");
        }
    }

    #[test]
    fn weight_overrides_are_validated() {
        let ok = WeightOverride::parse("semantic=0.25,structural=0.75").unwrap();
        assert_eq!(
            ok,
            WeightOverride {
                semantic: 0.25,
                structural: 0.75
            }
        );
        for bad in [
            "semantic=0.5",
            "semantic=0.5,structural=0.6",
            "semantic=-0.5,structural=1.5",
            "semantic=0.5,structural=0.5,graph=0",
            "semantic=0.5,semantic=0.5",
            "semantic=NaN,structural=1",
            "0.5,0.5",
            "Bundle",
            "bundle,semantic=1",
        ] {
            let err = WeightOverride::parse(bad).unwrap_err();
            assert!(matches!(err, SandboxDemoError::Config(_)), "{bad}: {err:?}");
            let err = ScoringConfig::parse(None, Some(bad)).unwrap_err();
            assert!(matches!(err, SandboxDemoError::Config(_)), "{bad}: {err:?}");
        }
        let config = ScoringConfig::parse(None, Some("semantic=0.3,structural=0.7")).unwrap();
        assert_eq!(
            config.weights,
            BlendWeightsChoice::Env(WeightOverride {
                semantic: 0.3,
                structural: 0.7
            })
        );
        assert_eq!(config.weights.source(), "env");
        assert!(!config.is_runtime_rerank());
        assert!(config.explicit);
    }

    #[test]
    fn min_max_maps_constants_to_zero() {
        assert_eq!(min_max(&[2.0, 4.0, 3.0]), vec![0.0, 1.0, 0.5]);
        assert_eq!(min_max(&[1.0, 1.0]), vec![0.0, 0.0]);
        assert!(min_max(&[]).is_empty());
    }

    #[test]
    fn a_blend_is_the_weighted_sum_of_min_maxed_sources() {
        let scorer = StructuralScorer::parse("blend:cosine=0.25,popularity=0.75").unwrap();
        let out = scorer
            .combine(3, |kind| {
                Ok(match kind {
                    ScorerKind::Cosine => vec![0.5, 0.6, 0.7],
                    ScorerKind::Popularity => vec![1.0, 0.0, 0.0],
                    _ => unreachable!(),
                })
            })
            .unwrap();
        let expected = [0.75, 0.125, 0.25];
        for (got, want) in out.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{out:?}");
        }
    }
}
