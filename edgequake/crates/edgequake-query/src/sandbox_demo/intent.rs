//! Opt-in intent routing for the sandbox demo.
//!
//! Off by default: with [`INTENT_ROUTING_ENV`] unset every query takes the
//! default path and the response is byte-identical to before the knob
//! existed (no `intent` field at all).
//!
//! With `EDGEQUAKE_SANDBOX_INTENT_ROUTING=next_touch`, a query that matches
//! the fixed [next-touch lexicon](NEXT_TOUCH_CUES) is answered by
//! `final = semantic + λ·structural'`, where `structural` comes from
//! [`INTENT_SCORER_ENV`] (`agent_frequency` default) and `structural'` is
//! the spread-matched value from [`spread_matched`]. Every other query
//! keeps the default path exactly; the response then carries
//! `"intent": null`.
//!
//! # The next-touch lexicon
//!
//! Fixed in code before any evaluation, not tuned on results. The query is
//! lowercased and split on whitespace; each token has non-alphanumeric
//! characters trimmed from both ends (so `agent:claude-code-demo` stays one
//! token and `next?` becomes `next`). A query matches when a cue phrase
//! from [`NEXT_TOUCH_CUES`] and an action word from [`NEXT_TOUCH_ACTIONS`]
//! occur within [`NEXT_TOUCH_WINDOW`] tokens of each other, in either
//! order (`will it touch`, `open next`). Whole tokens only: `reuse` is not
//! `use`, `willing` is not `will`.

use super::error::{Result, SandboxDemoError};
use super::structural::{ScorerKind, StructuralScorer};

/// `off` (default) | `next_touch`.
pub const INTENT_ROUTING_ENV: &str = "EDGEQUAKE_SANDBOX_INTENT_ROUTING";
/// The structural scorer an intent-routed query blends in; same grammar as
/// `EDGEQUAKE_SANDBOX_STRUCTURAL_SCORER`. Unset means `agent_frequency`.
pub const INTENT_SCORER_ENV: &str = "EDGEQUAKE_SANDBOX_INTENT_SCORER";
/// λ, the weight on the spread-matched structural score. Unset means `1.0`.
pub const INTENT_LAMBDA_ENV: &str = "EDGEQUAKE_SANDBOX_INTENT_LAMBDA";

/// The default λ.
pub const DEFAULT_INTENT_LAMBDA: f32 = 1.0;

/// Cue phrases, each a sequence of whole tokens.
pub const NEXT_TOUCH_CUES: &[&[&str]] = &[
    &["will"],
    &["next"],
    &["likely"],
    &["usually"],
    &["typically"],
    &["tends", "to"],
    &["tend", "to"],
    &["going", "to"],
];

/// Action words, with their third-person forms.
pub const NEXT_TOUCH_ACTIONS: &[&str] = &[
    "touch", "touches", "use", "uses", "access", "accesses", "open", "opens", "run", "runs",
    "need", "needs",
];

/// Maximum token distance between the end of a cue and an action word
/// after it, or between an action word and the start of a cue after it.
pub const NEXT_TOUCH_WINDOW: usize = 4;

/// A recognized query intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    /// "What will this agent touch next?"
    NextTouch,
}

impl Intent {
    /// The wire name, as returned in the response's `intent`.
    pub fn name(self) -> &'static str {
        match self {
            Self::NextTouch => "next_touch",
        }
    }
}

/// Whether, and how, queries are routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntentRouting {
    /// Every query takes the default path; no `intent` field.
    #[default]
    Off,
    /// Next-touch queries take the intent blend.
    NextTouch,
}

impl IntentRouting {
    /// Parse an [`INTENT_ROUTING_ENV`] value; blank counts as unset.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim).filter(|v| !v.is_empty()) {
            None | Some("off") => Ok(Self::Off),
            Some("next_touch") => Ok(Self::NextTouch),
            Some(other) => Err(config(format!(
                "{INTENT_ROUTING_ENV}: unknown value {other:?}; expected off or next_touch"
            ))),
        }
    }
}

/// The intent-routing knobs a generation was loaded with.
#[derive(Debug, Clone, PartialEq)]
pub struct IntentConfig {
    /// Whether routing is on.
    pub routing: IntentRouting,
    /// The structural scorer an intent-routed query blends in.
    pub scorer: StructuralScorer,
    /// λ.
    pub lambda: f32,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            routing: IntentRouting::Off,
            scorer: StructuralScorer::Single(ScorerKind::AgentFrequency),
            lambda: DEFAULT_INTENT_LAMBDA,
        }
    }
}

impl IntentConfig {
    /// Build from the three variables' values; blank counts as unset.
    pub fn parse(
        routing: Option<&str>,
        scorer: Option<&str>,
        lambda: Option<&str>,
    ) -> Result<Self> {
        let defaults = Self::default();
        let scorer = match scorer.map(str::trim).filter(|v| !v.is_empty()) {
            Some(v) => StructuralScorer::parse(v)
                .map_err(|e| config(format!("{INTENT_SCORER_ENV}: {e}")))?,
            None => defaults.scorer,
        };
        let lambda = match lambda.map(str::trim).filter(|v| !v.is_empty()) {
            Some(v) => {
                let parsed: f32 = v
                    .parse()
                    .map_err(|_| config(format!("{INTENT_LAMBDA_ENV}: {v:?} is not a number")))?;
                if !parsed.is_finite() || parsed < 0.0 {
                    return Err(config(format!(
                        "{INTENT_LAMBDA_ENV}: must be finite and >= 0, got {parsed}"
                    )));
                }
                parsed
            }
            None => defaults.lambda,
        };
        Ok(Self {
            routing: IntentRouting::parse(routing)?,
            scorer,
            lambda,
        })
    }

    /// Read [`INTENT_ROUTING_ENV`], [`INTENT_SCORER_ENV`], and
    /// [`INTENT_LAMBDA_ENV`].
    pub fn from_env() -> Result<Self> {
        let routing = std::env::var(INTENT_ROUTING_ENV).ok();
        let scorer = std::env::var(INTENT_SCORER_ENV).ok();
        let lambda = std::env::var(INTENT_LAMBDA_ENV).ok();
        Self::parse(routing.as_deref(), scorer.as_deref(), lambda.as_deref())
    }

    /// The intent that routes `query`, or `None` for the default path.
    pub fn route(&self, query: &str) -> Option<Intent> {
        match self.routing {
            IntentRouting::Off => None,
            IntentRouting::NextTouch => is_next_touch(query).then_some(Intent::NextTouch),
        }
    }
}

fn tokens(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Whether `query` matches the next-touch lexicon.
pub fn is_next_touch(query: &str) -> bool {
    let tokens = tokens(query);
    let actions: Vec<usize> = (0..tokens.len())
        .filter(|i| NEXT_TOUCH_ACTIONS.contains(&tokens[*i].as_str()))
        .collect();
    if actions.is_empty() {
        return false;
    }
    NEXT_TOUCH_CUES.iter().any(|cue| {
        tokens
            .windows(cue.len())
            .enumerate()
            .any(|(start, window)| {
                if !window.iter().zip(cue.iter()).all(|(t, c)| t == c) {
                    return false;
                }
                let end = start + cue.len() - 1;
                actions.iter().any(|&a| {
                    (a > end && a - end <= NEXT_TOUCH_WINDOW)
                        || (a < start && start - a <= NEXT_TOUCH_WINDOW)
                })
            })
    })
}

/// The structural values z-scored over the candidate set and rescaled to
/// the semantic scores' standard deviation (population, over the same
/// candidates), so `λ = 1` puts one structural standard deviation on the
/// same footing as one semantic standard deviation. A constant structural
/// source, or a constant semantic score, maps to all `0`: it cannot move
/// the order.
pub fn spread_matched(structural: &[f32], semantic: &[f32]) -> Vec<f32> {
    let (mean, std) = mean_std(structural);
    let (_, semantic_std) = mean_std(semantic);
    structural
        .iter()
        .map(|&s| {
            if std > f64::EPSILON && semantic_std > f64::EPSILON {
                ((f64::from(s) - mean) / std * semantic_std) as f32
            } else {
                0.0
            }
        })
        .collect()
}

fn mean_std(values: &[f32]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let var = values
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

fn config(message: String) -> SandboxDemoError {
    SandboxDemoError::Config(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_touch_questions_match() {
        for query in [
            "What will claude-code-demo touch next?",
            "Which resources is agent:claude-code-demo likely to access?",
            "What does this agent usually need?",
            "Which files does it typically open?",
            "What is it going to run?",
            "Which binaries does it tend to use?",
            "What tends to get accessed... what does it tends to access",
            "WHAT WILL IT TOUCH",
            "Which file does it open next?",
            "will it touch /usr/bin/wc",
            "Next, which directory will the agent most likely access",
        ] {
            assert!(is_next_touch(query), "{query}");
        }
    }

    #[test]
    fn other_questions_do_not_match() {
        for query in [
            "Which governed resources show the riskiest access pattern? Refused, risky access.",
            "Which resources were denied?",
            "Who will approve this?",
            "What did it touch last week?",
            "Will the reuse policy change?",
            "Is it willing to open source this?",
            "usage of the next release",
            "touch",
            "",
            "will a b c d e touch",
            "it opened the file and will be fine",
        ] {
            assert!(!is_next_touch(query), "{query}");
        }
    }

    #[test]
    fn routing_parses_and_defaults_off() {
        for value in [None, Some(""), Some("  "), Some("off")] {
            assert_eq!(IntentRouting::parse(value).unwrap(), IntentRouting::Off);
        }
        assert_eq!(
            IntentRouting::parse(Some(" next_touch ")).unwrap(),
            IntentRouting::NextTouch
        );
        for bad in ["on", "Next_Touch", "1"] {
            assert!(IntentRouting::parse(Some(bad)).is_err(), "{bad}");
        }
        let config = IntentConfig::parse(None, None, None).unwrap();
        assert_eq!(config, IntentConfig::default());
        assert_eq!(config.route("What will it touch next?"), None);
        assert_eq!(config.scorer.label(), "agent_frequency");
        assert_eq!(config.lambda, 1.0);
    }

    #[test]
    fn scorer_and_lambda_parse_and_validate() {
        let config = IntentConfig::parse(
            Some("next_touch"),
            Some("blend:popularity=0.5,agent_frequency=0.5"),
            Some("0.25"),
        )
        .unwrap();
        assert_eq!(
            config.route("What will it touch next?"),
            Some(Intent::NextTouch)
        );
        assert_eq!(config.route("Which resources were denied?"), None);
        assert_eq!(
            config.scorer.label(),
            "blend:popularity=0.5,agent_frequency=0.5"
        );
        assert_eq!(config.lambda, 0.25);
        let popular = IntentConfig::parse(Some("next_touch"), Some("popularity"), None).unwrap();
        assert_eq!(popular.scorer.label(), "popularity");
        for (scorer, lambda) in [
            (Some("pagerank"), None),
            (None, Some("-1")),
            (None, Some("NaN")),
            (None, Some("inf")),
            (None, Some("x")),
        ] {
            let err = IntentConfig::parse(Some("next_touch"), scorer, lambda).unwrap_err();
            assert!(matches!(err, SandboxDemoError::Config(_)), "{err:?}");
        }
    }

    #[test]
    fn spread_matching_rescales_z_scores_to_the_semantic_std() {
        // structural mean 2, population std sqrt(2/3); semantic std 0.1.
        let structural = [1.0, 2.0, 3.0];
        let semantic = [0.4, 0.5, 0.6];
        let out = spread_matched(&structural, &semantic);
        let sem_std = (0.02f64 / 3.0).sqrt();
        let z = 1.0 / (2.0f64 / 3.0).sqrt();
        let expected = [-z * sem_std, 0.0, z * sem_std];
        for (got, want) in out.iter().zip(expected) {
            assert!((f64::from(*got) - want).abs() < 1e-6, "{out:?}");
        }
        // The rescaled values carry the semantic spread and zero mean.
        let (mean, std) = mean_std(&out);
        assert!(mean.abs() < 1e-7);
        assert!((std - sem_std).abs() < 1e-6);
        // Affine changes to the structural source do not change the result.
        let shifted: Vec<f32> = structural.iter().map(|s| s * 10.0 + 7.0).collect();
        for (a, b) in spread_matched(&shifted, &semantic).iter().zip(&out) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn spread_matching_maps_degenerate_inputs_to_zero() {
        assert_eq!(spread_matched(&[0.3, 0.3], &[0.1, 0.9]), vec![0.0, 0.0]);
        assert_eq!(spread_matched(&[0.1, 0.9], &[0.5, 0.5]), vec![0.0, 0.0]);
        assert_eq!(spread_matched(&[0.7], &[0.2]), vec![0.0]);
        assert!(spread_matched(&[], &[]).is_empty());
    }
}
