use serde::{Deserialize, Serialize};

/// Why a checkpoint could not be used to continue a run.
///
/// Every variant is a case where continuing would silently measure something
/// other than what the original run measured. Resuming wrong is worse than not
/// resuming: the numbers keep coming, they look like the old ones, and nothing
/// says they are answering a different question.
#[derive(Debug, Clone, PartialEq)]
pub enum ResumeError {
    /// The snapshot is not the shape this optimizer writes.
    Malformed(String),
    /// The checkpoint came from a different strategy. CMA-ES state cannot seed
    /// a GA, and a run that silently started over would look like one that
    /// resumed.
    Strategy {
        expected: &'static str,
        found: String,
    },
    /// The parameters have a different number of genes than when the run
    /// started — a weight was added to the game, or pinned, since.
    Dimension { expected: usize, found: usize },
    /// The baseline differs from the one the checkpoint was measured against.
    ///
    /// The likeliest way to hit this is resuming with the previous run's
    /// *output* as `--seed-params`. Fitness is a win rate against the baseline,
    /// so that would restart the scale at even money and make every number
    /// after the resume incomparable with every number before it — while
    /// looking, in the log, exactly like a run that collapsed.
    Baseline { expected: usize, found: usize },
}

impl std::fmt::Display for ResumeError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(detail) => write!(out, "the checkpoint is not readable: {detail}"),
            Self::Strategy { expected, found } => write!(
                out,
                "the checkpoint was written by `{found}` but this run uses `{expected}`; \
                 resume with the same strategy or start a fresh run"
            ),
            Self::Dimension { expected, found } => write!(
                out,
                "the checkpoint holds {found} genes and these parameters have {expected}; \
                 the tuned parameters changed since the run started"
            ),
            Self::Baseline { expected, found } => write!(
                out,
                "the checkpoint was measured against a baseline of {found} genes and this run's \
                 baseline has {expected}, or the values differ. Fitness is a win rate against \
                 the baseline, so resuming would change what every number means. Resume with \
                 the same --seed-params the original run used, not with its output"
            ),
        }
    }
}

impl std::error::Error for ResumeError {}

/// An optimizer's state, tagged so a mismatched resume fails loudly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub strategy: String,
    pub dimension: usize,
    pub state: serde_json::Value,
}

impl Snapshot {
    pub fn new<T: Serialize>(strategy: &'static str, dimension: usize, state: &T) -> Self {
        Self {
            strategy: strategy.to_string(),
            dimension,
            state: serde_json::to_value(state).expect("optimizer state serializes"),
        }
    }

    /// Read the state back, checking it belongs to this strategy first.
    pub fn open<T: for<'de> Deserialize<'de>>(
        value: &serde_json::Value,
        strategy: &'static str,
        dimension: usize,
    ) -> Result<T, ResumeError> {
        let snapshot: Snapshot = serde_json::from_value(value.clone())
            .map_err(|error| ResumeError::Malformed(error.to_string()))?;
        if snapshot.strategy != strategy {
            return Err(ResumeError::Strategy {
                expected: strategy,
                found: snapshot.strategy,
            });
        }
        if snapshot.dimension != dimension {
            return Err(ResumeError::Dimension {
                expected: dimension,
                found: snapshot.dimension,
            });
        }
        serde_json::from_value(snapshot.state)
            .map_err(|error| ResumeError::Malformed(error.to_string()))
    }
}

/// `f64::NEG_INFINITY` as JSON `null`, which is the only value here that JSON
/// cannot carry.
///
/// A fresh optimizer's best fitness is negative infinity — nothing measured yet
/// — and `serde_json` writes that as `null` and then refuses to read it back as
/// a number. Without this a checkpoint taken before the first generation would
/// write cleanly and fail to load.
pub mod maybe_infinite {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, out: S) -> Result<S::Ok, S::Error> {
        match value.is_finite() {
            true => out.serialize_some(value),
            false => out.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(input: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(input)?.unwrap_or(f64::NEG_INFINITY))
    }
}
