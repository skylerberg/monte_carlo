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

/// Floating point as its exact bit pattern.
///
/// **`serde_json` does not round-trip `f64`.** Its writer is correct — the text
/// it emits is the shortest decimal that names the value — but its parser is not
/// correctly rounded, and reading that text back can land one unit in the last
/// place away. A checkpoint written and reloaded through `serde_json`'s own
/// numbers is therefore *nearly* the state that was saved, and for a search that
/// compounds its state every generation, nearly is not a resume: one bit in a
/// recombination weight moves the next candidate, which moves everything after
/// it. Storing the bits sidesteps the parser entirely.
///
/// It also carries the infinities for free, which matters because a fresh
/// optimizer's best fitness is negative infinity — a value JSON cannot express
/// as a number at all, and which `serde_json` would otherwise write as `null`
/// and refuse to read back.
///
/// The cost is a checkpoint no one can read. That is the right trade here: this
/// file is machine state for resuming, and the parameters a human wants to look
/// at are written separately, in the open, where a last-place difference cannot
/// matter.
pub mod exact {
    /// One `f64`.
    pub mod scalar {
        use serde::{Deserialize, Deserializer, Serializer};

        pub fn serialize<S: Serializer>(value: &f64, out: S) -> Result<S::Ok, S::Error> {
            out.serialize_u64(value.to_bits())
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(input: D) -> Result<f64, D::Error> {
            Ok(f64::from_bits(u64::deserialize(input)?))
        }
    }

    /// A row of `f64`.
    pub mod vector {
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        pub fn serialize<S: Serializer>(value: &[f64], out: S) -> Result<S::Ok, S::Error> {
            value
                .iter()
                .map(|entry| entry.to_bits())
                .collect::<Vec<_>>()
                .serialize(out)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(input: D) -> Result<Vec<f64>, D::Error> {
            Ok(Vec::<u64>::deserialize(input)?
                .into_iter()
                .map(f64::from_bits)
                .collect())
        }
    }

    /// Rows of `f64`.
    pub mod matrix {
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        pub fn serialize<S: Serializer>(value: &[Vec<f64>], out: S) -> Result<S::Ok, S::Error> {
            value
                .iter()
                .map(|row| row.iter().map(|entry| entry.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
                .serialize(out)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(input: D) -> Result<Vec<Vec<f64>>, D::Error> {
            Ok(Vec::<Vec<u64>>::deserialize(input)?
                .into_iter()
                .map(|row| row.into_iter().map(f64::from_bits).collect())
                .collect())
        }
    }
}
