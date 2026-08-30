//! Offline hyperparameter tuning for games built on [`mcts`].
//!
//! A game's evaluation weights are hand-set numbers, and the only honest way to
//! judge a change to them is to play the new numbers against the old and count.
//! This crate automates that loop: it proposes candidate parameter vectors,
//! measures each one's win rate against a fixed baseline in threaded self-play,
//! and feeds the ranking back to a black-box optimizer.
//!
//! # The thing that actually governs whether this works
//!
//! Fitness here is a win rate over `n` games, so its standard error is
//! `sqrt(0.25 / n)` at worst — 2.5 percentage points even at 400 games. Write
//! `sigma_f` for the spread of *true* strength across a generation's
//! candidates. The correlation between true strength and what the run measures
//! is
//!
//! ```text
//! rho(n, sigma_f) = sigma_f / sqrt(sigma_f^2 + 0.25 / n)
//! ```
//!
//! and every optimizer in here is ranking by the noisy quantity. When `rho` is
//! low the ranking is mostly noise, candidates from the bottom half get bred,
//! and a generation is as likely to move away from the optimum as towards it.
//! Two consequences shape the whole design:
//!
//! - **More games per candidate beats more candidates**, up to a point.
//!   [`Evaluation::games`] is the single most important number in a run, and
//!   values below ~200 are usually wasted compute.
//! - **Cheap variance reduction is worth more than a better optimizer.** Every
//!   candidate in a generation plays the *same* seeds, and every matchup is
//!   played from both seats. See [`arena`].
//!
//! # Layout
//!
//! - [`Tunable`] maps a game's parameter struct to and from a flat gene vector.
//! - [`Match`] tells the driver how to start a game and how to turn genes into
//!   a search context.
//! - [`Optimizer`] is the ask/tell interface the strategies implement:
//!   [`Ga`] and [`CmaEs`].
//! - [`run`] is the loop that ties them together.

mod arena;
mod cma;
mod eigen;
mod ga;
mod optimizer;
mod run;
mod sampling;
mod tunable;

pub use arena::{evaluate, play, Evaluation, Match, MatchOutcome};
pub use cma::{CmaEs, CmaParams};
pub use ga::{Ga, GaParams};
pub use optimizer::Optimizer;
pub use run::{run, GenerationReport, TuneConfig, TuneReport};
pub use tunable::Tunable;
