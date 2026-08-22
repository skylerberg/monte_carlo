//! Monte Carlo Tree Search for board games, including imperfect-information
//! games via Information Set MCTS.
//!
//! # Shape of the library
//!
//! A game implements [`Game`]. A [`Searcher`] owns the reusable buffers and the
//! retained tree, and [`Config`] holds the knobs. There is no policy trait to
//! override: everything the search varies on is either a config field or a hook
//! on your own game type, which keeps the hot path monomorphic and inlinable.
//!
//! ```ignore
//! let mut searcher = Searcher::new(&state);
//! let result = searcher.search(&state, &ctx, player, &config, None, &mut rng);
//! state.apply(&result.choice);
//! searcher.reuse_subtree(&result.choice);   // keep the subtree for next time
//! ```
//!
//! # What it does
//!
//! * **Information Set MCTS.** A fresh determinization per iteration, and an
//!   exploration term based on how often a choice was *available* rather than
//!   how often its parent was visited.
//! * **max^n backup** over per-player reward vectors, so any number of players
//!   and any reward scale work without a zero-sum assumption.
//! * **Simultaneous moves** via decoupled UCT, with regret matching as the
//!   default selection rule so that a game with a mixed equilibrium is played
//!   mixed rather than exploitably. See [`SimultaneousPolicy`] for exactly what
//!   converges to what, and what determinization does to that claim.
//! * **Budgets** by iteration count, wall clock, or an external cancellation
//!   flag, and early termination once the answer provably cannot change.
//! * **Tree reuse** across moves, and root parallelism behind the `parallel`
//!   feature.
//!
//! # Allocation
//!
//! A steady-state iteration allocates nothing but tree growth. Determinization
//! writes into one scratch state; move generation fills one reused buffer; each
//! `Choice` is stored once per edge.

#![warn(missing_docs)]

mod budget;
mod duct;
mod early_stop;
mod game;
mod node;
mod search;
mod select;
mod util;

#[cfg(feature = "parallel")]
mod parallel;

/// Re-exported so implementors can name the exact `rand_core` version that this
/// crate's `Rng` bounds refer to.
pub use rand_core;

pub use game::{Game, JointChoices, PlayerSet, PlayerSetIter, Rewards, SimultaneousPolicy, Status};
pub use node::{Marginals, Node, NodeKind};
pub use search::{Config, RootPolicy, SearchResult, Searcher, StopReason};

#[cfg(feature = "parallel")]
pub use parallel::RootParallel;
