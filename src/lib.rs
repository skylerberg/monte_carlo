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
//! ```
//! # use mcts::rand_core::Rng;
//! # use mcts::{Game, Status};
//! use mcts::rand_core::SeedableRng;
//! use mcts::{Config, Searcher};
//! # /// Subtraction Nim: take 1 to 3 chips, and taking the last one wins.
//! # struct Nim {
//! #     chips: u32,
//! #     to_move: u8,
//! # }
//! # impl Game for Nim {
//! #     type Choice = u32;
//! #     type Rewards = [f64; 2];
//! #     type Context = ();
//! #     type Side = ();
//! #     fn status(&self, _: &()) -> Status<[f64; 2]> {
//! #         if self.chips == 0 {
//! #             let mut rewards = [0.0; 2];
//! #             rewards[(self.to_move ^ 1) as usize] = 1.0;
//! #             return Status::Terminal(rewards);
//! #         }
//! #         Status::Active { player: self.to_move }
//! #     }
//! #     fn choices_into(&self, _: &(), out: &mut Vec<u32>) {
//! #         out.extend((1..=3).filter(|&take| take <= self.chips));
//! #     }
//! #     fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), take: &u32, _: &mut R) {
//! #         self.chips -= take;
//! #         self.to_move ^= 1;
//! #     }
//! #     fn rollout<R: Rng + ?Sized>(&mut self, ctx: &(), rng: &mut R) -> [f64; 2] {
//! #         while self.chips > 0 {
//! #             let take = 1 + (rng.next_u64() % self.chips.min(3) as u64) as u32;
//! #             self.apply_choice(ctx, &take, rng);
//! #         }
//! #         match self.status(ctx) {
//! #             Status::Terminal(rewards) => rewards,
//! #             _ => unreachable!("no chips left is terminal"),
//! #         }
//! #     }
//! #     fn new_buffer(&self) -> Self {
//! #         Nim { chips: 0, to_move: 0 }
//! #     }
//! #     fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
//! #         dest.chips = self.chips;
//! #         dest.to_move = self.to_move;
//! #     }
//! # }
//! # let mut state = Nim { chips: 12, to_move: 0 };
//! # let (ctx, player) = ((), 0);
//! # let config = Config { iterations: 1_000, ..Config::default() };
//! # let mut rng = wyrand::WyRand::seed_from_u64(0xC0FFEE);
//! let mut searcher = Searcher::new(&state);
//! let result = searcher.search(&state, &ctx, player, &config, None, &mut rng);
//! state.apply_choice(&ctx, &result.choice, &mut rng);
//! searcher.reuse_subtree(&result.choice);   // keep the subtree for next time
//! ```
//!
//! The `Game` those lines run against is hidden, along with its setup;
//! `examples/tic_tac_toe.rs` is a complete implementation with nothing hidden.
//!
//! # What it does
//!
//! * **Information Set MCTS.** A fresh determinization per iteration, and,
//!   below the root, an exploration term based on how often a choice was
//!   *available* rather than how often its parent was visited. The root scores
//!   its children against its own visit count instead — see the comment in
//!   `src/select.rs` for why that one level is deliberately the textbook rule.
//! * **max^n backup** over per-player reward vectors, so any number of players
//!   and any reward scale work without a zero-sum assumption.
//! * **Simultaneous moves** via decoupled UCT, with regret matching as the
//!   default selection rule so that a game with a mixed equilibrium is played
//!   mixed rather than exploitably. See [`SimultaneousPolicy`] for exactly what
//!   converges to what, and what determinization does to that claim.
//! * **Budgets** by iteration count or wall clock — one of the two is
//!   required — either of which an external cancellation flag can cut short.
//!   To run until cancelled and no sooner, set [`Config::iterations`] to
//!   `u32::MAX`; the flag is polled once an iteration. Plus early termination
//!   once the answer provably cannot change, which that sentinel does disable:
//!   the proof asks whether the iterations left could overturn the leader, and
//!   against a budget that large the answer is always yes.
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
mod rank;
mod search;
mod select;
mod util;

#[cfg(feature = "parallel")]
mod parallel;

/// Re-exported so implementors can name the exact `rand_core` version that this
/// crate's `Rng` bounds refer to.
pub use rand_core;

pub use game::{Game, JointChoices, PlayerSet, PlayerSetIter, Rewards, SimultaneousPolicy, Status};
pub use node::{Edge, Marginals, Node, NodeKind};
pub use search::{Config, RootPolicy, SearchResult, Searcher, SimultaneousConfig, StopReason};

#[cfg(feature = "parallel")]
pub use parallel::RootParallel;
