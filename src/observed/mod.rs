//! Search over one tree per player, for games that implement
//! [`crate::Observed`].
//!
//! A separate searcher rather than a mode of [`crate::Searcher`], so that a game
//! which conceals nothing from its opponents pays nothing for the ability: the
//! tree's node, descent and backup are untouched, and none of what a second
//! observer needs — an edge identified by a perception rather than by a move, a
//! tree per player, a determinization per observer — is in them.
//!
//! # What it is for
//!
//! A move whose value depends on what it conceals is worth what the opponent
//! believes about it. [`Searcher`](crate::Searcher) cannot express that. Its
//! edges are identified by [`Game::Choice`](crate::Game::Choice), so two moves
//! that conceal different things are two subtrees, and the simulated opponent
//! answering inside one of them has only ever been shown that one concealed
//! value. The opponent then responds to the move it can see through, and a
//! deception scores no better than disclosing.
//!
//! [`ObservedSearcher`] identifies an edge by
//! [`Observed::observed_key`](crate::Observed::observed_key) instead: the move
//! as one player perceives it. Moves that player cannot tell apart lead to one
//! node, whose statistics are the mixture over what the move might have been —
//! that player's information set, and the belief a bluff is played against. One
//! such tree is kept per player, each player's own moves are chosen from its own
//! tree, and every tree is advanced along the same sampled world so their paths
//! stay aligned.
//!
//! # What it does not do
//!
//! * **Sequential decisions only.** A state reporting
//!   [`Status::Simultaneous`](crate::Status) is refused rather than searched
//!   wrongly: concurrent moves need the arm-by-arm treatment in
//!   [`Searcher`](crate::Searcher), and a game with both kinds of node is not
//!   supported here.
//! * **No transposition merging.** One searcher per notion of node identity: a
//!   graph over per-observer keys is a different searcher again, and stacking
//!   two merges is how a tree stops being explainable.
//! * **No early termination.** The proof in [`crate::early_stop`] is about one
//!   tree's candidates; the observed searcher's answer is read from the
//!   perspective player's tree while other trees grow alongside it. Every search
//!   here runs to its budget, so [`Config::early_termination`] is ignored rather
//!   than honoured against the wrong statistics.
//! * **No fixed-choice fast path.** `Game::choices_fixed` is not consulted:
//!   an edge is identified by a key, the key is not stored with the move, so
//!   the moves have to be enumerated to find out which one the chosen key
//!   belongs to. Every other [`Config`](crate::Config) knob is honoured.
//!
//! `OBSERVERS.md` in the repository has the algorithm, the key contract, the
//! measurements from the bluffing fixture, and the alternatives considered.

mod node;
mod search;

pub use node::ObservedNode;
pub use search::ObservedSearcher;
