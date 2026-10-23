//! A game with one concealed move, deliberately tiny.
//!
//! Seat 0 holds two cards, a weak one and a strong one, and makes one decision:
//! conceal one of them (revealing the other) or reveal both. Concealing leaves
//! seat 1 a single decision — call or pass — and the score depends on what was
//! concealed.
//!
//! The payoffs are chosen so that bluffing is part of the equilibrium, and so
//! that the two searchers have different equilibrium values:
//!
//! | seat 0 plays | call | pass |
//! |---|---|---|
//! | conceal the strong card (reveals the weak) | 0.6667 | 0.4667 |
//! | conceal the weak card (reveals the strong) | 0.3333 | 0.7333 |
//! | reveal both | 0.4000 | 0.4000 |
//!
//! Seat 1 prefers to pass against a concealed strong card (its reward is
//! `1 - score`) and to call against a concealed weak one, so a searcher that can
//! see through the concealment makes bluffing worthless: seat 0's best is then
//! `conceal the strong card` worth 0.4667. A searcher that cannot picks the
//! mixed equilibrium — conceal the weak card a third of the time, and seat 1
//! calls four ninths of the time — worth 0.5556 to seat 0.
//!
//! Nothing is hidden from the crate's point of view except through
//! [`Observed::observed_key`]: the game is deterministic and
//! `determinize_into` copies. That is the point. The concealed card is a secret
//! the *move* carries, not a card the deal hides, so the only thing standing
//! between a simulated opponent and the truth is the key.

#![allow(dead_code)]

use mcts::rand_core::Rng;
use mcts::{Game, Observed, Status};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Move {
    /// Conceal card `i`, revealing the other.
    Hide(usize),
    /// Reveal card `i`, and with it everything seat 0 holds.
    Reveal(usize),
    Call,
    Pass,
}

/// What one player sees of another's move.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    /// The move, exactly as made: the acting player's own view, and every
    /// observer's view of a move that hides nothing.
    Exact(Move),
    /// "A card, face down." Which card it is does not enter the key, which is
    /// what merges the two concealments into one information set.
    Hidden,
}

/// Seat 0's strong card is index 1.
const STRONG: usize = 1;

const REVEALED_STRONG: f64 = 1.0 / 3.0;
const REVEALED_WEAK: f64 = 1.0 / 15.0;
const CONCEALED_PASS: f64 = 0.4;
const CONCEALED_CALL_STRONG: f64 = 0.6;
const CONCEALED_CALL_WEAK: f64 = 0.0;

#[derive(Clone, Debug)]
pub struct Bluff {
    move0: Option<Move>,
    answer: Option<Move>,
    to_move: u8,
    over: bool,
}

impl Bluff {
    pub fn new() -> Self {
        Self {
            move0: None,
            answer: None,
            to_move: 0,
            over: false,
        }
    }

    fn revealed(card: usize) -> f64 {
        if card == STRONG {
            REVEALED_STRONG
        } else {
            REVEALED_WEAK
        }
    }

    fn score(&self) -> f64 {
        match self.move0.expect("a finished game has seat 0's move") {
            Move::Reveal(_) => Self::revealed(0) + Self::revealed(STRONG),
            Move::Hide(concealed) => {
                let shown = Self::revealed(1 - concealed);
                match self.answer.expect("a concealment is answered") {
                    Move::Pass => shown + CONCEALED_PASS,
                    Move::Call if concealed == STRONG => shown + CONCEALED_CALL_STRONG,
                    Move::Call => shown + CONCEALED_CALL_WEAK,
                    _ => unreachable!("only call and pass answer a concealment"),
                }
            }
            _ => unreachable!("seat 0 conceals or reveals"),
        }
    }
}

impl Game for Bluff {
    type Choice = Move;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _ctx: &()) -> Status<[f64; 2]> {
        if self.over {
            let score = self.score();
            return Status::Terminal([score, 1.0 - score]);
        }
        Status::Active {
            player: self.to_move,
        }
    }

    fn choices_into(&self, _ctx: &(), out: &mut Vec<Move>) {
        if self.over {
            return;
        }
        if self.to_move == 0 {
            out.extend([
                Move::Hide(0),
                Move::Hide(1),
                Move::Reveal(0),
                Move::Reveal(1),
            ]);
        } else {
            out.extend([Move::Call, Move::Pass]);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _ctx: &(), choice: &Move, _rng: &mut R) {
        match *choice {
            Move::Hide(card) => {
                self.move0 = Some(*choice);
                self.to_move = 1;
                debug_assert!(card <= 1);
            }
            Move::Reveal(_) => {
                self.move0 = Some(*choice);
                self.to_move = 0;
                self.over = true;
            }
            Move::Call | Move::Pass => {
                self.answer = Some(*choice);
                self.to_move = 0;
                self.over = true;
            }
        }
    }

    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &(), rng: &mut R) -> [f64; 2] {
        let mut choices = Vec::new();
        while !self.over {
            self.choices_into(ctx, &mut choices);
            let pick = choices[(rng.next_u64() % choices.len() as u64) as usize];
            self.apply_choice(ctx, &pick, rng);
            choices.clear();
        }
        let score = self.score();
        [score, 1.0 - score]
    }

    fn new_buffer(&self) -> Self {
        Self::new()
    }

    /// Nothing to resample: the secret is in the move, not in the state.
    fn determinize_into<R: Rng + ?Sized>(
        &self,
        dest: &mut Self,
        _ctx: &(),
        _perspective: u8,
        _rng: &mut R,
    ) {
        *dest = self.clone();
    }
}

impl Observed for Bluff {
    type Key = Key;

    fn observed_key(&self, _ctx: &(), choice: &Move, actor: u8, observer: u8) -> Key {
        if actor == observer {
            return Key::Exact(*choice);
        }
        match choice {
            Move::Hide(_) => Key::Hidden,
            other => Key::Exact(*other),
        }
    }
}
