//! Fixtures for [`mcts::GraphSearcher`].
//!
//! Two wrappers that never merge anything, which is how a graph search is held
//! to the tree search bit for bit, and a handful of games that do transpose, or
//! repeat, or leak hidden state into their key.

#![allow(dead_code)]

use core::hash::{Hash, Hasher};

use mcts::rand_core::Rng;
use mcts::{ChoicesOnRevisit, Game, JointChoices, SimultaneousPolicy, Status, Transposition};

use super::{AlwaysWin, CountToThree};

/// Forwards every `Game` hook, associated consts included, to `$inner`.
macro_rules! delegate_game {
    ($inner:tt) => {
        type Choice = G::Choice;
        type Rewards = G::Rewards;
        type Context = G::Context;
        type Side = G::Side;

        const CHOICES_ON_REVISIT: ChoicesOnRevisit = G::CHOICES_ON_REVISIT;
        const CHILD_INDEX_THRESHOLD: usize = G::CHILD_INDEX_THRESHOLD;
        const SIMULTANEOUS_POLICY: SimultaneousPolicy = G::SIMULTANEOUS_POLICY;

        fn status(&self, ctx: &G::Context) -> Status<G::Rewards> {
            self.$inner.status(ctx)
        }

        fn choices_into(&self, ctx: &G::Context, out: &mut Vec<G::Choice>) {
            self.$inner.choices_into(ctx, out)
        }

        fn choices_for_into(&self, ctx: &G::Context, player: u8, out: &mut Vec<G::Choice>) {
            self.$inner.choices_for_into(ctx, player, out)
        }

        fn rollout<R: Rng + ?Sized>(&mut self, ctx: &G::Context, rng: &mut R) -> G::Rewards {
            self.$inner.rollout(ctx, rng)
        }

        fn advance<R: Rng + ?Sized>(
            &mut self,
            ctx: &G::Context,
            side: &mut G::Side,
            perspective: u8,
            rng: &mut R,
        ) {
            self.$inner.advance(ctx, side, perspective, rng)
        }

        fn init_side(&self, ctx: &G::Context, side: &mut G::Side) {
            self.$inner.init_side(ctx, side)
        }

        fn begin_iteration(side: &mut G::Side) {
            G::begin_iteration(side)
        }

        fn credit_iteration(side: &mut G::Side, rewards: &G::Rewards) {
            G::credit_iteration(side, rewards)
        }

        fn heuristic_bias(&self, ctx: &G::Context, mover: u8) -> f32 {
            self.$inner.heuristic_bias(ctx, mover)
        }
    };
}

/// A game that never merges a position, so a [`mcts::GraphSearcher`] over it
/// must search exactly as a [`mcts::Searcher`] over the game itself.
#[derive(Clone)]
pub struct Untransposed<G>(pub G);

impl<G: Game> Game for Untransposed<G> {
    delegate_game!(0);

    fn apply_choice<R: Rng + ?Sized>(&mut self, ctx: &G::Context, choice: &G::Choice, rng: &mut R) {
        self.0.apply_choice(ctx, choice, rng)
    }

    fn apply_joint<R: Rng + ?Sized>(
        &mut self,
        ctx: &G::Context,
        joint: JointChoices<'_, G::Choice>,
        rng: &mut R,
    ) {
        self.0.apply_joint(ctx, joint, rng)
    }

    fn new_buffer(&self) -> Self {
        Untransposed(self.0.new_buffer())
    }

    fn determinize_into<R: Rng + ?Sized>(
        &self,
        dest: &mut Self,
        ctx: &G::Context,
        perspective: u8,
        rng: &mut R,
    ) {
        self.0.determinize_into(&mut dest.0, ctx, perspective, rng)
    }
}

impl<G: Game> Transposition for Untransposed<G> {
    type Key = ();

    fn transposition_key(&self, _: &G::Context) -> Option<()> {
        None
    }
}

/// A game whose key is the whole move history, so every position is keyed and
/// every lookup runs, but no two routes ever share a key — a graph search over
/// it exercises the merge table and must still search exactly as the tree.
#[derive(Clone)]
pub struct PathKeyed<G> {
    pub game: G,
    pub path: u64,
}

impl<G> PathKeyed<G> {
    pub fn new(game: G) -> Self {
        Self { game, path: 0 }
    }

    /// Order-sensitive, so `[a, b]` and `[b, a]` key differently.
    fn extend<T: Hash>(&mut self, step: &T) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.path.hash(&mut hasher);
        step.hash(&mut hasher);
        self.path = hasher.finish();
    }
}

impl<G: Game> Game for PathKeyed<G> {
    delegate_game!(game);

    fn apply_choice<R: Rng + ?Sized>(&mut self, ctx: &G::Context, choice: &G::Choice, rng: &mut R) {
        self.extend(choice);
        self.game.apply_choice(ctx, choice, rng)
    }

    fn apply_joint<R: Rng + ?Sized>(
        &mut self,
        ctx: &G::Context,
        joint: JointChoices<'_, G::Choice>,
        rng: &mut R,
    ) {
        for (player, choice) in joint.iter() {
            self.extend(&(player, choice));
        }
        self.game.apply_joint(ctx, joint, rng)
    }

    fn new_buffer(&self) -> Self {
        Self {
            game: self.game.new_buffer(),
            path: self.path,
        }
    }

    fn determinize_into<R: Rng + ?Sized>(
        &self,
        dest: &mut Self,
        ctx: &G::Context,
        perspective: u8,
        rng: &mut R,
    ) {
        dest.path = self.path;
        self.game
            .determinize_into(&mut dest.game, ctx, perspective, rng)
    }
}

impl<G: Game> Transposition for PathKeyed<G> {
    type Key = u64;

    fn transposition_key(&self, _: &G::Context) -> Option<u64> {
        Some(self.path)
    }
}

/// Every choice leads to the same next ply, so the whole game is four
/// positions — one per ply — however it is played.
impl Transposition for AlwaysWin {
    type Key = u32;

    fn transposition_key(&self, _: &()) -> Option<u32> {
        Some(self.ply)
    }
}

/// Only the number of ones so far matters, not their order: three plies of
/// binary choices are fifteen tree nodes and ten positions.
impl Transposition for CountToThree {
    type Key = (u32, u32);

    fn transposition_key(&self, _: &()) -> Option<(u32, u32)> {
        Some((self.ply, self.points))
    }
}

/// Pick `k` of `n` items one at a time; the score is what the picked items are
/// worth together, so the order they were picked in does not matter at all. A
/// tree searches every order of every set separately; a graph searches each set
/// once.
#[derive(Clone)]
pub struct Collect {
    pub values: Vec<f64>,
    pub k: u32,
    pub picked: u64,
}

impl Collect {
    /// Item `i` is worth `i`, so the best set is the top `k` items.
    pub fn new(n: usize, k: u32) -> Self {
        Self {
            values: (0..n).map(|i| i as f64).collect(),
            k,
            picked: 0,
        }
    }

    fn score(&self) -> f64 {
        let best: f64 = self.values.iter().rev().take(self.k as usize).sum();
        let got: f64 = (0..self.values.len())
            .filter(|&i| self.picked & (1 << i) != 0)
            .map(|i| self.values[i])
            .sum();
        got / best
    }
}

impl Game for Collect {
    type Choice = usize;
    type Rewards = [f64; 1];
    type Context = ();
    type Side = ();

    const CHOICES_ON_REVISIT: ChoicesOnRevisit = ChoicesOnRevisit::SameEverywhere;

    fn status(&self, _: &()) -> Status<[f64; 1]> {
        if self.picked.count_ones() >= self.k {
            Status::Terminal([self.score()])
        } else {
            Status::Active { player: 0 }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend((0..self.values.len()).filter(|&i| self.picked & (1 << i) == 0));
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, _: &mut R) {
        self.picked |= 1 << choice;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &(), rng: &mut R) -> [f64; 1] {
        let mut open = Vec::new();
        while self.picked.count_ones() < self.k {
            open.clear();
            self.choices_into(ctx, &mut open);
            let i = ((rng.next_u64() as u128 * open.len() as u128) >> 64) as usize;
            self.picked |= 1 << open[i];
        }
        [self.score()]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

impl Transposition for Collect {
    type Key = u64;

    fn transposition_key(&self, _: &()) -> Option<u64> {
        Some(self.picked)
    }
}

/// A token on a ring of `size` squares, moved one or two squares forward by
/// two players in turn; whoever lands on square 0 wins.
///
/// Play can circle the ring for ever, so the key — the square and the player to
/// move — repeats along a line of play, which is the cycle a graph search has
/// to stop at rather than loop on. Rollouts are capped and score a draw.
#[derive(Clone)]
pub struct Ring {
    pub size: u8,
    pub square: u8,
    pub to_move: u8,
    pub winner: Option<u8>,
}

impl Ring {
    pub fn new(size: u8, square: u8) -> Self {
        Self {
            size,
            square,
            to_move: 0,
            winner: None,
        }
    }
}

impl Game for Ring {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    const CHOICES_ON_REVISIT: ChoicesOnRevisit = ChoicesOnRevisit::SameEverywhere;

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.winner {
            Some(winner) => {
                let mut rewards = [0.0; 2];
                rewards[winner as usize] = 1.0;
                Status::Terminal(rewards)
            }
            None => Status::Active {
                player: self.to_move,
            },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend([1, 2]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), step: &u8, _: &mut R) {
        self.square = (self.square + step) % self.size;
        if self.square == 0 {
            self.winner = Some(self.to_move);
        }
        self.to_move ^= 1;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &(), rng: &mut R) -> [f64; 2] {
        for _ in 0..64 {
            if let Status::Terminal(rewards) = self.status(ctx) {
                return rewards;
            }
            let step = 1 + (rng.next_u64() & 1) as u8;
            self.apply_choice(ctx, &step, rng);
        }
        match self.status(ctx) {
            Status::Terminal(rewards) => rewards,
            _ => [0.5, 0.5],
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

impl Transposition for Ring {
    type Key = (u8, u8, Option<u8>);

    fn transposition_key(&self, _: &()) -> Option<(u8, u8, Option<u8>)> {
        Some((self.square, self.to_move, self.winner))
    }
}

/// Player 1 holds a hidden card that every determinization redraws, and this
/// key reads it — exactly the mistake the key contract forbids.
#[derive(Clone)]
pub struct LeakyKey {
    pub hidden: u8,
    pub moves: u8,
}

impl Game for LeakyKey {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.moves >= 3 {
            Status::Terminal([0.5, 0.5])
        } else {
            Status::Active {
                player: self.moves % 2,
            }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend([0, 1]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        self.moves += 1;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        self.moves = 3;
        [0.5, 0.5]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.hidden = (rng.next_u64() % 4) as u8;
    }
}

impl Transposition for LeakyKey {
    type Key = (u8, u8);

    fn transposition_key(&self, _: &()) -> Option<(u8, u8)> {
        Some((self.moves, self.hidden))
    }
}
