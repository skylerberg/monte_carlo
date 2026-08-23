//! What a simultaneous node charges per iteration, and what shape charges most.
//!
//! A simultaneous level is not a sequential one with more children. Every
//! iteration runs one bandit per participant over that participant's own arms,
//! packs the chosen tuple into a joint key, finds or creates the joint
//! successor, and carries one importance weight per participant back up. None
//! of that exists on the sequential path, so none of it is covered by
//! `benches/search.rs` and all of it is measured here:
//!
//! * `rps` — three arms, two players, one ply and no game underneath, run under
//!   both selection policies. `Duct` is the cheap rule that does not converge
//!   and `RegretMatching` is the default that does; what the default costs is a
//!   number in this table rather than an argument.
//! * `simul_wide` — arms per player at arity two. Arm lookup switches from a
//!   linear scan to a hash index at `Game::CHILD_INDEX_THRESHOLD`, so 8 scans
//!   and 32 and 128 index, and the joint successor count grows as the square.
//! * `simul_arity` — eight arms per player, two to four participants. Arity is
//!   the multiplier on selection, on key packing and on backup's importance
//!   weights, and the exponent on the joint space. Arms are pooled across
//!   slots, so this group crosses `CHILD_INDEX_THRESHOLD` on the way up:
//!   arity 2 holds 16 arms and scans, arity 3 and 4 hold 24 and 32 and index.
//!   Part of the step from 2 to 3 is that switch, not arity itself.
//!
//! `simul_wide/8` and `simul_arity/2` are deliberately the same configuration —
//! eight arms, two players, one budget — measured from two different groups.
//! They are the session's own noise check: if they disagree by more than a
//! couple of percent, nothing else in the table is worth 3% comparisons either.
//!
//! Run as:
//!
//! ```text
//! cargo bench --bench simultaneous -- --warm-up-time 0.5 --measurement-time 2.0
//! ```
//!
//! `benchmarks/BASELINE.md`'s protocol is what makes these numbers comparable,
//! and two of its rules are the caller's rather than this file's: run the whole
//! command twice and keep the second run, then re-run the first configuration
//! alone and void the session if it has drifted more than 3%. What the file
//! does provide is the interleaving — the two `rps` policies sit adjacent at
//! each budget, so a machine that warms up or throttles mid-session moves both
//! points of the comparison together.
//!
//! Everything runs at `Config::default()`'s selection constants — including
//! `regret_matching_exploration`, whose floor decides how often the rebuilt
//! distribution is a near-uniform one. A bench that pins its own value there
//! would be timing a configuration nobody runs.

use std::marker::PhantomData;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, JointChoices, PlayerSet, Searcher, SimultaneousPolicy, Status};
use wyrand::WyRand;

/// Widest arity benched, and therefore the reward array's length.
const MAX_ARITY: usize = 4;

/// Budget for the shape groups. Fixed rather than scaled with width or arity so
/// that the per-iteration numbers are directly comparable down each column;
/// what varies between the points is the node, not how much of it was built.
const SHAPE_ITERATIONS: u32 = 20_000;

fn seeded() -> WyRand {
    WyRand::seed_from_u64(0x5EED_1234)
}

/// `Config::default()` but for the budget and `early_termination`, which has to
/// be off: a search that stops early has not run the iterations the throughput
/// number divides by, and `Duct` — which can prove a root child — would stop at
/// a different point from `RegretMatching`, which never can.
fn config(iterations: u32) -> Config {
    Config {
        iterations,
        early_termination: false,
        ..Config::default()
    }
}

// ---------------------------------------------------------------- policy ----

trait Policy: 'static {
    const POLICY: SimultaneousPolicy;
    const NAME: &'static str;
}

struct Matching;
impl Policy for Matching {
    const POLICY: SimultaneousPolicy = SimultaneousPolicy::RegretMatching;
    const NAME: &'static str = "regret_matching";
}

struct Decoupled;
impl Policy for Decoupled {
    const POLICY: SimultaneousPolicy = SimultaneousPolicy::Duct;
    const NAME: &'static str = "duct";
}

// ------------------------------------------------------------------ game ----

/// One pairwise round of cyclic rock-paper-scissors: `a` beats the actions just
/// above it and loses to the ones just below, wrapping, with the diametrically
/// opposite action drawn at even widths.
///
/// Skew-symmetric — `payoff(a, b) + payoff(b, a) == 1` for every pair — and its
/// two-player equilibrium is uniform and strictly mixed at every width, which is
/// the property the bench needs: a game with a pure equilibrium lets regret
/// matching collapse onto one arm, after which its distribution rebuild is
/// measuring a degenerate case rather than the one the policy exists for. What
/// makes that hold above two players is how [`Cyclic`] combines these rounds.
fn payoff(a: usize, b: usize, width: usize) -> f64 {
    let d = (a + width - b) % width;
    if d == 0 || 2 * d == width {
        0.5
    } else if 2 * d < width {
        1.0
    } else {
        0.0
    }
}

/// One simultaneous ply and nothing else: no rollout, no hidden state, and a
/// `u8` choice, so what the bench costs is the decoupled layer itself.
///
/// Participants are arranged in a cycle and every player's payoff is what they
/// take from their successor **less what their predecessor takes from them**.
/// Those halves telescope around the cycle, so the arity payoffs sum to
/// `arity / 2` whatever is played and no profile pays everybody at once — which
/// is what keeps the problem genuinely mixed above two players. At arity 2 the
/// second term is the first one's mirror and the whole expression collapses to
/// [`payoff`] itself, so the two-player groups measure exactly the game they
/// always did.
///
/// Scoring each player against their successor alone, which this bench did
/// first, does not survive three participants: at width 8 that game has 24 pure
/// equilibria at arity 3 and 152 at arity 4, every one of them paying all three
/// or four players 1.0 at once. Regret matching then concentrates — measured
/// strategy entropy 0.64 to 1.27 nats against the 2.08 of a uniform mixture over
/// eight arms — and the joint-child population collapses with it, to 279 and 501
/// children against the 487 and 2 722 the constant-sum form builds. That is 1.7x
/// and 5.4x less tree, and the size of that tree is the quantity `simul_arity`
/// exists to compare against `simul_arity/2`.
struct Cyclic<P> {
    width: usize,
    arity: u8,
    resolved: bool,
    rewards: [f64; MAX_ARITY],
    marker: PhantomData<fn() -> P>,
}

impl<P> Cyclic<P> {
    fn new(width: usize, arity: u8) -> Self {
        Self {
            width,
            arity,
            resolved: false,
            rewards: [0.0; MAX_ARITY],
            marker: PhantomData,
        }
    }
}

// Hand-written: deriving would demand `P: Clone`, which the marker does not
// actually need.
impl<P> Clone for Cyclic<P> {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            arity: self.arity,
            resolved: self.resolved,
            rewards: self.rewards,
            marker: PhantomData,
        }
    }
}

impl<P: Policy> Game for Cyclic<P> {
    type Choice = u8;
    type Rewards = [f64; MAX_ARITY];
    type Context = ();
    type Side = ();

    const SIMULTANEOUS_POLICY: SimultaneousPolicy = P::POLICY;
    // Re-enumerate the root's marginals every iteration. That is what every
    // simultaneous node below a root pays anyway, and at 128 arms it is where
    // the arm lookup this group exists to measure actually happens.
    const ROOT_CHOICES_INVARIANT: bool = false;

    fn status(&self, _: &()) -> Status<[f64; MAX_ARITY]> {
        if self.resolved {
            Status::Terminal(self.rewards)
        } else {
            Status::Simultaneous {
                players: PlayerSet::first_n(self.arity),
            }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend(0..self.width as u8);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("mcts bench: Cyclic has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        let arity = joint.len();
        for slot in 0..arity {
            let mine = joint[slot] as usize;
            let next = joint[(slot + 1) % arity] as usize;
            let previous = joint[(slot + arity - 1) % arity] as usize;
            self.rewards[joint.player_at(slot) as usize] =
                0.5 + (payoff(mine, next, self.width) - payoff(previous, mine, self.width)) / 2.0;
        }
        self.resolved = true;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; MAX_ARITY] {
        // Only ever reached on a freshly created joint child, whose round
        // `apply_joint` has already resolved.
        self.rewards
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// One whole search, tree included: a fresh `Searcher` per repetition, so the
/// build and the teardown of the joint tree are inside the measurement the way
/// they are inside a real move decision.
///
/// Returns the root's visit count purely so the call has a result to hand
/// `black_box`.
fn search<P: Policy>(width: usize, arity: u8, iterations: u32) -> u32 {
    let game: Cyclic<P> = Cyclic::new(width, arity);
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &config(iterations), None, &mut seeded());
    result.root_visits
}

// --------------------------------------------------------------- benches ----

/// The simultaneous core with the game removed, under both policies. The two
/// points sit next to each other at each budget so the A/B is interleaved
/// rather than batched, as `benchmarks/BASELINE.md` requires.
fn bench_rps(c: &mut Criterion) {
    let mut group = c.benchmark_group("rps");
    group.sample_size(20);
    for (label, iterations) in [("10k", 10_000u32), ("100k", 100_000)] {
        group.throughput(Throughput::Elements(iterations as u64));
        group.bench_with_input(
            BenchmarkId::new(Decoupled::NAME, label),
            &iterations,
            |b, &n| b.iter(|| search::<Decoupled>(3, 2, n)),
        );
        group.bench_with_input(
            BenchmarkId::new(Matching::NAME, label),
            &iterations,
            |b, &n| b.iter(|| search::<Matching>(3, 2, n)),
        );
    }
    group.finish();
}

/// What arms per player cost. Selection is linear in arms, the arm lookup
/// crosses from a scan to a hash index between 8 and 32, and the joint space
/// grows as the square — so the per-iteration cost should grow faster than the
/// width but far slower than the joint child count.
fn bench_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("simul_wide");
    group.sample_size(20);
    group.throughput(Throughput::Elements(SHAPE_ITERATIONS as u64));
    for width in [8usize, 32, 128] {
        group.bench_with_input(BenchmarkId::from_parameter(width), &width, |b, &w| {
            b.iter(|| search::<Matching>(w, 2, SHAPE_ITERATIONS))
        });
    }
    group.finish();
}

/// What a participant costs. Every added player is another bandit to run,
/// another slot in the joint key, and another importance weight in backup, on
/// top of multiplying the joint space by the width.
fn bench_arity(c: &mut Criterion) {
    let mut group = c.benchmark_group("simul_arity");
    group.sample_size(20);
    group.throughput(Throughput::Elements(SHAPE_ITERATIONS as u64));
    for arity in [2u8, 3, 4] {
        group.bench_with_input(BenchmarkId::from_parameter(arity), &arity, |b, &k| {
            b.iter(|| search::<Matching>(8, k, SHAPE_ITERATIONS))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rps, bench_wide, bench_arity);
criterion_main!(benches);
