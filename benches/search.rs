//! Search throughput on synthetic games shaped like the two real consumers.
//!
//! Neither real game can be depended on from here, so each bench reproduces the
//! workload characteristics that actually drive cost:
//!
//! * `narrow` — small branching, imperfect information, a flat ~2 KB state copy
//!   per iteration, and a rollout that dominates the profile.
//! * `wide` — thousands of legal choices per node, `Choice` values that own heap
//!   data and are therefore expensive to clone, hash and compare, and long runs
//!   of consecutive moves by the same player.
//! * `tiny` — almost no game at all, so what is left is the search core. This is
//!   the only one where a few percent in `select` or backpropagation is visible
//!   above the noise floor.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, Searcher, Status};
use wyrand::WyRand;

fn seeded() -> WyRand {
    WyRand::seed_from_u64(0x5EED_1234)
}

fn config(iterations: u32) -> Config {
    Config {
        iterations,
        exploration_constant: 0.75,
        early_termination: false,
        ..Config::default()
    }
}

#[inline]
fn pick<R: Rng + ?Sized>(rng: &mut R, n: usize) -> usize {
    ((rng.next_u64() as u128 * n as u128) >> 64) as usize
}

/// Deterministic pseudo-random from the state, so branching varies by position
/// without consuming the search's RNG.
#[inline]
fn scramble(x: u64) -> u64 {
    let mut x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 32)
}

// ---------------------------------------------------------------- narrow ----

const STATE_BYTES: usize = 2048;
const NARROW_PLIES: u32 = 60;
const NARROW_PLAYERS: u8 = 3;

/// Small branching, three players, and a 2 KB state that copies as one flat
/// memcpy — the shape of a card game whose collections are all bitsets.
#[derive(Clone)]
struct Narrow {
    cells: [u8; STATE_BYTES],
    /// Information the searching player cannot see; reshuffled per iteration.
    hidden: [u8; 64],
    ply: u32,
    to_move: u8,
}

impl Narrow {
    fn new() -> Self {
        let mut cells = [0u8; STATE_BYTES];
        for (i, cell) in cells.iter_mut().enumerate() {
            *cell = scramble(i as u64) as u8;
        }
        Self {
            cells,
            hidden: [7; 64],
            ply: 0,
            to_move: 0,
        }
    }

    fn width(&self) -> usize {
        10 + (scramble(self.ply as u64 ^ self.cells[0] as u64) % 31) as usize
    }

    fn score(&self, player: u8) -> f64 {
        let stride = player as usize + 1;
        self.cells
            .iter()
            .step_by(64 * stride)
            .map(|&c| c as u32)
            .sum::<u32>() as f64
            / 4096.0
    }

    fn rewards(&self) -> [f64; NARROW_PLAYERS as usize] {
        let mut out = [0.0; NARROW_PLAYERS as usize];
        for (player, slot) in out.iter_mut().enumerate() {
            *slot = self.score(player as u8);
        }
        out
    }

    fn step(&mut self, choice: u16) {
        let at = (choice as usize * 37) % STATE_BYTES;
        self.cells[at] = self.cells[at].wrapping_add(choice as u8 | 1);
        self.cells[(at + 101) % STATE_BYTES] ^= choice as u8;
        self.ply += 1;
        self.to_move = (self.to_move + 1) % NARROW_PLAYERS;
    }
}

impl Game for Narrow {
    type Choice = u16;
    type Rewards = [f64; NARROW_PLAYERS as usize];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<Self::Rewards> {
        if self.ply >= NARROW_PLIES {
            Status::Terminal(self.rewards())
        } else {
            Status::Active {
                player: self.to_move,
            }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u16>) {
        // Roughly a fifth of choices are illegal in any given determinization,
        // which is what makes the availability bookkeeping do real work.
        let hidden = self.hidden[(self.ply as usize) % 64];
        out.extend(
            (0..self.width() as u16).filter(|choice| (choice.wrapping_add(hidden as u16)) % 5 != 0),
        );
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &u16, _: &mut R) {
        self.step(*choice);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> Self::Rewards {
        while self.ply < NARROW_PLIES {
            // Sample a few candidates and keep the one that scores best, so the
            // rollout costs what an expert policy costs rather than what a
            // uniform-random one does.
            let width = self.width();
            let mut best = 0u16;
            let mut best_score = i32::MIN;
            for _ in 0..4 {
                let candidate = pick(rng, width) as u16;
                let at = (candidate as usize * 37) % STATE_BYTES;
                let score = self.cells[at] as i32 - self.cells[(at + 101) % STATE_BYTES] as i32;
                if score > best_score {
                    best_score = score;
                    best = candidate;
                }
            }
            self.step(best);
        }
        self.rewards()
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        for i in (1..dest.hidden.len()).rev() {
            dest.hidden.swap(i, pick(rng, i + 1));
        }
    }
}

// ------------------------------------------------------------------ wide ----

const WIDE_PLIES: u32 = 24;
/// Consecutive moves by the same player, as an action-point economy produces.
const WIDE_CHAIN: u32 = 3;

/// A choice that owns heap data, so cloning it into a node, hashing it and
/// comparing it all chase a pointer.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Route(Vec<u16>);

#[derive(Clone)]
struct Wide {
    width: usize,
    ply: u32,
    to_move: u8,
    total: u64,
}

impl Game for Wide {
    type Choice = Route;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    // Heap-owning choices cross over sooner; see benchmarks/BASELINE.md.
    const CHILD_INDEX_THRESHOLD: usize = 8;

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.ply >= WIDE_PLIES {
            let share = (self.total % 1000) as f64 / 1000.0;
            Status::Terminal([share, 1.0 - share])
        } else {
            Status::Active {
                player: self.to_move,
            }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<Route>) {
        out.extend((0..self.width as u16).map(|i| {
            Route(vec![
                (i % 97).wrapping_add(3),
                (i / 97).wrapping_add(5),
                i ^ 0x5A5A,
                i,
            ])
        }));
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &Route, _: &mut R) {
        self.total = self.total.wrapping_add(choice.0[3] as u64);
        self.ply += 1;
        if self.ply.is_multiple_of(WIDE_CHAIN) {
            self.to_move ^= 1;
        }
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        while self.ply < WIDE_PLIES {
            let choice = pick(rng, self.width) as u64;
            self.total = self.total.wrapping_add(choice);
            self.ply += 1;
            if self.ply.is_multiple_of(WIDE_CHAIN) {
                self.to_move ^= 1;
            }
        }
        let share = (self.total % 1000) as f64 / 1000.0;
        [share, 1.0 - share]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

// ------------------------------------------------------------------ tiny ----

/// Branching two, eight plies, a `u64` of state and no rollout: whatever this
/// costs is the search core itself.
#[derive(Clone, Copy)]
struct Tiny {
    path: u64,
    ply: u32,
}

impl Game for Tiny {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.ply >= 8 {
            let value = (scramble(self.path) % 256) as f64 / 255.0;
            Status::Terminal([value, 1.0 - value])
        } else {
            Status::Active {
                player: (self.ply % 2) as u8,
            }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend([0u8, 1]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &u8, _: &mut R) {
        self.path = self.path << 1 | *choice as u64;
        self.ply += 1;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        while self.ply < 8 {
            let choice = (rng.next_u64() & 1) as u8;
            self.apply_choice(&(), &choice, rng);
        }
        match self.status(&()) {
            Status::Terminal(rewards) => rewards,
            Status::Active { .. } => unreachable!(),
        }
    }

    fn new_buffer(&self) -> Self {
        *self
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        *dest = *self;
    }
}

// ----------------------------------------------------------------- benches --

fn bench_narrow(c: &mut Criterion) {
    let mut group = c.benchmark_group("narrow");
    group.sample_size(20);
    for iterations in [1_000u32, 10_000] {
        group.throughput(Throughput::Elements(iterations as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(iterations),
            &iterations,
            |b, &iterations| {
                let game = Narrow::new();
                b.iter(|| {
                    let mut searcher = Searcher::new(&game);
                    searcher.search(&game, &(), 0, &config(iterations), None, &mut seeded())
                });
            },
        );
    }
    group.finish();
}

fn bench_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("wide");
    group.sample_size(10);
    for width in [100usize, 400, 1_600] {
        // The root alone holds `width` children, so the budget has to exceed it
        // by a good margin or the search never descends and the bench measures
        // root expansion instead of steady-state cost.
        let iterations = 3 * width as u32;
        group.throughput(Throughput::Elements(iterations as u64));
        group.bench_with_input(BenchmarkId::from_parameter(width), &width, |b, &width| {
            let game = Wide {
                width,
                ply: 0,
                to_move: 0,
                total: 1,
            };
            b.iter(|| {
                let mut searcher = Searcher::new(&game);
                searcher.search(&game, &(), 0, &config(iterations), None, &mut seeded())
            });
        });
    }
    group.finish();
}

fn bench_tiny(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiny");
    group.throughput(Throughput::Elements(100_000));
    group.bench_function("100k", |b| {
        let game = Tiny { path: 0, ply: 0 };
        b.iter(|| {
            let mut searcher = Searcher::new(&game);
            searcher.search(&game, &(), 0, &config(100_000), None, &mut seeded())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_tiny, bench_narrow, bench_wide);
criterion_main!(benches);
