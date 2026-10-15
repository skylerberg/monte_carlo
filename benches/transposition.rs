//! What merging transpositions costs and buys, against the tree.
//!
//! * `core` — the search core with the game removed, through the tree and
//!   through a graph that never merges. The gap is the price of the graph's
//!   storage and lookups on a game that gets nothing for it.
//! * `commute` — a game whose moves commute, so most of a tree is duplicated
//!   positions: the graph searches each position once.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, GraphSearcher, Searcher, Status, Transposition};
use wyrand::WyRand;

fn seeded() -> WyRand {
    WyRand::seed_from_u64(0x5EED_1234)
}

fn config(iterations: u32) -> Config {
    Config {
        iterations,
        early_termination: false,
        ..Config::default()
    }
}

#[inline]
fn pick<R: Rng + ?Sized>(rng: &mut R, n: usize) -> usize {
    ((rng.next_u64() as u128 * n as u128) >> 64) as usize
}

#[inline]
fn scramble(x: u64) -> u64 {
    let mut x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 32)
}

/// Four choices a ply, eight plies, a payoff hashed from the line: nothing for
/// the game to do, so the time is the search's. Keyed by nothing.
#[derive(Clone)]
struct Core {
    path: u64,
    ply: u32,
}

impl Game for Core {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.ply >= 8 {
            let p = (scramble(self.path) >> 11) as f64 / (1u64 << 53) as f64;
            Status::Terminal([p, 1.0 - p])
        } else {
            Status::Active {
                player: (self.ply % 2) as u8,
            }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend(0..4);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &u8, _: &mut R) {
        self.path = self.path.wrapping_mul(31).wrapping_add(*choice as u64 + 1);
        self.ply += 1;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &(), rng: &mut R) -> [f64; 2] {
        while self.ply < 8 {
            let choice = pick(rng, 4) as u8;
            self.apply_choice(ctx, &choice, rng);
        }
        match self.status(ctx) {
            Status::Terminal(rewards) => rewards,
            _ => unreachable!(),
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

impl Transposition for Core {
    type Key = ();

    fn transposition_key(&self, _: &()) -> Option<()> {
        None
    }
}

/// Pick 5 of 14 items one at a time, scored on the set; the order is
/// irrelevant, so the 240 240 orderings of a full line are 2 002 sets.
#[derive(Clone)]
struct Commute {
    picked: u16,
}

const ITEMS: u32 = 14;
const PICKS: u32 = 5;

impl Commute {
    fn score(&self) -> f64 {
        let value: u64 = (0..ITEMS)
            .filter(|i| self.picked & (1 << i) != 0)
            .map(|i| scramble(i as u64 + 7) % 100)
            .sum();
        value as f64 / (100 * PICKS) as f64
    }
}

impl Game for Commute {
    type Choice = u8;
    type Rewards = [f64; 1];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 1]> {
        if self.picked.count_ones() >= PICKS {
            Status::Terminal([self.score()])
        } else {
            Status::Active { player: 0 }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend((0..ITEMS as u8).filter(|&i| self.picked & (1 << i) == 0));
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &u8, _: &mut R) {
        self.picked |= 1 << choice;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 1] {
        while self.picked.count_ones() < PICKS {
            let open = ITEMS - self.picked.count_ones();
            let mut k = pick(rng, open as usize);
            for i in 0..ITEMS {
                if self.picked & (1 << i) == 0 {
                    if k == 0 {
                        self.picked |= 1 << i;
                        break;
                    }
                    k -= 1;
                }
            }
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

impl Transposition for Commute {
    type Key = u16;

    fn transposition_key(&self, _: &()) -> Option<u16> {
        Some(self.picked)
    }
}

fn bench_core(c: &mut Criterion) {
    let mut group = c.benchmark_group("core");
    group.sample_size(20);
    let iterations = 100_000;
    group.throughput(Throughput::Elements(u64::from(iterations)));
    let game = Core { path: 0, ply: 0 };
    group.bench_function("tree", |b| {
        b.iter(|| {
            let mut searcher = Searcher::new(&game);
            searcher.search(&game, &(), 0, &config(iterations), None, &mut seeded())
        })
    });
    group.bench_function("graph_unmerged", |b| {
        b.iter(|| {
            let mut searcher = GraphSearcher::new(&game);
            searcher.search(&game, &(), 0, &config(iterations), None, &mut seeded())
        })
    });
    group.finish();
}

fn bench_commute(c: &mut Criterion) {
    let mut group = c.benchmark_group("commute");
    group.sample_size(20);
    let game = Commute { picked: 0 };
    for iterations in [10_000u32, 50_000] {
        group.throughput(Throughput::Elements(u64::from(iterations)));
        group.bench_with_input(
            BenchmarkId::new("tree", iterations),
            &iterations,
            |b, &n| {
                b.iter(|| {
                    let mut searcher = Searcher::new(&game);
                    searcher.search(&game, &(), 0, &config(n), None, &mut seeded())
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("graph", iterations),
            &iterations,
            |b, &n| {
                b.iter(|| {
                    let mut searcher = GraphSearcher::new(&game);
                    searcher.search(&game, &(), 0, &config(n), None, &mut seeded())
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_core, bench_commute);
criterion_main!(benches);
