//! Where the linear child scan stops paying.
//!
//! `expand` has to find, for every legal choice, whether this node already has a
//! child for it. Below some child count a linear `PartialEq` scan beats hashing;
//! above it the scan is quadratic in the branching factor. The crossover depends
//! on how expensive `Choice` equality and hashing are, so it is measured rather
//! than assumed, for a cheap `Copy` choice and for one that owns a `Vec`.
//!
//! The result sets the default for `Game::CHILD_INDEX_THRESHOLD`.

use std::hash::Hash;
use std::marker::PhantomData;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, Searcher, Status};
use wyrand::WyRand;

const ITERATIONS: u32 = 64;

trait Strategy: 'static {
    const THRESHOLD: usize;
    const NAME: &'static str;
}

struct Scan;
impl Strategy for Scan {
    const THRESHOLD: usize = usize::MAX;
    const NAME: &'static str = "scan";
}

struct Indexed;
impl Strategy for Indexed {
    const THRESHOLD: usize = 0;
    const NAME: &'static str = "index";
}

trait ChoiceKind: Clone + Eq + Hash + 'static {
    const NAME: &'static str;
    fn make(i: usize) -> Self;
}

impl ChoiceKind for u64 {
    const NAME: &'static str = "u64";
    fn make(i: usize) -> Self {
        i as u64
    }
}

/// Stands in for a choice that owns heap payloads, as asteroids' does: equality
/// and hashing both chase a pointer, and the discriminating field is last so
/// comparison cannot short-circuit on the first element.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Heavy(Vec<u32>);

impl ChoiceKind for Heavy {
    const NAME: &'static str = "vec";
    fn make(i: usize) -> Self {
        Heavy(vec![7, 9, 11, i as u32])
    }
}

struct WideBandit<S, C> {
    ply: u32,
    width: usize,
    marker: PhantomData<fn() -> (S, C)>,
}

// Hand-written: deriving would demand `S: Clone` and `C: Clone`, which the
// marker does not actually need.
impl<S, C> Clone for WideBandit<S, C> {
    fn clone(&self) -> Self {
        Self {
            ply: self.ply,
            width: self.width,
            marker: PhantomData,
        }
    }
}

impl<S: Strategy, C: ChoiceKind> Game for WideBandit<S, C> {
    type Choice = C;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    const CHILD_INDEX_THRESHOLD: usize = S::THRESHOLD;
    // Re-enumerate the root every iteration. That is where lookup cost shows up,
    // and it is what any node below the root pays anyway.
    const ROOT_CHOICES_INVARIANT: bool = false;

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.ply >= 2 {
            Status::Terminal([0.5, 0.5])
        } else {
            Status::Active { player: 0 }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<C>) {
        out.extend((0..self.width).map(C::make));
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &C, _: &mut R) {
        self.ply += 1;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        [0.5, 0.5]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

fn search<S: Strategy, C: ChoiceKind>(width: usize) {
    let game: WideBandit<S, C> = WideBandit {
        ply: 0,
        width,
        marker: PhantomData,
    };
    let mut searcher = Searcher::new(&game);
    let config = Config {
        iterations: ITERATIONS,
        early_termination: false,
        ..Config::default()
    };
    searcher.search(
        &game,
        &(),
        0,
        &config,
        None,
        &mut WyRand::seed_from_u64(0x5eed),
    );
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("child_lookup");
    group.sample_size(10);

    for width in [4usize, 8, 16, 32, 64, 128, 256, 512, 1024] {
        group.bench_with_input(
            BenchmarkId::new(format!("{}/{}", Scan::NAME, u64::NAME), width),
            &width,
            |b, &w| b.iter(|| search::<Scan, u64>(w)),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("{}/{}", Indexed::NAME, u64::NAME), width),
            &width,
            |b, &w| b.iter(|| search::<Indexed, u64>(w)),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("{}/{}", Scan::NAME, Heavy::NAME), width),
            &width,
            |b, &w| b.iter(|| search::<Scan, Heavy>(w)),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("{}/{}", Indexed::NAME, Heavy::NAME), width),
            &width,
            |b, &w| b.iter(|| search::<Indexed, Heavy>(w)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
