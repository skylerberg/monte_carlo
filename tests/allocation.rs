//! The crate's allocation claim, as a test rather than a sentence.
//!
//! `lib.rs` promises that "a steady-state iteration allocates nothing but tree
//! growth", and the README sells the crate on it. Nothing checked it. This does,
//! by counting: once a tree has saturated, running five times the iterations
//! must allocate exactly what running one times the iterations allocated.
//!
//! Counts rather than wall clock, deliberately. An allocation count is a linear,
//! reproducible quantity that does not care what else the machine is doing, so
//! this gate is recordable on a contended machine where a timing gate is not.
//! It does not prove the search got no slower; it proves the search does not do
//! more of the one thing the design forbids it to do per iteration.
//!
//! One `#[test]` on purpose: the counter is global, so a second test running
//! concurrently on another thread would be counted into this one's region.
//!
//! Every fixture below is allocation-free *itself* — its state is a couple of
//! integers and its `determinize_into` copies them — so that every allocation
//! the counter sees belongs to the library. That rules out the shared
//! `common::GameTree`, whose state is a heap tree and whose determinization
//! deep-clones it: measured, it allocates 25 times per iteration at width 24,
//! all of it the fixture's own `clone_from`. A fixture that allocates cannot
//! witness a library that does not.

#[allow(dead_code)]
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{AlwaysWin, Rps, VariableRps};
use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, Searcher, Status};
use wyrand::WyRand;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// Only the count is instrumented; every request is forwarded to the system
// allocator unchanged, so the search runs against the same allocator it always
// does.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// How many choices each player has at [`WideLine`], and how many joint
/// successors [`WideLine`]'s two plies therefore have.
///
/// Above `WideLine::CHILD_INDEX_THRESHOLD`, so both plies leave the linear
/// child scan for the hash index — which is the allocating path, and so the one
/// worth pinning: the index is built once per node and must not be rebuilt per
/// visit.
const WIDE: usize = 24;

/// Two sequential plies of `WIDE` choices each: player 0 picks, then player 1
/// picks knowing it, and player 0 scores 1.0 only if the two picks match.
///
/// A wide analogue of matching pennies, and wide on purpose. Every root child is
/// worth the same to player 0 under best play, so nothing starves and the whole
/// `1 + WIDE + WIDE * WIDE` node tree is expanded early — which is what makes
/// "saturated" true rather than hoped for.
///
/// Its state is two `u8`s. That is the load-bearing property here, not the
/// payoffs: `determinize_into` copies them and allocates nothing, so the counter
/// sees the library alone.
#[derive(Clone)]
struct WideLine {
    ply: u8,
    first: u8,
}

impl Game for WideLine {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    const CHILD_INDEX_THRESHOLD: usize = 8;

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.ply {
            2 => Status::Terminal([0.0, 1.0]),
            3 => Status::Terminal([1.0, 0.0]),
            ply => Status::Active { player: ply },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend(0..WIDE as u8);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &u8, _: &mut R) {
        match self.ply {
            0 => {
                self.first = *choice;
                self.ply = 1;
            }
            // Terminal state 3 is "the picks matched", 2 is "they did not".
            _ => self.ply = if *choice == self.first { 3 } else { 2 },
        }
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        while self.ply < 2 {
            let choice = ((rng.next_u64() as u128 * WIDE as u128) >> 64) as u8;
            self.apply_choice(&(), &choice, rng);
        }
        match self.ply {
            3 => [1.0, 0.0],
            _ => [0.0, 1.0],
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

fn config(iterations: u32) -> Config {
    Config {
        iterations,
        early_termination: false,
        ..Config::default()
    }
}

/// Allocations performed by one whole search, searcher construction included.
///
/// Construction is inside the region on purpose: a `Searcher` built for a bigger
/// budget must not size any buffer by the budget, and counting the constructor
/// is what would catch it.
fn allocations_for<G: Game<Context = ()>>(
    game: &G,
    perspective: u8,
    iterations: u32,
) -> (usize, usize) {
    let mut searcher = Searcher::new(game);
    let mut rng = WyRand::seed_from_u64(7);
    let cfg = config(iterations);

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    searcher.search(game, &(), perspective, &cfg, None, &mut rng);
    let after = ALLOCATIONS.load(Ordering::Relaxed);

    let nodes = searcher
        .tree()
        .expect("a search leaves a tree")
        .node_count();
    (after - before, nodes)
}

/// `name` must saturate its tree well inside `small` iterations, so every
/// allocation the search will ever make has already happened by then.
///
/// `nodes` is that saturation stated as a number: the size of the fixture's
/// whole tree, counted by hand from its rules. Without it the equality below is
/// satisfiable by a search that stalled — one that stopped growing at ten of the
/// `WIDE` root children would allocate the same at both budgets and prove
/// nothing. Asserting the tree really is the whole game is what makes "saturated"
/// a fact rather than an assumption, and it is checked at *both* budgets, so a
/// small budget that happened to fall short would fail rather than lower the bar.
///
/// The two budgets travel as a pair so that the four numbers a call site passes
/// cannot be read in the wrong order.
fn assert_saturates<G: Game<Context = ()>>(
    name: &str,
    game: &G,
    perspective: u8,
    nodes: usize,
    (small, large): (u32, u32),
) {
    let (few, few_nodes) = allocations_for(game, perspective, small);
    let (many, many_nodes) = allocations_for(game, perspective, large);
    assert_eq!(
        (few_nodes, many_nodes),
        (nodes, nodes),
        "{name}: the tree holds {few_nodes} nodes at {small} iterations and \
         {many_nodes} at {large}, not the {nodes} the whole game has. The budgets \
         below do not saturate it, so an equal allocation count would be evidence \
         about a stalled search rather than about a steady-state one."
    );
    assert!(
        few > 0,
        "{name}: the search allocated nothing even once, so this measured no search at all"
    );
    assert_eq!(
        few,
        many,
        "{name}: {small} iterations allocated {few} times and {large} allocated {many}. \
         A saturated tree has nothing left to grow, so the extra {} iterations should \
         have allocated nothing at all — the search is allocating per iteration.",
        large - small
    );
}

#[test]
fn a_saturated_search_stops_allocating() {
    // Three choices, three plies, every leaf terminal: 1 + 3 + 9 + 27 nodes,
    // saturated within a few hundred iterations. Measured, 16 allocations at
    // both budgets.
    assert_saturates("always_win", &AlwaysWin { ply: 0 }, 1, 40, (2_000, 10_000));

    // Two plies of `WIDE`, so 1 + 24 + 576 nodes and the hash child index built
    // at every internal one. Measured, 177 allocations at both budgets.
    assert_saturates(
        "wide_line",
        &WideLine { ply: 0, first: 0 },
        0,
        1 + WIDE + WIDE * WIDE,
        (4_000, 20_000),
    );

    // Simultaneous: three arms per player and nine joint successors, so the
    // marginal arrays and every joint child exist long before the small budget
    // runs out. The node count is 1 + 9 rather than 1 + 9 + 6 arms, which is the
    // design's claim that a joint successor is a node and an arm is not.
    // Measured, 20 allocations at both budgets.
    assert_saturates("rps", &Rps::default(), 0, 10, (2_000, 10_000));

    // Simultaneous with legality that varies per determinization, which is the
    // one path that regenerates a player's action list every iteration and
    // refreshes the root legality mask against it. If any of that allocated per
    // iteration it would show here and nowhere else. Measured, 23 allocations at
    // both budgets — three more than `rps` and not one per iteration.
    assert_saturates(
        "variable_rps",
        &VariableRps::default(),
        0,
        10,
        (2_000, 10_000),
    );
}
