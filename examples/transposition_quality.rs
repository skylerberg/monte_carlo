//! Does merging transpositions buy better play at the same budget?
//!
//! Pick 5 of 16 items, one at a time, and score the set by the synergy of every
//! pair in it: a random symmetric matrix, so the best set is not the greedy one
//! and a search has to weigh combinations. The order of the picks never matters,
//! so a tree holds every set once per order it was picked in and a graph holds
//! it once. Each game is played to the end by `Searcher` and by `GraphSearcher`
//! at equal iteration budgets; the optimum is brute-forced over all 4 368 sets.
//!
//! `cargo run --release --example transposition_quality`

use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, GraphSearcher, Searcher, Status, Transposition};
use wyrand::WyRand;

const ITEMS: usize = 16;
const PICKS: u32 = 5;

#[derive(Clone)]
struct Synergy {
    pairs: [[f64; ITEMS]; ITEMS],
    picked: u16,
}

impl Synergy {
    fn new(seed: u64) -> Self {
        let mut rng = WyRand::seed_from_u64(seed);
        let mut pairs = [[0.0; ITEMS]; ITEMS];
        for (i, j) in (0..ITEMS).flat_map(|i| (i + 1..ITEMS).map(move |j| (i, j))) {
            let value = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            pairs[i][j] = value;
            pairs[j][i] = value;
        }
        Self { pairs, picked: 0 }
    }

    fn total(&self, picked: u16) -> f64 {
        let mut total = 0.0;
        for i in 0..ITEMS {
            for j in i + 1..ITEMS {
                if picked & (1 << i) != 0 && picked & (1 << j) != 0 {
                    total += self.pairs[i][j];
                }
            }
        }
        total
    }

    fn best(&self) -> f64 {
        (0u32..1 << ITEMS)
            .filter(|set| set.count_ones() == PICKS)
            .map(|set| self.total(set as u16))
            .fold(0.0, f64::max)
    }
}

impl Game for Synergy {
    type Choice = u8;
    type Rewards = [f64; 1];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 1]> {
        if self.picked.count_ones() >= PICKS {
            // Ten pairs of values in [0, 1]: the scale is [0, 10].
            Status::Terminal([self.total(self.picked) / 10.0])
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

    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &(), rng: &mut R) -> [f64; 1] {
        let mut open = Vec::with_capacity(ITEMS);
        while self.picked.count_ones() < PICKS {
            open.clear();
            self.choices_into(ctx, &mut open);
            let k = ((rng.next_u64() as u128 * open.len() as u128) >> 64) as usize;
            self.picked |= 1 << open[k];
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

impl Transposition for Synergy {
    type Key = u16;

    fn transposition_key(&self, _: &()) -> Option<u16> {
        Some(self.picked)
    }
}

/// Play one game to the end, searching before every pick, and return the
/// fraction of the optimum reached.
fn play(seed: u64, iterations: u32, graph: bool) -> f64 {
    let mut game = Synergy::new(seed);
    let best = game.best();
    let cfg = Config {
        iterations,
        exploration_constant: 0.3,
        ..Config::default()
    };
    let mut rng = WyRand::seed_from_u64(seed ^ 0xA5A5);
    let mut tree = Searcher::new(&game);
    let mut graph_searcher = GraphSearcher::new(&game);
    while game.picked.count_ones() < PICKS {
        let choice = if graph {
            graph_searcher
                .search(&game, &(), 0, &cfg, None, &mut rng)
                .choice
        } else {
            tree.search(&game, &(), 0, &cfg, None, &mut rng).choice
        };
        game.apply_choice(&(), &choice, &mut rng);
    }
    game.total(game.picked) / best
}

fn main() {
    let seeds = 80u64;
    println!("| iterations | tree: mean of optimum | tree: optimal | graph: mean of optimum | graph: optimal |");
    println!("|---:|---:|---:|---:|---:|");
    for iterations in [250u32, 1_000, 4_000, 16_000] {
        let mut rows = [(0.0f64, 0u32); 2];
        for (row, graph) in rows.iter_mut().zip([false, true]) {
            for seed in 0..seeds {
                let fraction = play(seed, iterations, graph);
                row.0 += fraction;
                row.1 += u32::from(fraction > 1.0 - 1e-12);
            }
        }
        println!(
            "| {iterations} | {:.4} | {}/{seeds} | {:.4} | {}/{seeds} |",
            rows[0].0 / seeds as f64,
            rows[0].1,
            rows[1].0 / seeds as f64,
            rows[1].1,
        );
    }
}
