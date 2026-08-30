//! End-to-end cover for the tuning loop, on a game small enough to reason about.
//!
//! The game is a drafting race: eight items sit on the table, players alternate
//! taking one, and the winner is whoever's haul is worth more under the game's
//! *true* item weights. A player's search context holds its *believed* weights,
//! which feed the leaf evaluation and nothing else — the same split
//! `mcts-tune` relies on everywhere, and the reason a candidate whose beliefs
//! are closer to the truth drafts better and wins more.

use std::hash::Hash;

use mcts::rand_core::Rng;
use mcts::{Config, Game, Status};
use mcts_tune::{
    evaluate, play, CmaEs, CmaParams, Evaluation, Ga, GaParams, Match, Optimizer, Tunable,
    TuneConfig,
};

const ITEMS: usize = 8;
const UNCLAIMED: u8 = 2;

/// What each item is really worth, by kind. A drafter who believes kind 0 is
/// the valuable one is right.
const TRUE_WEIGHT: [f64; 2] = [3.0, 1.0];
const VALUE: [f64; ITEMS] = [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];
const KIND: [usize; ITEMS] = [0, 1, 0, 1, 1, 0, 1, 0];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Draft {
    owner: [u8; ITEMS],
    to_move: u8,
    remaining: u8,
}

impl Draft {
    fn new() -> Self {
        Self {
            owner: [UNCLAIMED; ITEMS],
            to_move: 0,
            remaining: ITEMS as u8,
        }
    }

    /// A player's haul under some set of weights.
    fn haul(&self, player: u8, weights: [f64; 2]) -> f64 {
        (0..ITEMS)
            .filter(|&item| self.owner[item] == player)
            .map(|item| VALUE[item] * weights[KIND[item]])
            .sum()
    }
}

/// The believed weights, which is what a run tunes.
#[derive(Clone, Debug, PartialEq)]
struct Beliefs {
    kind_zero: f64,
    kind_one: f64,
}

impl Tunable for Beliefs {
    fn gene_names() -> &'static [&'static str] {
        &["kindZero", "kindOne"]
    }

    fn to_genes(&self) -> Vec<f64> {
        vec![self.kind_zero, self.kind_one]
    }

    fn with_genes(&self, genes: &[f64]) -> Self {
        Self {
            kind_zero: genes[0],
            kind_one: genes[1],
        }
    }

    fn repair(genes: &mut [f64]) {
        for gene in genes {
            *gene = gene.clamp(0.01, 20.0);
        }
    }
}

impl Game for Draft {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = Beliefs;
    type Side = ();

    const ROOT_CHOICES_INVARIANT: bool = true;

    fn status(&self, _ctx: &Beliefs) -> Status<[f64; 2]> {
        if self.remaining > 0 {
            return Status::Active {
                player: self.to_move,
            };
        }
        // Scored on the true weights: the rules do not care what either player
        // believed while drafting.
        let light = self.haul(0, TRUE_WEIGHT);
        let dark = self.haul(1, TRUE_WEIGHT);
        Status::Terminal(if light > dark {
            [1.0, 0.0]
        } else if dark > light {
            [0.0, 1.0]
        } else {
            [0.5, 0.5]
        })
    }

    fn choices_into(&self, _ctx: &Beliefs, out: &mut Vec<u8>) {
        out.clear();
        for item in 0..ITEMS {
            if self.owner[item] == UNCLAIMED {
                out.push(item as u8);
            }
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _ctx: &Beliefs, choice: &u8, _rng: &mut R) {
        self.owner[*choice as usize] = self.to_move;
        self.to_move = 1 - self.to_move;
        self.remaining -= 1;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &Beliefs, _rng: &mut R) -> [f64; 2] {
        if let Status::Terminal(rewards) = self.status(ctx) {
            return rewards;
        }
        let weights = [ctx.kind_zero, ctx.kind_one];
        let difference = self.haul(0, weights) - self.haul(1, weights);
        let light = 0.5 + 0.5 * (difference / 10.0).tanh();
        [light, 1.0 - light]
    }

    fn new_buffer(&self) -> Self {
        *self
    }

    fn determinize_into<R: Rng + ?Sized>(
        &self,
        dest: &mut Self,
        _ctx: &Beliefs,
        _perspective: u8,
        _rng: &mut R,
    ) {
        *dest = *self;
    }
}

struct DraftMatch {
    base: Beliefs,
    iterations: u32,
}

impl Match for DraftMatch {
    type Game = Draft;
    type Params = Beliefs;

    fn base(&self) -> &Beliefs {
        &self.base
    }

    fn initial_state<R: Rng + ?Sized>(&self, _rng: &mut R) -> Draft {
        Draft::new()
    }

    fn context(&self, params: &Beliefs) -> Beliefs {
        params.clone()
    }

    fn config(&self) -> Config {
        Config {
            iterations: self.iterations,
            time_limit_ms: None,
            max_reward: 1.0,
            min_reward: 0.0,
            ..Config::default()
        }
    }
}

fn arena(base: Beliefs) -> DraftMatch {
    DraftMatch {
        base,
        iterations: 300,
    }
}

fn truthful() -> Beliefs {
    Beliefs {
        kind_zero: 3.0,
        kind_one: 1.0,
    }
}

fn inverted() -> Beliefs {
    Beliefs {
        kind_zero: 1.0,
        kind_one: 3.0,
    }
}

#[test]
fn a_game_is_reproducible_from_its_seed() {
    let table = arena(truthful());
    let one = table.context(&truthful());
    let two = table.context(&inverted());

    let first = play(&table, [&one, &two], 12345);
    let second = play(&table, [&one, &two], 12345);
    assert_eq!(first.rewards, second.rewards);
    assert_eq!(first.decisions, second.decisions);
    assert!(!first.hit_cap);
}

#[test]
fn every_game_reaches_a_terminal_position() {
    let table = arena(truthful());
    let one = table.context(&truthful());
    let two = table.context(&inverted());
    for seed in 0..8 {
        let outcome = play(&table, [&one, &two], seed);
        assert!(!outcome.hit_cap, "seed {seed} ran to the decision cap");
        assert_eq!(outcome.decisions, ITEMS as u32);
    }
}

/// The seat swap is the whole reason `evaluate` can be read as strength rather
/// than as a first-move advantage. A candidate identical to the baseline plays
/// the identical game from both seats, so the two halves have to cancel to
/// exactly one half.
#[test]
fn identical_parameters_score_exactly_even() {
    let table = arena(truthful());
    let baseline = truthful().to_genes();
    let scores = evaluate(
        &table,
        std::slice::from_ref(&baseline),
        &baseline,
        &Evaluation {
            games: 8,
            seed: 99,
            threads: 2,
        },
    );
    assert_eq!(scores.len(), 1);
    assert!(
        (scores[0] - 0.5).abs() < 1e-12,
        "an exact mirror scored {}",
        scores[0]
    );
}

#[test]
fn better_beliefs_beat_worse_ones() {
    let table = arena(inverted());
    let scores = evaluate(
        &table,
        &[truthful().to_genes(), inverted().to_genes()],
        &inverted().to_genes(),
        &Evaluation {
            games: 16,
            seed: 7,
            threads: 4,
        },
    );
    assert!(
        scores[0] > 0.5,
        "truthful beliefs scored {} against inverted ones",
        scores[0]
    );
    assert!(
        (scores[1] - 0.5).abs() < 1e-12,
        "the baseline against itself scored {}",
        scores[1]
    );
}

#[test]
fn evaluation_is_independent_of_thread_count() {
    let table = arena(inverted());
    let candidates = vec![truthful().to_genes(), inverted().to_genes()];
    let baseline = inverted().to_genes();
    let plan = |threads| Evaluation {
        games: 8,
        seed: 3,
        threads,
    };
    let one = evaluate(&table, &candidates, &baseline, &plan(1));
    let many = evaluate(&table, &candidates, &baseline, &plan(8));
    assert_eq!(one, many);
}

/// Both optimizers, driven on a fitness they cannot game: distance to a known
/// target. This is where a sign error in the CMA-ES update or a bad comparison
/// in the GA's ranking shows up, without paying for games to find it.
fn maximize<O: Optimizer>(
    optimizer: &mut O,
    target: &[f64],
    generations: usize,
) -> (Vec<f64>, f64) {
    for _ in 0..generations {
        let candidates = optimizer.ask();
        let fitness: Vec<f64> = candidates
            .iter()
            .map(|genes| {
                let distance: f64 = genes
                    .iter()
                    .zip(target)
                    .map(|(gene, want)| (gene - want) * (gene - want))
                    .sum();
                -distance
            })
            .collect();
        optimizer.tell(&candidates, &fitness);
    }
    let (genes, fitness) = optimizer.best();
    (genes.to_vec(), fitness)
}

/// A gene seeded at zero has no magnitude of its own, and the step size it gets
/// instead decides whether it is tunable or frozen. The median of its
/// neighbours puts it in the same range as the rest of the vector.
#[test]
fn a_zero_gene_borrows_its_step_from_its_neighbours() {
    let scales = Beliefs {
        kind_zero: 0.0,
        kind_one: 4.0,
    }
    .gene_scales();
    assert_eq!(scales, vec![4.0, 4.0]);

    // Nothing to borrow from falls back to one rather than to zero, which would
    // freeze the whole vector.
    let empty = Beliefs {
        kind_zero: 0.0,
        kind_one: 0.0,
    }
    .gene_scales();
    assert_eq!(empty, vec![1.0, 1.0]);
}

/// The behaviour the step size exists for: a weight seeded at zero has to be
/// able to reach a useful value. Under the old hardcoded `1e-3` floor this gene
/// moved about 0.00025 per mutation and stayed at zero for any run anyone would
/// actually wait for.
#[test]
fn both_optimizers_move_a_gene_off_zero() {
    let start = Beliefs {
        kind_zero: 0.0,
        kind_one: 4.0,
    };
    let target = [7.0, 2.0];

    let mut cma = CmaEs::new(
        &start,
        CmaParams {
            seed: 9,
            ..CmaParams::default()
        },
    );
    let (genes, _) = maximize(&mut cma, &target, 120);
    assert!(
        (genes[0] - target[0]).abs() < 0.5,
        "cma-es left the zeroed gene at {}",
        genes[0]
    );

    let mut ga = Ga::new(
        &start,
        GaParams {
            seed: 9,
            ..GaParams::default()
        },
    );
    let (genes, _) = maximize(&mut ga, &target, 120);
    assert!(
        genes[0] > 3.0,
        "the ga left the zeroed gene at {}",
        genes[0]
    );
}

#[test]
fn cma_es_closes_on_a_known_optimum() {
    let start = Beliefs {
        kind_zero: 1.0,
        kind_one: 1.0,
    };
    let target = [7.0, 2.0];
    let mut optimizer = CmaEs::new(
        &start,
        CmaParams {
            seed: 4,
            ..CmaParams::default()
        },
    );
    let (genes, fitness) = maximize(&mut optimizer, &target, 120);
    assert!(
        fitness > -0.05,
        "cma-es stalled at fitness {fitness} with genes {genes:?}"
    );
    for (gene, want) in genes.iter().zip(&target) {
        assert!(
            (gene - want).abs() < 0.2,
            "cma-es landed on {genes:?}, wanted {target:?}"
        );
    }
}

#[test]
fn the_ga_improves_on_its_seed() {
    let start = Beliefs {
        kind_zero: 1.0,
        kind_one: 1.0,
    };
    let target = [7.0, 2.0];
    let mut optimizer = Ga::new(
        &start,
        GaParams {
            seed: 4,
            ..GaParams::default()
        },
    );
    let (genes, fitness) = maximize(&mut optimizer, &target, 120);
    let seeded = -((1.0 - 7.0f64).powi(2) + (1.0 - 2.0f64).powi(2));
    assert!(
        fitness > seeded + 30.0,
        "the ga reached {fitness} from {seeded}, genes {genes:?}"
    );
}

/// A repair that clamps has to be reflected in what the optimizer is told,
/// otherwise the distribution adapts around candidates that were never played.
#[test]
fn candidates_are_repaired_before_they_are_measured() {
    let start = Beliefs {
        kind_zero: 1.0,
        kind_one: 1.0,
    };
    let mut optimizer = CmaEs::new(
        &start,
        CmaParams {
            initial_sigma: 30.0,
            seed: 11,
            ..CmaParams::default()
        },
    );
    let mut candidates = optimizer.ask();
    for genes in &mut candidates {
        Beliefs::repair(genes);
    }
    for genes in &candidates {
        for gene in genes {
            assert!((0.01..=20.0).contains(gene), "repair let {gene} through");
        }
    }
}

#[test]
fn a_full_run_reports_every_generation() {
    let table = arena(inverted());
    let mut optimizer = CmaEs::new(
        &inverted(),
        CmaParams {
            population: 4,
            seed: 21,
            ..CmaParams::default()
        },
    );
    let mut seen = Vec::new();
    let report = mcts_tune::run(
        &table,
        &mut optimizer,
        &TuneConfig {
            generations: 3,
            evaluation: Evaluation {
                games: 4,
                seed: 5,
                threads: 4,
            },
            reseed_each_generation: true,
        },
        |generation| seen.push((generation.generation, generation.best_fitness)),
    );

    assert_eq!(seen.len(), 3);
    assert_eq!(
        seen.iter().map(|(g, _)| *g).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(report.generations, 3);
    assert_eq!(report.games, 3 * 4 * 4);
    assert_eq!(report.best_genes.len(), Beliefs::gene_names().len());
}
