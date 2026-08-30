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
    evaluate, play, Checkpoint, CmaEs, CmaParams, Evaluation, Ga, GaParams, Match, Opponents,
    Optimizer, ResumeError, Tunable, TuneConfig,
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
            opponents: Opponents::Baseline,
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
            opponents: Opponents::Baseline,
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
        opponents: Opponents::Baseline,
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
                opponents: Opponents::Baseline,
            },
            reseed_each_generation: true,
        },
        None,
        |generation| seen.push((generation.generation, generation.best_fitness)),
    )
    .expect("a fresh run cannot fail to resume");

    assert_eq!(seen.len(), 3);
    assert_eq!(
        seen.iter().map(|(g, _)| *g).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(report.generations, 3);
    assert_eq!(report.games, 3 * 4 * 4);
    assert_eq!(report.best_genes.len(), Beliefs::gene_names().len());
}

// ── Resume ──

fn resume_config(generations: usize) -> TuneConfig {
    TuneConfig {
        generations,
        evaluation: Evaluation {
            games: 4,
            seed: 5,
            threads: 2,
            opponents: Opponents::Baseline,
        },
        reseed_each_generation: true,
    }
}

/// A checkpoint as a caller actually keeps one: written to text and parsed
/// back.
///
/// Every resume test goes through this rather than passing the in-memory value
/// straight back. Handing the `Checkpoint` object over directly tests a code
/// path nobody uses, and it hides the whole class of bug where the state
/// survives in memory and not on disk — which is exactly what
/// `serde_json`'s float parser does to a run that stores its numbers as JSON
/// numbers.
fn through_a_file(checkpoint: &Checkpoint) -> Checkpoint {
    let text = serde_json::to_string(checkpoint).expect("a checkpoint serializes");
    serde_json::from_str(&text).expect("a checkpoint parses")
}

fn resume_params() -> CmaParams {
    CmaParams {
        population: 4,
        seed: 21,
        ..CmaParams::default()
    }
}

/// The test the whole feature exists for. Stopping after three generations and
/// resuming into a *fresh* optimizer has to produce the run that was never
/// interrupted — not merely a similar one. Anything less means the covariance,
/// the step size or the generator was rebuilt rather than restored, and the
/// generations after a resume would be exploring differently from the ones
/// before it while the log gave no sign.
#[test]
fn a_resumed_run_continues_exactly_where_it_stopped() {
    let table = arena(inverted());

    let mut uninterrupted = CmaEs::new(&inverted(), resume_params());
    let mut straight = Vec::new();
    let whole = mcts_tune::run(
        &table,
        &mut uninterrupted,
        &resume_config(6),
        None,
        |generation| straight.push((generation.generation, generation.best_fitness)),
    )
    .expect("a fresh run cannot fail to resume");

    let mut before = CmaEs::new(&inverted(), resume_params());
    let mut split = Vec::new();
    let mut saved: Option<Checkpoint> = None;
    mcts_tune::run(&table, &mut before, &resume_config(3), None, |generation| {
        split.push((generation.generation, generation.best_fitness));
        saved = Some(through_a_file(&generation.checkpoint));
    })
    .expect("a fresh run cannot fail to resume");

    let mut after = CmaEs::new(&inverted(), resume_params());
    let resumed = mcts_tune::run(
        &table,
        &mut after,
        &resume_config(6),
        saved.as_ref(),
        |generation| split.push((generation.generation, generation.best_fitness)),
    )
    .expect("the checkpoint matches this run");

    assert_eq!(
        straight, split,
        "a resumed run diverged from an uninterrupted one"
    );
    assert_eq!(resumed.best_genes, whole.best_genes);
    assert_eq!(resumed.best_fitness, whole.best_fitness);
    assert_eq!(
        resumed.games, whole.games,
        "games were double-counted or lost"
    );
}

#[test]
fn a_checkpoint_carries_the_optimizer_state_not_just_the_answer() {
    let mut optimizer = CmaEs::new(&inverted(), resume_params());
    let candidates = optimizer.ask();
    let fitness: Vec<f64> = (0..candidates.len()).map(|i| i as f64).collect();
    optimizer.tell(&candidates, &fitness);

    let snapshot = optimizer.snapshot();
    let expected = optimizer.ask();

    // A fresh optimizer proposes something else entirely until it is restored.
    let mut restored = CmaEs::new(&inverted(), resume_params());
    assert_ne!(restored.ask(), expected);
    restored
        .restore(&snapshot)
        .expect("same strategy, same dimension");
    assert_eq!(restored.ask(), expected);
}

#[test]
fn a_checkpoint_from_another_strategy_is_refused() {
    let mut ga = Ga::new(&inverted(), GaParams::default());
    let mut cma = CmaEs::new(&inverted(), resume_params());
    assert_eq!(
        cma.restore(&ga.snapshot()),
        Err(ResumeError::Strategy {
            expected: "cma-es",
            found: String::from("ga"),
        })
    );
    // And the other way, so neither direction silently starts over.
    assert!(matches!(
        ga.restore(&cma.snapshot()),
        Err(ResumeError::Strategy { .. })
    ));
}

/// Resuming against a different baseline is the trap this guard exists for: the
/// run would carry on producing win rates that are no longer comparable to the
/// ones before the interruption, and nothing in the log would say so.
#[test]
fn a_checkpoint_measured_against_another_baseline_is_refused() {
    let table = arena(inverted());
    let mut optimizer = CmaEs::new(&inverted(), resume_params());
    let mut saved: Option<Checkpoint> = None;
    mcts_tune::run(
        &table,
        &mut optimizer,
        &resume_config(1),
        None,
        |generation| {
            saved = Some(generation.checkpoint.clone());
        },
    )
    .expect("a fresh run cannot fail to resume");

    // Same strategy and dimension, different seed parameters.
    let elsewhere = arena(truthful());
    let mut fresh = CmaEs::new(&truthful(), resume_params());
    let outcome = mcts_tune::run(
        &elsewhere,
        &mut fresh,
        &resume_config(2),
        saved.as_ref(),
        |_| {},
    );
    assert!(
        matches!(outcome, Err(ResumeError::Baseline { .. })),
        "{outcome:?}"
    );
}

#[test]
fn a_malformed_checkpoint_is_refused() {
    let mut optimizer = CmaEs::new(&inverted(), resume_params());
    let outcome = optimizer.restore(&serde_json::json!({ "strategy": "cma-es" }));
    assert!(
        matches!(outcome, Err(ResumeError::Malformed(_))),
        "{outcome:?}"
    );
}

/// A checkpoint taken before anything has been measured holds a best fitness of
/// negative infinity, which JSON cannot carry as a number.
#[test]
fn an_unmeasured_optimizer_still_round_trips() {
    let mut optimizer = CmaEs::new(&inverted(), resume_params());
    let snapshot = optimizer.snapshot();
    let text = serde_json::to_string(&snapshot).expect("serializes");
    let read: serde_json::Value = serde_json::from_str(&text).expect("parses");
    optimizer.restore(&read).expect("restores");
    assert_eq!(optimizer.best().1, f64::NEG_INFINITY);
}

/// The bug this guards against is not hypothetical: `serde_json` writes an
/// `f64` correctly and reads it back up to one unit in the last place away. A
/// checkpoint holding its numbers as JSON numbers therefore restores *nearly*
/// the state it saved, and a search that compounds its state every generation
/// turns "nearly" into a different run.
#[test]
fn checkpointed_floats_survive_text_exactly() {
    let mut optimizer = CmaEs::new(&inverted(), resume_params());
    let candidates = optimizer.ask();
    let fitness: Vec<f64> = (0..candidates.len()).map(|i| i as f64 * 0.37).collect();
    optimizer.tell(&candidates, &fitness);

    let text = serde_json::to_string(&optimizer.snapshot()).expect("serializes");
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("parses");
    assert_eq!(
        parsed,
        optimizer.snapshot(),
        "the snapshot changed on its way through text"
    );

    // And the values a plain JSON number would have damaged are the ones that
    // decide the next candidate.
    let mut restored = CmaEs::new(&inverted(), resume_params());
    restored.restore(&parsed).expect("restores");
    assert_eq!(restored.snapshot(), optimizer.snapshot());
}

#[test]
fn a_checkpoint_for_a_different_population_is_refused() {
    let optimizer = CmaEs::new(&inverted(), resume_params());
    let snapshot = optimizer.snapshot();

    let mut wider = CmaEs::new(
        &inverted(),
        CmaParams {
            population: 8,
            ..resume_params()
        },
    );
    assert!(matches!(
        wider.restore(&snapshot),
        Err(ResumeError::Malformed(_))
    ));
}

// ── Round robin ──

/// Scores are zero sum inside the field, so whatever the candidates are, the
/// population average is one half. That is the property that makes the mode
/// ceiling-free — a population cannot run away from an opponent that is itself
/// — and the same property that stops the fitness column being read as progress.
#[test]
fn round_robin_scores_average_to_a_half() {
    let table = arena(inverted());
    let candidates = vec![
        truthful().to_genes(),
        inverted().to_genes(),
        Beliefs {
            kind_zero: 2.0,
            kind_one: 2.0,
        }
        .to_genes(),
        Beliefs {
            kind_zero: 5.0,
            kind_one: 0.5,
        }
        .to_genes(),
    ];
    let scores = evaluate(
        &table,
        &candidates,
        &inverted().to_genes(),
        &Evaluation {
            games: 4,
            seed: 11,
            threads: 3,
            opponents: Opponents::RoundRobin,
        },
    );
    assert_eq!(scores.len(), candidates.len());
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    assert!(
        (mean - 0.5).abs() < 1e-12,
        "the field averaged {mean}, not one half: {scores:?}"
    );
}

/// The ranking still has to mean something: truthful beliefs draft better, so
/// they should come out above inverted ones without a baseline anywhere.
#[test]
fn round_robin_ranks_the_better_candidate_higher() {
    let table = arena(inverted());
    let candidates = vec![
        inverted().to_genes(),
        truthful().to_genes(),
        Beliefs {
            kind_zero: 1.0,
            kind_one: 1.0,
        }
        .to_genes(),
    ];
    let scores = evaluate(
        &table,
        &candidates,
        &inverted().to_genes(),
        &Evaluation {
            games: 6,
            seed: 4,
            threads: 2,
            opponents: Opponents::RoundRobin,
        },
    );
    assert!(
        scores[1] > scores[0],
        "truthful {} did not beat inverted {}",
        scores[1],
        scores[0]
    );
}

/// Identical candidates cannot be told apart, and the seat swap is what
/// guarantees it: without it whichever sat first would collect the first-move
/// advantage and the optimizer would chase a difference that is not there.
#[test]
fn identical_candidates_tie_in_a_round_robin() {
    let table = arena(inverted());
    let twins = vec![truthful().to_genes(), truthful().to_genes()];
    let scores = evaluate(
        &table,
        &twins,
        &inverted().to_genes(),
        &Evaluation {
            games: 8,
            seed: 21,
            threads: 2,
            opponents: Opponents::RoundRobin,
        },
    );
    assert_eq!(scores[0], 0.5);
    assert_eq!(scores[1], 0.5);
}

#[test]
fn round_robin_is_independent_of_thread_count() {
    let table = arena(inverted());
    let candidates = vec![
        truthful().to_genes(),
        inverted().to_genes(),
        Beliefs {
            kind_zero: 3.0,
            kind_one: 3.0,
        }
        .to_genes(),
    ];
    let plan = |threads| Evaluation {
        games: 4,
        seed: 6,
        threads,
        opponents: Opponents::RoundRobin,
    };
    let one = evaluate(&table, &candidates, &inverted().to_genes(), &plan(1));
    let many = evaluate(&table, &candidates, &inverted().to_genes(), &plan(8));
    assert_eq!(one, many);
}

/// With a moving field the historical maximum is not the answer: an early
/// generation's score was won against weaker opposition, so keeping it would
/// hand back a candidate from before the run improved. `run` has to report the
/// latest generation instead.
#[test]
fn a_round_robin_run_reports_its_latest_generation() {
    let table = arena(inverted());
    let mut optimizer = CmaEs::new(
        &inverted(),
        CmaParams {
            population: 4,
            seed: 3,
            ..CmaParams::default()
        },
    );
    let mut last = Vec::new();
    let report = mcts_tune::run(
        &table,
        &mut optimizer,
        &TuneConfig {
            generations: 4,
            evaluation: Evaluation {
                games: 4,
                seed: 8,
                threads: 2,
                opponents: Opponents::RoundRobin,
            },
            reseed_each_generation: false,
        },
        None,
        |generation| last = generation.best_genes.to_vec(),
    )
    .expect("a fresh run cannot fail to resume");

    assert_eq!(
        report.best_genes, last,
        "the run reported something other than its final generation's best"
    );
}

/// The reported game count has to be the games actually played. `games` is per
/// pairing, and there are `n choose 2` pairings, so using the baseline formula
/// would under-report a round robin by more than threefold at eleven
/// candidates — and the cost of a run is sized from this number.
#[test]
fn a_round_robin_generation_costs_what_it_claims() {
    let table = arena(inverted());
    let mut optimizer = CmaEs::new(
        &inverted(),
        CmaParams {
            population: 4,
            seed: 3,
            ..CmaParams::default()
        },
    );
    let report = mcts_tune::run(
        &table,
        &mut optimizer,
        &TuneConfig {
            generations: 2,
            evaluation: Evaluation {
                games: 4,
                seed: 8,
                threads: 2,
                opponents: Opponents::RoundRobin,
            },
            reseed_each_generation: false,
        },
        None,
        |_| {},
    )
    .expect("a fresh run cannot fail to resume");

    // 4 choose 2 pairings, 4 games each, over 2 generations.
    assert_eq!(report.games, 2 * 6 * 4);
}
