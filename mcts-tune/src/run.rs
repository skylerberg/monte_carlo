use std::time::{Duration, Instant};

use mcts::Game;

use crate::arena::{evaluate, Evaluation, Match};
use crate::optimizer::Optimizer;
use crate::tunable::Tunable;

/// One tuning run's settings.
pub struct TuneConfig {
    pub generations: usize,
    pub evaluation: Evaluation,
    /// Draw a fresh set of shared game seeds each generation.
    ///
    /// The trade is real in both directions. Fixed seeds compare generations on
    /// identical ground, which is the cleanest reading of whether a run is
    /// making progress; but a candidate can then be selected for suiting those
    /// particular games, and nothing in the run will ever notice. Reseeding
    /// costs a little of that comparability and buys a population that has to
    /// keep being good on ground it has not seen.
    ///
    /// Either way the seeds are shared *within* a generation, which is where
    /// the variance reduction that matters lives.
    pub reseed_each_generation: bool,
}

/// What one generation found.
pub struct GenerationReport<'a> {
    pub generation: usize,
    /// The best candidate measured in this generation, and its win rate.
    pub best_genes: &'a [f64],
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub worst_fitness: f64,
    /// The best candidate measured so far in the run.
    ///
    /// This is a maximum over noisy measurements, so it is biased upward: the
    /// candidate that happens to be measured luckiest wins ties it would not
    /// win on a rerun. Treat it as the shortlist, not the answer, and re-measure
    /// it against the baseline before adopting it.
    pub incumbent_genes: &'a [f64],
    pub incumbent_fitness: f64,
    pub games: usize,
    pub elapsed: Duration,
}

/// A finished run.
pub struct TuneReport {
    pub best_genes: Vec<f64>,
    pub best_fitness: f64,
    pub generations: usize,
    pub games: usize,
}

/// Run the ask/measure/tell loop, reporting after each generation.
///
/// Every candidate is measured against the run's starting parameters, which
/// stay fixed for the whole run. A moving baseline — the incumbent, say — would
/// make each generation's fitness mean something different from the last, so
/// neither the numbers in the reports nor the noise analysis they rest on would
/// be comparable across a run.
///
/// `on_generation` is where checkpointing goes: this crate deliberately writes
/// no files and knows nothing about serialization, so the caller decides what a
/// generation's output looks like and where it lands.
pub fn run<M, O>(
    game: &M,
    optimizer: &mut O,
    config: &TuneConfig,
    mut on_generation: impl FnMut(&GenerationReport),
) -> TuneReport
where
    M: Match,
    O: Optimizer,
    <M::Game as Game>::Context: Sync,
{
    let baseline = game.base().to_genes();
    let mut games = 0;

    for generation in 0..config.generations {
        let started = Instant::now();

        let mut candidates = optimizer.ask();
        for genes in &mut candidates {
            assert_eq!(
                genes.len(),
                baseline.len(),
                "mcts-tune: {} proposed a candidate of {} genes where the parameters have {}",
                optimizer.name(),
                genes.len(),
                baseline.len()
            );
            M::Params::repair(genes);
        }

        let plan = Evaluation {
            games: config.evaluation.games,
            seed: if config.reseed_each_generation {
                // Multiplied through an odd constant rather than added to, so
                // consecutive generations do not overlap in the seeds they draw.
                config
                    .evaluation
                    .seed
                    .wrapping_add((generation as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            } else {
                config.evaluation.seed
            },
            threads: config.evaluation.threads,
        };

        let fitness = evaluate(game, &candidates, &baseline, &plan);
        games += candidates.len() * plan.games;
        optimizer.tell(&candidates, &fitness);

        let elapsed = started.elapsed();
        let (top, top_fitness) = fitness.iter().copied().enumerate().fold(
            (0usize, f64::NEG_INFINITY),
            |best, (index, value)| if value > best.1 { (index, value) } else { best },
        );
        let worst = fitness.iter().copied().fold(f64::INFINITY, f64::min);
        let mean = fitness.iter().sum::<f64>() / fitness.len().max(1) as f64;
        let (incumbent_genes, incumbent_fitness) = optimizer.best();

        on_generation(&GenerationReport {
            generation,
            best_genes: &candidates[top],
            best_fitness: top_fitness,
            mean_fitness: mean,
            worst_fitness: worst,
            incumbent_genes,
            incumbent_fitness,
            games,
            elapsed,
        });
    }

    let (best_genes, best_fitness) = optimizer.best();
    TuneReport {
        best_genes: best_genes.to_vec(),
        best_fitness,
        generations: config.generations,
        games,
    }
}
