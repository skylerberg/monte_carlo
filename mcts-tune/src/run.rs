use std::time::{Duration, Instant};

use mcts::Game;
use serde::{Deserialize, Serialize};

use crate::arena::{evaluate, Evaluation, Match};
use crate::optimizer::Optimizer;
use crate::resume::{maybe_infinite, ResumeError};
use crate::tunable::Tunable;

/// Everything needed to continue a run that was interrupted.
///
/// [`GenerationReport`] carries one of these after every generation, so a
/// caller that persists it loses at most one generation to a kill. The crate
/// still writes no files: where it goes, and whether it is written atomically,
/// is the caller's decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The strategy that wrote it, so a mismatched resume is refused rather
    /// than quietly starting over.
    pub strategy: String,
    /// The optimizer's own state — for CMA-ES the adapted covariance and step
    /// size, which is what an interruption would otherwise cost.
    pub optimizer: serde_json::Value,
    /// Generations already finished. A resumed run starts here.
    pub generations_done: usize,
    pub games: usize,
    /// The genes every candidate was measured against.
    ///
    /// Checked on resume, because fitness only means anything relative to this.
    pub baseline: Vec<f64>,
    pub best_genes: Vec<f64>,
    #[serde(with = "maybe_infinite")]
    pub best_fitness: f64,
}

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
    /// The run's state as of this generation. Persist it to make the run
    /// resumable; ignore it and an interruption costs everything.
    pub checkpoint: Checkpoint,
}

/// A finished run.
#[derive(Debug, Clone)]
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
/// `on_generation` is where output goes: this crate deliberately writes no
/// files, so the caller decides what a generation's output looks like and where
/// it lands. That includes [`GenerationReport::checkpoint`], which is what
/// makes the run resumable.
///
/// Pass `resume` to continue an interrupted run. The checkpoint's baseline must
/// match this run's, and the error explains why when it does not: fitness is a
/// win rate *against the baseline*, so resuming against a different one would
/// keep producing numbers that are no longer comparable to the ones before the
/// interruption. The commonest way to get that wrong is resuming with the
/// previous run's output as the seed parameters.
pub fn run<M, O>(
    game: &M,
    optimizer: &mut O,
    config: &TuneConfig,
    resume: Option<&Checkpoint>,
    mut on_generation: impl FnMut(&GenerationReport),
) -> Result<TuneReport, ResumeError>
where
    M: Match,
    // `?Sized` so a caller that picks its strategy at runtime can pass a
    // `&mut dyn Optimizer` rather than duplicating this call per strategy.
    O: Optimizer + ?Sized,
    <M::Game as Game>::Context: Sync,
{
    let baseline = game.base().to_genes();
    let mut games = 0;
    let mut first = 0;

    if let Some(checkpoint) = resume {
        if checkpoint.baseline != baseline {
            return Err(ResumeError::Baseline {
                expected: baseline.len(),
                found: checkpoint.baseline.len(),
            });
        }
        optimizer.restore(&checkpoint.optimizer)?;
        first = checkpoint.generations_done;
        games = checkpoint.games;
    }

    for generation in first..config.generations {
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
        let checkpoint = Checkpoint {
            strategy: optimizer.name().to_string(),
            optimizer: optimizer.snapshot(),
            generations_done: generation + 1,
            games,
            baseline: baseline.clone(),
            best_genes: incumbent_genes.to_vec(),
            best_fitness: incumbent_fitness,
        };

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
            checkpoint,
        });
    }

    let (best_genes, best_fitness) = optimizer.best();
    Ok(TuneReport {
        best_genes: best_genes.to_vec(),
        best_fitness,
        generations: config.generations,
        games,
    })
}
