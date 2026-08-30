use std::sync::atomic::{AtomicUsize, Ordering};

use mcts::rand_core::SeedableRng;
use mcts::{Config, Game, Rewards, Searcher, Status};

use crate::Tunable;
use wyrand::WyRand;

/// How to start a game and how to read a candidate's genes as a search context.
///
/// # The context contract
///
/// The two seats search with *different* contexts, and the driver has to pass
/// one of them to [`Game::status`] and [`Game::apply_choice`] to advance the
/// position. It passes seat 0's. That is only sound because the tuned part of a
/// context feeds the leaf evaluation and nothing else — a context that changes
/// which choices are legal, or when the game ends, would have the two seats
/// playing different games and the measurement would be meaningless. The same
/// assumption is already load-bearing inside the search itself, which applies
/// choices under whichever context it was called with.
pub trait Match: Sync {
    type Game: Game;
    /// The parameter struct being tuned.
    type Params: Tunable;

    /// The parameters a run starts from. Supplies every field the gene vector
    /// does not carry, so anything [`Tunable::to_genes`] leaves out keeps these
    /// values for the whole run.
    fn base(&self) -> &Self::Params;

    /// A fresh position. A game whose setup is random should draw it from
    /// `rng`, which is seeded per game; a deterministic opening ignores it.
    fn initial_state<R: mcts::rand_core::Rng + ?Sized>(&self, rng: &mut R) -> Self::Game;

    /// The search context these parameters describe.
    fn context(&self, params: &Self::Params) -> <Self::Game as Game>::Context;

    /// Search budget for every evaluation game. Both seats get the same one —
    /// what is being compared is the parameters, not the budget.
    fn config(&self) -> Config;

    /// Cap on decisions in one game, so a rules bug cannot hang a run that is
    /// meant to be left alone overnight.
    fn max_decisions(&self) -> u32 {
        10_000
    }

    /// The context a gene vector describes.
    ///
    /// Genes arrive already repaired — [`crate::run`] does that before anything
    /// sees them, so that the optimizer is told about the candidate that was
    /// actually played rather than the one it proposed.
    fn context_for(&self, genes: &[f64]) -> <Self::Game as Game>::Context {
        self.context(&self.base().with_genes(genes))
    }
}

/// One finished game.
pub struct MatchOutcome<Rw> {
    /// Payoffs as the game reported them, indexed by seat.
    pub rewards: Rw,
    pub decisions: u32,
    /// Whether the game was cut off at [`Match::max_decisions`] rather than
    /// reaching a terminal position. A run with these in it is measuring
    /// something other than what it thinks.
    pub hit_cap: bool,
}

/// Play one game between two contexts, seat 0 against seat 1.
///
/// Deterministic in `seed`: the same seed and the same pair of contexts replay
/// the same game, which is what lets [`evaluate`] compare candidates on shared
/// randomness.
pub fn play<M: Match>(
    game: &M,
    contexts: [&<M::Game as Game>::Context; 2],
    seed: u64,
) -> MatchOutcome<<M::Game as Game>::Rewards> {
    let mut rng = WyRand::seed_from_u64(seed);
    let mut state = game.initial_state(&mut rng);
    let config = game.config();
    let max_decisions = game.max_decisions();

    // One searcher per seat, both advanced past every choice, so each keeps the
    // subtree under the line actually played.
    let mut searchers = [Searcher::new(&state), Searcher::new(&state)];
    let mut decisions = 0;

    while decisions < max_decisions {
        let player = match state.status(contexts[0]) {
            Status::Terminal(rewards) => {
                return MatchOutcome {
                    rewards,
                    decisions,
                    hit_cap: false,
                }
            }
            Status::Active { player } => player,
            Status::Simultaneous { .. } => panic!(
                "mcts-tune: this game has simultaneous nodes, which the tuning driver does \
                 not play. A simultaneous node needs each seat's own context to pick that \
                 seat's marginal and the pair applied jointly; `play` only walks sequential \
                 positions."
            ),
        };

        let seat = player as usize;
        assert!(
            seat < 2,
            "mcts-tune: `play` seats exactly two players, but the game reported player \
             {player} to move. Tuning a game with more seats needs a different driver."
        );

        let result =
            searchers[seat].search(&state, contexts[seat], player, &config, None, &mut rng);
        state.apply_choice(contexts[0], &result.choice, &mut rng);
        for searcher in &mut searchers {
            searcher.reuse_subtree(&result.choice);
        }
        decisions += 1;
    }

    // A cut-off game has no result to report, so it scores the midpoint of the
    // declared reward range rather than a fabricated win for either seat.
    let midpoint = 0.5 * (config.min_reward + config.max_reward);
    MatchOutcome {
        rewards: <M::Game as Game>::Rewards::uniform(midpoint),
        decisions,
        hit_cap: true,
    }
}

/// Who the candidates are measured against.
///
/// The choice decides what a fitness number *means*, and the two meanings are
/// not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opponents {
    /// Every candidate plays the run's starting parameters.
    ///
    /// Fitness is then an absolute number on a fixed scale, comparable across
    /// the whole run: 0.62 in generation 40 really is better than 0.55 in
    /// generation 3. That is the mode to read progress from.
    ///
    /// It has a ceiling, and runs hit it. Once the population beats the
    /// baseline nine times in ten, every candidate scores between 0.88 and 0.95
    /// and the gaps between them are smaller than the standard error on each
    /// measurement — the ranking selection depends on becomes noise, and the
    /// run stops improving because it has run out of measuring instrument
    /// rather than because it has found anything. There is a second failure
    /// underneath that one: candidates are rewarded for punishing the *specific*
    /// opponent they all face, which is not the same as being strong.
    Baseline,
    /// Every candidate plays every other candidate.
    ///
    /// The field improves as the population does, so there is no ceiling to hit
    /// and no fixed opponent to specialise against — the pressure is always to
    /// beat the current best, whatever that has become.
    ///
    /// The cost is that fitness stops being an absolute number. Scores are
    /// zero-sum within the field, so the population mean sits at 0.5 by
    /// construction in every generation, and a 0.62 late in a run is measured
    /// against much stronger opposition than a 0.62 early on. Progress cannot
    /// be read from the fitness column; measure a generation's output against
    /// fixed weights separately to see it.
    ///
    /// It is more efficient than it looks. Every game scores *two* candidates
    /// rather than one, so at equal cost each candidate is measured on twice as
    /// many games as it would be against a baseline.
    ///
    /// The risk it carries is cycling: with three strategies where each beats
    /// one of the others, a population can chase itself in circles and improve
    /// at nothing. Suspect it when fitness is lively but the parameters keep
    /// returning to values they held before.
    RoundRobin,
}

/// How much evidence to gather about each candidate.
pub struct Evaluation {
    /// Games per candidate. The single most important number in a run: fitness
    /// is a win rate, so the standard error is at worst `sqrt(0.25 / games)`,
    /// and a generation whose candidates differ by less than that is ranked by
    /// noise. Halve it and you need four times as many generations to make the
    /// same progress, which is a bad trade well before 200.
    ///
    /// Under [`Opponents::RoundRobin`] this counts games per *pairing*, so each
    /// candidate is measured on `games * (candidates - 1)` of them and a
    /// generation costs `games * candidates * (candidates - 1) / 2` in total.
    pub games: usize,
    /// Base seed for the shared game seeds. Changing it between generations
    /// trades a little variance reduction for protection against a candidate
    /// that happens to suit one fixed set of seeds.
    pub seed: u64,
    pub threads: usize,
    pub opponents: Opponents,
}

/// Win rate for each candidate, on shared randomness.
///
/// Two variance reductions, both free and applied in either mode:
///
/// - **Common random numbers.** Game `g` is played from seed `seed + g` for
///   *every* candidate and every pairing, so candidates are compared on the same
///   draws rather than each against its own luck. Without this the difference
///   between two candidates carries both their sampling errors; with it, much of
///   that cancels.
/// - **Both seats.** The first half of a matchup seats one candidate first and
///   the second half seats it second, so a first-move advantage lands on both
///   sides of every comparison instead of being measured as strength.
///
/// Scores come from [`Rewards::reward`] at each candidate's own seat, so nothing
/// here assumes the game is zero sum.
///
/// `baseline` is used only by [`Opponents::Baseline`]; round-robin measures the
/// candidates against each other and ignores it.
pub fn evaluate<M: Match>(
    game: &M,
    candidates: &[Vec<f64>],
    baseline: &[f64],
    plan: &Evaluation,
) -> Vec<f64>
where
    <M::Game as Game>::Context: Sync,
{
    assert!(
        plan.games > 0,
        "mcts-tune: a candidate needs at least one game"
    );
    if candidates.is_empty() {
        return Vec::new();
    }

    let contexts: Vec<_> = candidates
        .iter()
        .map(|genes| game.context_for(genes))
        .collect();

    match plan.opponents {
        Opponents::Baseline => {
            let opponent = game.context_for(baseline);
            against_one(game, &contexts, &opponent, plan)
        }
        Opponents::RoundRobin => {
            assert!(
                candidates.len() >= 2,
                "mcts-tune: a round robin needs at least two candidates; with one there is \
                 nobody to play"
            );
            against_each_other(game, &contexts, plan)
        }
    }
}

/// Every candidate against a single fixed opponent.
fn against_one<M: Match>(
    game: &M,
    contexts: &[<M::Game as Game>::Context],
    opponent: &<M::Game as Game>::Context,
    plan: &Evaluation,
) -> Vec<f64>
where
    <M::Game as Game>::Context: Sync,
{
    let half = plan.games / 2;
    let total = contexts.len() * plan.games;
    let next = AtomicUsize::new(0);
    let (next, contexts) = (&next, &contexts);

    let sums = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..plan.threads.max(1))
            .map(|_| {
                scope.spawn(move || {
                    let mut local = vec![0.0f64; contexts.len()];
                    loop {
                        let job = next.fetch_add(1, Ordering::Relaxed);
                        if job >= total {
                            return local;
                        }
                        let candidate = job / plan.games;
                        let round = job % plan.games;

                        // The candidate takes seat 0 for the first half of its
                        // games and seat 1 for the second.
                        let seat = usize::from(round >= half);
                        let pair = if seat == 0 {
                            [&contexts[candidate], opponent]
                        } else {
                            [opponent, &contexts[candidate]]
                        };

                        let outcome = play(game, pair, plan.seed.wrapping_add(round as u64));
                        local[candidate] += outcome.rewards.reward(seat as u8);
                    }
                })
            })
            .collect();

        handles
            .into_iter()
            .fold(vec![0.0f64; contexts.len()], |mut total, handle| {
                for (slot, value) in total
                    .iter_mut()
                    .zip(handle.join().expect("worker panicked"))
                {
                    *slot += value;
                }
                total
            })
    });

    sums.into_iter()
        .map(|sum| sum / plan.games as f64)
        .collect()
}

/// Every candidate against every other, each pairing scoring both sides.
fn against_each_other<M: Match>(
    game: &M,
    contexts: &[<M::Game as Game>::Context],
    plan: &Evaluation,
) -> Vec<f64>
where
    <M::Game as Game>::Context: Sync,
{
    let pairs: Vec<(usize, usize)> = (0..contexts.len())
        .flat_map(|left| ((left + 1)..contexts.len()).map(move |right| (left, right)))
        .collect();

    let half = plan.games / 2;
    let total = pairs.len() * plan.games;
    let next = AtomicUsize::new(0);
    let (next, contexts, pairs) = (&next, &contexts, &pairs);

    let sums = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..plan.threads.max(1))
            .map(|_| {
                scope.spawn(move || {
                    let mut local = vec![0.0f64; contexts.len()];
                    loop {
                        let job = next.fetch_add(1, Ordering::Relaxed);
                        if job >= total {
                            return local;
                        }
                        let (left, right) = pairs[job / plan.games];
                        let round = job % plan.games;

                        // Which of the two sits first flips at the halfway
                        // point, exactly as it does against a baseline.
                        let seat = usize::from(round >= half);
                        let pair = if seat == 0 {
                            [&contexts[left], &contexts[right]]
                        } else {
                            [&contexts[right], &contexts[left]]
                        };

                        let outcome = play(game, pair, plan.seed.wrapping_add(round as u64));
                        // One game, two scores: this is why a round robin
                        // measures each candidate on twice the games a baseline
                        // does for the same cost.
                        local[left] += outcome.rewards.reward(seat as u8);
                        local[right] += outcome.rewards.reward(1 - seat as u8);
                    }
                })
            })
            .collect();

        handles
            .into_iter()
            .fold(vec![0.0f64; contexts.len()], |mut total, handle| {
                for (slot, value) in total
                    .iter_mut()
                    .zip(handle.join().expect("worker panicked"))
                {
                    *slot += value;
                }
                total
            })
    });

    let played = (contexts.len() - 1) * plan.games;
    sums.into_iter().map(|sum| sum / played as f64).collect()
}
