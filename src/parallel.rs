use std::sync::atomic::AtomicBool;

use rand_core::Rng;
use rustc_hash::FxHashMap;

use crate::game::Game;
use crate::node::Node;
use crate::search::{Config, SearchResult, Searcher, StopReason};

/// Root parallelisation: independent trees, independent determinizations, one
/// merge at the end.
///
/// Each worker keeps its own retained tree, so subtree reuse works here too.
/// Every `Send`/`Sync` bound lives on this type, not on [`Game`], so games that
/// are not thread-safe and targets without threads are unaffected.
pub struct RootParallel<G: Game, R> {
    workers: Vec<(Searcher<G>, R)>,
}

impl<G, R> RootParallel<G, R>
where
    G: Game + Send + Sync,
    G::Choice: Send + Sync,
    G::Context: Sync,
    G::Side: Send,
    R: Rng + Send,
{
    /// `make_rng` is called once per worker with its index. Derive each seed
    /// deterministically from one master seed if you want reproducible runs;
    /// results are then reproducible for a fixed thread count.
    pub fn new(threads: usize, template: &G, mut make_rng: impl FnMut(usize) -> R) -> Self {
        assert!(threads > 0, "mcts: RootParallel needs at least one thread");
        Self {
            workers: (0..threads)
                .map(|i| (Searcher::new(template), make_rng(i)))
                .collect(),
        }
    }

    /// Number of workers.
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// The per-worker trees, for inspection.
    pub fn trees(&self) -> impl Iterator<Item = &Node<G::Choice>> {
        self.workers.iter().filter_map(|(s, _)| s.tree())
    }

    /// Re-root every worker's tree at `choice`.
    pub fn reuse_subtree(&mut self, choice: &G::Choice) {
        for (searcher, _) in &mut self.workers {
            searcher.reuse_subtree(choice);
        }
    }

    /// Discard every worker's retained tree.
    pub fn clear_trees(&mut self) {
        for (searcher, _) in &mut self.workers {
            searcher.clear_tree();
        }
    }

    /// `cfg.iterations` is the budget **per worker**, matching what an
    /// independent tree per thread means: N threads do N times the work.
    pub fn search(
        &mut self,
        state: &G,
        ctx: &G::Context,
        perspective: u8,
        cfg: &Config,
        cancel: Option<&AtomicBool>,
    ) -> SearchResult<G::Choice> {
        let results: Vec<SearchResult<G::Choice>> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .workers
                .iter_mut()
                .map(|(searcher, rng)| {
                    scope.spawn(move || searcher.search(state, ctx, perspective, cfg, cancel, rng))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("mcts: worker thread panicked"))
                .collect()
        });

        self.merge(results)
    }

    fn merge(&self, results: Vec<SearchResult<G::Choice>>) -> SearchResult<G::Choice> {
        let stop_reason = merged_stop_reason(&results);
        if stop_reason == StopReason::SingleChoice {
            return results.into_iter().next().expect("at least one worker");
        }

        // Summed in first-seen order rather than hash order, so the merge is
        // deterministic for a fixed thread count.
        let mut order: Vec<G::Choice> = Vec::new();
        let mut position: FxHashMap<G::Choice, usize> = FxHashMap::default();
        let mut totals: Vec<(u32, f64)> = Vec::new();

        for (searcher, _) in &self.workers {
            let Some(root) = searcher.tree() else {
                continue;
            };
            for child in root.children() {
                let Some(choice) = child.choice() else {
                    continue;
                };
                let index = match position.get(choice) {
                    Some(&index) => index,
                    None => {
                        let index = order.len();
                        order.push(choice.clone());
                        position.insert(choice.clone(), index);
                        totals.push((0, 0.0));
                        index
                    }
                };
                totals[index].0 += child.visits();
                totals[index].1 += child.cumulative_reward();
            }
        }

        let mut best = 0;
        let mut best_visits = 0;
        for (i, &(visits, _)) in totals.iter().enumerate() {
            if visits > best_visits {
                best_visits = visits;
                best = i;
            }
        }

        if order.is_empty() {
            // No worker completed an iteration; fall back to any worker's answer.
            return results.into_iter().next().expect("at least one worker");
        }

        let best_mean_reward = if best_visits == 0 {
            0.0
        } else {
            totals[best].1 / best_visits as f64
        };

        SearchResult {
            choice: order.swap_remove(best),
            iterations_used: results.iter().map(|r| r.iterations_used).sum(),
            reused_iterations: results.iter().map(|r| r.reused_iterations).sum(),
            stop_reason,
            root_visits: results.iter().map(|r| r.root_visits).sum(),
            best_visits,
            best_mean_reward,
        }
    }
}

/// The most informative reason any worker stopped for.
fn merged_stop_reason<C>(results: &[SearchResult<C>]) -> StopReason {
    let rank = |reason: StopReason| match reason {
        StopReason::SingleChoice => 4,
        StopReason::Cancelled => 3,
        StopReason::Deadline => 2,
        StopReason::Proven => 1,
        StopReason::Budget => 0,
    };
    results
        .iter()
        .map(|r| r.stop_reason)
        .max_by_key(|&reason| rank(reason))
        .unwrap_or(StopReason::Budget)
}
