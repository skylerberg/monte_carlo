use core::hash::Hash;
use std::sync::atomic::AtomicBool;

use rand_core::Rng;
use rustc_hash::FxHashMap;

use crate::game::Game;
use crate::node::{Node, NodeKind};
use crate::search::{Config, RootPolicy, SearchResult, Searcher, StopReason};
use crate::util::uniform_01;

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

    /// Re-root every worker's tree at the joint action actually played.
    ///
    /// See [`Searcher::reuse_joint`], including its warning that misses are the
    /// common case — and they are commoner still here, because each worker
    /// materialized joint successors from its own determinizations, so a tuple
    /// present in one tree need not be present in any other.
    pub fn reuse_joint(&mut self, played: &[(u8, G::Choice)]) {
        for (searcher, _) in &mut self.workers {
            searcher.reuse_joint(played);
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

        self.merge(state, ctx, perspective, cfg, results)
    }

    /// Combine the per-worker trees into one answer.
    ///
    /// A simultaneous root merges the perspective player's **marginals**, keyed
    /// by `Choice`, and never touches a joint successor: joint edges are keyed
    /// by per-tree marginal indices, and each worker appended its arms in its
    /// own determinizations' order, so worker 1's arm 2 is not worker 2's arm 2
    /// and summing joint children across workers would pool the statistics of
    /// different action pairs with nothing to catch it. A `Choice` means the
    /// same thing in every tree.
    ///
    /// The merged answer is read off whatever the single-threaded extraction
    /// would read it off: pooled visits where the policy is pure, and the
    /// pooled mixed strategy — sampled or argmaxed, per [`RootPolicy`] — where
    /// it mixes. `best_visits` and `best_mean_reward` are always the pooled
    /// statistics of the action returned; they report on the action, not on how
    /// it was chosen.
    fn merge(
        &mut self,
        state: &G,
        ctx: &G::Context,
        perspective: u8,
        cfg: &Config,
        results: Vec<SearchResult<G::Choice>>,
    ) -> SearchResult<G::Choice> {
        let stop_reason = merged_stop_reason(&results);
        if stop_reason == StopReason::SingleChoice {
            return results.into_iter().next().expect("at least one worker");
        }

        let simultaneous = self.workers.iter().any(|(searcher, _)| {
            searcher
                .tree()
                .is_some_and(|root| root.simultaneous_players().is_some())
        });
        // Under `Duct` the per-worker strategy is one-hot at that worker's
        // most-visited arm, so pooled visits already say the same thing and
        // both root policies read the visit leader. Under a policy that mixes,
        // neither policy may: `duct::best_arm` argmaxes the same
        // availability-divided strategy weight `Sampled` draws from, so a merge
        // answering `MostVisited` with the pooled-visit argmax would return an
        // action the single-threaded path was deliberately corrected away from.
        let mixes = simultaneous && G::SIMULTANEOUS_POLICY.mixes();
        let sample = mixes && cfg.root_policy == RootPolicy::Sampled;

        let mut merged = Merged::new();
        let mut strategy = Vec::new();
        let mut shares = Vec::new();
        // Arms accumulate every action any determinization offered, so merging
        // them unfiltered can return a move the perspective player does not
        // have in the real position. The single-threaded path filters against
        // the position in `extract_marginal`; this is that filter, applied by
        // seeding the merge with the legal actions and refusing to add any
        // other. Sequential roots keep merging by insertion, unchanged.
        if simultaneous {
            let mut legal = Vec::new();
            state.choices_for_into(ctx, perspective, &mut legal);
            for choice in &legal {
                merged.slot(choice);
            }
        }
        for (searcher, _) in &self.workers {
            let Some(root) = searcher.tree() else {
                continue;
            };
            match root.simul() {
                Some(simul) => {
                    let Some(slot) = simul.players.slot_of(perspective) else {
                        continue;
                    };
                    let marginals = simul.marginals(slot);
                    let stats = &simul.arm_stats[simul.slot_range(slot)];
                    if mixes {
                        // Normalized per worker, then renormalized by `spread`
                        // over the arms that survive the legality filter: a raw
                        // `strategy_sum` would weight whichever tree ran
                        // longest, and a deadline or a cancellation makes
                        // worker iteration counts unequal in practice.
                        marginals.policy_into(&mut strategy);
                    }
                    shares.clear();
                    for arm in 0..marginals.len() {
                        let Some(index) = merged.get(marginals.choice(arm)) else {
                            continue;
                        };
                        merged.visits[index] += stats[arm].visits;
                        merged.reward[index] += stats[arm].cumulative_reward;
                        if mixes {
                            shares.push((index, strategy[arm]));
                        }
                    }
                    spread(root.visits() as f64, &shares, &mut merged.strategy);
                }
                None => {
                    for child in root.children() {
                        debug_assert_eq!(
                            child.kind(),
                            NodeKind::Choice,
                            "mcts: the sequential root merge reached a joint successor. Joint \
                             edges are keyed by per-tree marginal indices and must never be \
                             merged across workers; a simultaneous root merges the perspective \
                             player's marginals instead."
                        );
                        let choice = child
                            .choice()
                            .expect("a sequential root's children all carry a choice");
                        let index = merged.slot(choice);
                        merged.visits[index] += child.visits();
                        merged.reward[index] += child.cumulative_reward();
                    }
                }
            }
        }

        // Seeding makes `order` non-empty before any worker has contributed, so a
        // simultaneous merge asks whether anything was actually merged rather
        // than whether the map is empty.
        let contributed = merged.visits.iter().any(|&visits| visits > 0);
        if merged.order.is_empty() || (simultaneous && !contributed) {
            // No worker completed an iteration; fall back to any worker's answer.
            return results.into_iter().next().expect("at least one worker");
        }

        let best = if sample {
            // Worker 0's rng, borrowed only now that the thread scope has
            // closed, so the draw stays deterministic for a fixed thread count.
            let rng = &mut self.workers[0].1;
            sample_merged(&merged.strategy, rng).unwrap_or_else(|| merged.leader())
        } else if mixes {
            merged.strategy_leader().unwrap_or_else(|| merged.leader())
        } else {
            merged.leader()
        };

        let best_visits = merged.visits[best];
        let best_mean_reward = if best_visits == 0 {
            0.0
        } else {
            merged.reward[best] / best_visits as f64
        };

        SearchResult {
            choice: merged.order.swap_remove(best),
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

/// Per-`Choice` totals pooled across workers.
///
/// Kept in first-seen order rather than hash order, so the merged answer is
/// reproducible for a fixed thread count. The parallel arrays are indexed
/// together; `slot` is the only thing that appends to them.
struct Merged<C> {
    order: Vec<C>,
    position: FxHashMap<C, usize>,
    visits: Vec<u32>,
    reward: Vec<f64>,
    strategy: Vec<f64>,
}

impl<C: Clone + Eq + Hash> Merged<C> {
    fn new() -> Self {
        Self {
            order: Vec::new(),
            position: FxHashMap::default(),
            visits: Vec::new(),
            reward: Vec::new(),
            strategy: Vec::new(),
        }
    }

    /// The entry for `choice`, without creating one. A simultaneous merge seeds
    /// the map with the position's legal actions and then only ever looks up, so
    /// an arm for an action the player cannot play here has nowhere to land.
    fn get(&self, choice: &C) -> Option<usize> {
        self.position.get(choice).copied()
    }

    fn slot(&mut self, choice: &C) -> usize {
        match self.position.get(choice) {
            Some(&index) => index,
            None => {
                let index = self.order.len();
                self.order.push(choice.clone());
                self.position.insert(choice.clone(), index);
                self.visits.push(0);
                self.reward.push(0.0);
                self.strategy.push(0.0);
                index
            }
        }
    }

    /// The entry carrying the most merged strategy mass, first maximum so ties
    /// break deterministically. `None` if no worker contributed any, which is
    /// the caller's cue to fall back to pooled visits.
    fn strategy_leader(&self) -> Option<usize> {
        let mut best = None;
        let mut best_weight = 0.0;
        for (index, &weight) in self.strategy.iter().enumerate() {
            if weight > best_weight {
                best_weight = weight;
                best = Some(index);
            }
        }
        best
    }

    /// The most-visited entry, first maximum so ties break deterministically.
    fn leader(&self) -> usize {
        let mut best = 0;
        let mut best_visits = 0;
        for (index, &visits) in self.visits.iter().enumerate() {
            if visits > best_visits {
                best_visits = visits;
                best = index;
            }
        }
        best
    }
}

/// Add `weight` to `out`, split across `shares` in proportion to the mass each
/// carries.
///
/// `shares` is one worker's normalized strategy restricted to the arms that
/// survived the legality filter, so it sums to 1 only when every action that
/// worker saw is playable in the real position. Renormalizing over what
/// survived is the whole point: `Marginals::policy_into` normalizes over every
/// arm, and scaling those truncated numbers by `weight` would make a worker's
/// influence its visit count times whatever mass it happened to place on
/// unplayable actions — a quantity that varies several-fold between workers
/// that ran the same number of iterations. A worker whose surviving arms carry
/// no mass says nothing about this position and contributes nothing.
fn spread(weight: f64, shares: &[(usize, f64)], out: &mut [f64]) {
    let total: f64 = shares.iter().map(|&(_, share)| share).sum();
    if total <= 0.0 {
        return;
    }
    for &(index, share) in shares {
        out[index] += weight * share / total;
    }
}

/// Inverse-CDF draw over the merged strategy, or `None` if it carries no mass.
fn sample_merged<R: Rng + ?Sized>(strategy: &[f64], rng: &mut R) -> Option<usize> {
    let total: f64 = strategy.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let target = uniform_01(rng) * total;
    let mut acc = 0.0;
    let mut last = None;
    for (index, &weight) in strategy.iter().enumerate() {
        if weight <= 0.0 {
            continue;
        }
        acc += weight;
        if acc >= target {
            return Some(index);
        }
        last = Some(index);
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{JointChoices, PlayerSet, Status};
    use crate::util::below;
    use rand_core::SeedableRng;
    use wyrand::WyRand;

    /// Rock-paper-scissors on `[0, 1]`, indexed `[player 0][player 1]`.
    const RPS: [[f64; 3]; 3] = [[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]];

    fn config(iterations: u32) -> Config {
        Config {
            iterations,
            early_termination: false,
            ..Config::default()
        }
    }

    /// One sequential ply and nothing else: choice `c` pays `c / 2` to player 0,
    /// so the most-visited merge has one unambiguous answer.
    #[derive(Clone, Default)]
    struct Ladder {
        payoff: Option<f64>,
    }

    impl Game for Ladder {
        type Choice = u8;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        fn status(&self, _ctx: &()) -> Status<[f64; 2]> {
            match self.payoff {
                Some(payoff) => Status::Terminal([payoff, 1.0 - payoff]),
                None => Status::Active { player: 0 },
            }
        }

        fn choices_into(&self, _ctx: &(), out: &mut Vec<u8>) {
            out.extend([0, 1, 2]);
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _ctx: &(), choice: &u8, _rng: &mut R) {
            self.payoff = Some(*choice as f64 / 2.0);
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _ctx: &(), _rng: &mut R) -> [f64; 2] {
            let payoff = self.payoff.expect("the choice has been applied");
            [payoff, 1.0 - payoff]
        }

        fn new_buffer(&self) -> Self {
            Self::default()
        }

        fn determinize_into<R: Rng + ?Sized>(
            &self,
            dest: &mut Self,
            _ctx: &(),
            _perspective: u8,
            _rng: &mut R,
        ) {
            dest.clone_from(self);
        }
    }

    /// Two simultaneous plies of rock-paper-scissors. Two rather than one so a
    /// joint action leads to another decision node, which is the only thing
    /// `reuse_joint` has to re-root at.
    #[derive(Clone, Default)]
    struct TwoRoundRps {
        round: u8,
        payoff: f64,
    }

    impl TwoRoundRps {
        fn play(&mut self, a: u8, b: u8) {
            self.payoff += RPS[a as usize][b as usize];
            self.round += 1;
        }
    }

    impl Game for TwoRoundRps {
        type Choice = u8;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        fn status(&self, _ctx: &()) -> Status<[f64; 2]> {
            if self.round < 2 {
                Status::Simultaneous {
                    players: PlayerSet::first_n(2),
                }
            } else {
                let payoff = self.payoff / 2.0;
                Status::Terminal([payoff, 1.0 - payoff])
            }
        }

        fn choices_into(&self, _ctx: &(), out: &mut Vec<u8>) {
            out.extend([0, 1, 2]);
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _ctx: &(), _choice: &u8, _rng: &mut R) {
            unreachable!("every node of this game is simultaneous")
        }

        fn apply_joint<R: Rng + ?Sized>(
            &mut self,
            _ctx: &(),
            joint: JointChoices<'_, u8>,
            _rng: &mut R,
        ) {
            self.play(*joint.get(0), *joint.get(1));
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _ctx: &(), rng: &mut R) -> [f64; 2] {
            while self.round < 2 {
                self.play(below(rng, 3) as u8, below(rng, 3) as u8);
            }
            let payoff = self.payoff / 2.0;
            [payoff, 1.0 - payoff]
        }

        fn new_buffer(&self) -> Self {
            Self::default()
        }

        fn determinize_into<R: Rng + ?Sized>(
            &self,
            dest: &mut Self,
            _ctx: &(),
            _perspective: u8,
            _rng: &mut R,
        ) {
            dest.clone_from(self);
        }
    }

    /// Payoff to player 0 per action at [`RareFavourite`]; player 1 scores
    /// `1.0 - payoff` and its own action does not enter, so the fixture is one
    /// player's bandit and its right answer does not move.
    ///
    /// The two common actions pay the same on purpose. Regret matching then
    /// splits its mass between them on the three iterations in four that offer
    /// nothing else, so neither rivals [`RARE_ACTION`] on mass conditional on
    /// being legal while both comfortably outrank it on raw visits.
    const RARE_PAYOFFS: [f64; 3] = [0.5, 0.5, 1.0];

    /// The action [`RareFavourite`] pays best for and offers least often.
    const RARE_ACTION: u8 = 2;

    /// A simultaneous ply whose best action for player 0 is legal in the real
    /// position and legal in only one determinization in four.
    ///
    /// The mirror image of the merge's legality filter, and what tells a
    /// visit-argmax extraction from a strategy-argmax one. Raw visits rank
    /// [`RARE_ACTION`] last — three iterations in four cannot select it at all
    /// — while the availability-divided strategy weight ranks it first by more
    /// than two to one, because it takes nearly all the mass on the iterations
    /// where it is offered. `duct::best_arm` reads the latter, so a merge
    /// reading the former returns a different action from a single-threaded
    /// search of the same position.
    #[derive(Clone, Default)]
    struct RareFavourite {
        payoff: Option<f64>,
        withheld: bool,
    }

    impl Game for RareFavourite {
        type Choice = u8;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        fn status(&self, _ctx: &()) -> Status<[f64; 2]> {
            match self.payoff {
                Some(payoff) => Status::Terminal([payoff, 1.0 - payoff]),
                None => Status::Simultaneous {
                    players: PlayerSet::first_n(2),
                },
            }
        }

        fn choices_into(&self, _ctx: &(), _out: &mut Vec<u8>) {
            unreachable!("RareFavourite names each player's actions through choices_for_into")
        }

        fn choices_for_into(&self, _ctx: &(), player: u8, out: &mut Vec<u8>) {
            out.extend([0, 1]);
            if player == 0 && !self.withheld {
                out.push(RARE_ACTION);
            }
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _ctx: &(), _choice: &u8, _rng: &mut R) {
            unreachable!("every node of this game is simultaneous")
        }

        fn apply_joint<R: Rng + ?Sized>(
            &mut self,
            _ctx: &(),
            joint: JointChoices<'_, u8>,
            _rng: &mut R,
        ) {
            self.payoff = Some(RARE_PAYOFFS[*joint.get(0) as usize]);
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _ctx: &(), rng: &mut R) -> [f64; 2] {
            let withheld = self.withheld;
            let payoff = *self.payoff.get_or_insert_with(|| {
                RARE_PAYOFFS[below(rng, if withheld { 2 } else { 3 }) as usize]
            });
            [payoff, 1.0 - payoff]
        }

        fn new_buffer(&self) -> Self {
            Self::default()
        }

        fn determinize_into<R: Rng + ?Sized>(
            &self,
            dest: &mut Self,
            _ctx: &(),
            _perspective: u8,
            rng: &mut R,
        ) {
            dest.clone_from(self);
            dest.withheld = below(rng, 4) != 0;
        }
    }

    fn seeded<G>(threads: usize, template: &G) -> RootParallel<G, WyRand>
    where
        G: Game<Choice = u8, Context = (), Side = ()> + Send + Sync,
    {
        RootParallel::new(threads, template, |i| WyRand::seed_from_u64(i as u64 + 1))
    }

    /// Pooled arm visits per choice for `player`, and how many joint successors
    /// the workers materialized between them.
    fn pooled_arms<G>(workers: &RootParallel<G, WyRand>, player: u8) -> ([u32; 3], usize)
    where
        G: Game<Choice = u8, Context = (), Side = ()> + Send + Sync,
    {
        let mut pooled = [0u32; 3];
        let mut joint_children = 0;
        for root in workers.trees() {
            let marginals = root
                .marginals(player)
                .expect("this player acts at the root");
            for arm in 0..marginals.len() {
                pooled[*marginals.choice(arm) as usize] += marginals.visits(arm);
            }
            joint_children += root.children().len();
        }
        (pooled, joint_children)
    }

    /// The merged strategy the workers add up to, for `player`, at a fixture
    /// where every action is legal in the real position — so this mirrors the
    /// merge's arithmetic term for term rather than approximating it.
    fn pooled_strategy<G>(workers: &RootParallel<G, WyRand>, player: u8) -> [f64; 3]
    where
        G: Game<Choice = u8, Context = (), Side = ()> + Send + Sync,
    {
        let mut pooled = [0.0; 3];
        let mut strategy = Vec::new();
        for root in workers.trees() {
            let marginals = root
                .marginals(player)
                .expect("this player acts at the root");
            marginals.policy_into(&mut strategy);
            let shares: Vec<(usize, f64)> = strategy
                .iter()
                .enumerate()
                .map(|(arm, &share)| (*marginals.choice(arm) as usize, share))
                .collect();
            spread(root.visits() as f64, &shares, &mut pooled);
        }
        pooled
    }

    /// First maximum, matching the merge's own tie-break.
    fn leader(pooled: &[u32; 3]) -> usize {
        (0..3).fold(0, |best, i| if pooled[i] > pooled[best] { i } else { best })
    }

    /// First maximum, matching `Merged::strategy_leader`.
    fn strategy_leader(pooled: &[f64; 3]) -> usize {
        (0..3).fold(0, |best, i| if pooled[i] > pooled[best] { i } else { best })
    }

    #[test]
    fn a_sequential_merge_pools_visits_by_choice() {
        let game = Ladder::default();
        let mut workers = seeded(4, &game);
        let result = workers.search(&game, &(), 0, &config(200), None);

        assert_eq!(result.choice, 2);
        assert_eq!(result.root_visits, 800);
        let pooled: u32 = workers
            .trees()
            .map(|root| {
                root.children()
                    .iter()
                    .find(|child| child.choice() == Some(&2))
                    .map_or(0, |child| child.visits())
            })
            .sum();
        assert_eq!(result.best_visits, pooled);
        assert!((result.best_mean_reward - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_simultaneous_merge_pools_the_perspective_players_arms() {
        let game = TwoRoundRps::default();
        let cfg = Config {
            root_policy: RootPolicy::MostVisited,
            ..config(2_000)
        };
        let mut workers = seeded(4, &game);
        let result = workers.search(&game, &(), 0, &cfg, None);

        let (pooled, joint_children) = pooled_arms(&workers, 0);
        assert!(
            joint_children > 0,
            "the workers materialized joint successors, which the merge must ignore"
        );
        assert_eq!(result.root_visits, 8_000);
        // Under a policy that mixes, `MostVisited` is the argmax of the pooled
        // strategy — what `duct::best_arm` ranks — and not of pooled visits.
        // The pooled visits of that action are still what gets reported.
        assert_eq!(
            result.choice,
            strategy_leader(&pooled_strategy(&workers, 0)) as u8
        );
        assert_eq!(result.best_visits, pooled[result.choice as usize]);
        assert!(
            result.best_visits < result.root_visits,
            "an arm's visits are a share of the root's, not a joint child's"
        );
    }

    #[test]
    fn a_sampled_simultaneous_merge_draws_from_the_pooled_strategy() {
        let game = TwoRoundRps::default();
        let mut workers = seeded(4, &game);
        let mut seen = [false; 3];
        for _ in 0..12 {
            let result = workers.search(&game, &(), 0, &config(500), None);
            seen[result.choice as usize] = true;
        }
        assert!(
            seen.iter().filter(|&&hit| hit).count() >= 2,
            "the equilibrium of rock-paper-scissors is uniform, so a sampled merge \
             must not keep returning one action"
        );
    }

    /// The merge must answer `MostVisited` with the action a single-threaded
    /// search of the same position answers with. Under `RegretMatching` that is
    /// the argmax of the availability-divided strategy weight, not of raw arm
    /// visits: at [`RareFavourite`] the two rank different actions, and the
    /// assertion below that the pooled-visit leader is *not* the returned
    /// action is what keeps this test from passing on a coincidence.
    #[test]
    fn a_most_visited_merge_agrees_with_a_single_threaded_search() {
        let game = RareFavourite::default();
        let cfg = Config {
            root_policy: RootPolicy::MostVisited,
            ..config(4_000)
        };

        let mut solo = Searcher::new(&game);
        let mut rng = WyRand::seed_from_u64(97);
        let single = solo.search(&game, &(), 0, &cfg, None, &mut rng);

        let mut workers = seeded(4, &game);
        let result = workers.search(&game, &(), 0, &cfg, None);

        let (pooled, _) = pooled_arms(&workers, 0);
        assert_ne!(
            leader(&pooled) as u8,
            RARE_ACTION,
            "the fixture bites only while raw pooled visits rank the best action \
             below an action that is merely offered more often: {pooled:?}"
        );
        assert_eq!(
            single.choice, RARE_ACTION,
            "the single-threaded extraction divides by availability"
        );
        assert_eq!(
            result.choice, single.choice,
            "one position, one policy, two answers"
        );
        assert_eq!(
            result.best_visits, pooled[result.choice as usize],
            "pooled visits are still what `best_visits` reports"
        );
    }

    /// Two workers that ran the same number of iterations must get the same say
    /// in the merged strategy, however much of their own mass sat on actions
    /// that are not playable in the real position and so never reached the
    /// merge. The numbers are one worker's normalized strategy with the illegal
    /// entries already dropped: the first kept 0.15 of its mass, the second
    /// 0.50. Scaling those remainders raw hands the second worker 3.3 times the
    /// influence its visits bought it — and flips the answer, since the two
    /// workers disagree about which legal action leads.
    #[test]
    fn a_workers_say_in_the_merge_is_its_visits_not_its_legal_mass() {
        let mut merged = [0.0; 2];
        spread(1_000.0, &[(0, 0.10), (1, 0.05)], &mut merged);
        spread(1_000.0, &[(0, 0.20), (1, 0.30)], &mut merged);

        assert!(
            (merged[0] + merged[1] - 2_000.0).abs() < 1e-9,
            "each worker contributes exactly its visits: {merged:?}"
        );
        assert!(
            (merged[0] - (2_000.0 / 3.0 + 400.0)).abs() < 1e-9,
            "{merged:?}"
        );
        assert!(
            (merged[1] - (1_000.0 / 3.0 + 600.0)).abs() < 1e-9,
            "{merged:?}"
        );
        assert!(
            merged[0] > merged[1],
            "the first worker prefers action 0 two to one and the second prefers \
             action 1 three to two, so with equal say action 0 leads: {merged:?}"
        );
    }

    /// A worker whose surviving arms carry no mass has nothing to say about the
    /// legal actions. It contributes no strategy — and, in the merge, its visits
    /// and rewards still pool.
    #[test]
    fn a_worker_with_no_legal_mass_contributes_no_strategy() {
        let mut merged = [7.0, 3.0];
        spread(1_000.0, &[(0, 0.0), (1, 0.0)], &mut merged);
        assert_eq!(merged, [7.0, 3.0]);
    }

    #[test]
    fn reuse_joint_re_roots_the_workers_holding_the_tuple() {
        let game = TwoRoundRps::default();
        let mut workers = seeded(4, &game);
        let first = workers.search(&game, &(), 0, &config(500), None);

        let (opponent_arms, _) = pooled_arms(&workers, 1);
        let opponent = leader(&opponent_arms) as u8;
        workers.reuse_joint(&[(0, first.choice), (1, opponent)]);
        // Workers that never materialized this tuple drop their trees, which is
        // the expected case rather than a failure.
        assert!(workers.trees().count() > 0);

        let mut played = game.clone();
        played.play(first.choice, opponent);
        let second = workers.search(&played, &(), 0, &config(500), None);
        assert!(second.reused_iterations > 0);
        assert!(second.choice < 3);
    }
}
