use std::sync::atomic::{AtomicBool, Ordering};

use rand_core::Rng;

use crate::budget::Deadline;
use crate::early_stop;
use crate::game::{Game, Rewards, Status};
use crate::node::Node;
use crate::select::select;
use crate::util::below;

/// How often the wall clock is consulted, in iterations.
const DEADLINE_CHECK_MASK: u32 = 31;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Knobs for one search.
pub struct Config {
    /// Total visits the root should reach, counting any reused from a retained
    /// tree. Zero means run until `time_limit_ms` expires.
    pub iterations: u32,
    /// Wall-clock budget. Requires the `time` feature.
    pub time_limit_ms: Option<u64>,
    /// UCB1 exploration constant. Tune it against your reward scale.
    pub exploration_constant: f64,
    /// Weight on `Game::heuristic_bias`. Zero disables progressive bias, and
    /// skips evaluating the prior entirely.
    pub progressive_bias_weight: f64,
    /// Stop early once the remaining iterations provably cannot change the
    /// chosen move. Only applies when `iterations` is non-zero.
    pub early_termination: bool,
    /// Bounds of the reward scale, used by the early-termination proof. Set
    /// these to your game's actual range — the proof is unsound otherwise.
    pub max_reward: f64,
    /// See [`Config::max_reward`].
    pub min_reward: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            iterations: 10_000,
            time_limit_ms: None,
            exploration_constant: 0.75,
            progressive_bias_weight: 0.0,
            early_termination: true,
            max_reward: 1.0,
            min_reward: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Why a search stopped.
pub enum StopReason {
    /// Only one legal choice; no search was run.
    SingleChoice,
    /// The iteration budget was reached.
    Budget,
    /// The wall-clock budget expired.
    Deadline,
    /// The cancellation flag was set.
    Cancelled,
    /// The remaining iterations could not have changed the answer.
    Proven,
}

#[derive(Clone, Debug)]
/// What a search decided, and how it got there.
pub struct SearchResult<C> {
    /// The most visited root choice.
    pub choice: C,
    /// Iterations run by this call, excluding any inherited from a reused tree.
    pub iterations_used: u32,
    /// Iterations inherited from a retained subtree.
    pub reused_iterations: u32,
    /// Why the search stopped.
    pub stop_reason: StopReason,
    /// Total visits at the root, including reused ones.
    pub root_visits: u32,
    /// Visits to the chosen child.
    pub best_visits: u32,
    /// Mean reward of the chosen child, for the player who moved there.
    pub best_mean_reward: f64,
}

/// Owns every reusable allocation and the retained tree.
///
/// Keep one per player per game and call `search` repeatedly: the scratch state,
/// choice buffer and availability buffers are all reused, so a steady-state
/// iteration allocates nothing beyond growing the tree itself.
pub struct Searcher<G: Game> {
    scratch: G,
    choices: Vec<G::Choice>,
    avail: Vec<bool>,
    root_avail: Vec<bool>,
    path: Vec<u32>,
    side: G::Side,
    root: Option<Node<G::Choice>>,
    /// Whether the retained tree describes the position about to be searched.
    ///
    /// A search clears this; only `reuse_subtree` sets it. So carrying a tree
    /// forward is something you ask for, and forgetting to leaves you with a
    /// correct search rather than one rooted at last turn's position.
    tree_is_current: bool,
}

impl<G: Game> Searcher<G> {
    /// `template` is only used to size the scratch state; its contents are
    /// overwritten by the first determinization.
    pub fn new(template: &G) -> Self {
        Self {
            scratch: template.new_buffer(),
            choices: Vec::new(),
            avail: Vec::new(),
            root_avail: Vec::new(),
            path: Vec::new(),
            side: Default::default(),
            root: None,
            tree_is_current: false,
        }
    }

    /// The retained tree, for inspection or display.
    pub fn tree(&self) -> Option<&Node<G::Choice>> {
        self.root.as_ref()
    }

    /// Discard the retained tree.
    pub fn clear_tree(&mut self) {
        self.root = None;
        self.tree_is_current = false;
    }

    /// Re-root the retained tree at `choice` so the next search inherits its
    /// statistics. Returns false, and drops the tree, if `choice` is not a root
    /// child.
    ///
    /// Carrying a tree forward has to be asked for: a search leaves the tree
    /// marked stale, and only this re-arms it. Forgetting to call it therefore
    /// costs a fresh search rather than one rooted at the previous position.
    ///
    /// Only sound when the next search really continues from that position — the
    /// same player still to act, and no hidden information revealed in between.
    /// The caller owns that judgement.
    pub fn reuse_subtree(&mut self, choice: &G::Choice) -> bool {
        let Some(root) = self.root.as_mut() else {
            return false;
        };
        if root.reroot_at(choice) {
            self.tree_is_current = true;
            true
        } else {
            self.root = None;
            self.tree_is_current = false;
            false
        }
    }

    /// Search from `state` on behalf of `perspective` and return the most
    /// visited root choice.
    ///
    /// Panics if `state` is terminal, has no legal choices, or if the config
    /// specifies neither an iteration nor a time budget.
    pub fn search<R: Rng + ?Sized>(
        &mut self,
        state: &G,
        ctx: &G::Context,
        perspective: u8,
        cfg: &Config,
        cancel: Option<&AtomicBool>,
        rng: &mut R,
    ) -> SearchResult<G::Choice> {
        assert!(
            cfg.iterations != 0 || cfg.time_limit_ms.is_some(),
            "mcts: Config has neither an iteration nor a time budget"
        );

        let root_player = match state.status(ctx) {
            Status::Active { player } => player,
            Status::Terminal(_) => panic!("mcts: search called on a terminal state"),
        };

        self.choices.clear();
        state.choices_into(ctx, &mut self.choices);
        match self.choices.len() {
            0 => panic!("mcts: search called on a state with no legal choices"),
            1 => {
                return SearchResult {
                    choice: self.choices[0].clone(),
                    iterations_used: 0,
                    reused_iterations: 0,
                    stop_reason: StopReason::SingleChoice,
                    root_visits: 0,
                    best_visits: 0,
                    best_mean_reward: 0.0,
                }
            }
            _ => {}
        }

        // Anything retained from an earlier search describes an earlier
        // position unless `reuse_subtree` re-rooted it since.
        if !self.tree_is_current {
            self.root = None;
        }
        if self.root.is_none() {
            self.root = Some(Node::new_root(root_player));
        }
        state.init_side(ctx, &mut self.side);

        let Self {
            scratch,
            choices,
            avail,
            root_avail,
            path,
            side,
            root,
            tree_is_current,
        } = self;
        let root = root.as_mut().expect("root was just created");

        let target = cfg.iterations;
        let deadline = Deadline::new(cfg.time_limit_ms);
        let reused_iterations = root.visits;
        let mut iterations_used: u32 = 0;

        let stop_reason = loop {
            if target != 0 && root.visits >= target {
                break StopReason::Budget;
            }
            if let Some(flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    break StopReason::Cancelled;
                }
            }
            if iterations_used & DEADLINE_CHECK_MASK == 0 && deadline.expired() {
                break StopReason::Deadline;
            }

            G::begin_iteration(side);
            state.determinize_into(scratch, ctx, perspective, rng);
            scratch.advance(ctx, side, perspective, rng);

            let rewards = run_iteration::<G, R>(
                root,
                scratch,
                ctx,
                side,
                choices,
                avail,
                root_avail,
                path,
                cfg,
                perspective,
                rng,
            );

            G::credit_iteration(side, &rewards);
            iterations_used += 1;

            if cfg.early_termination && target != 0 && early_stop::settled(root, target, cfg) {
                break StopReason::Proven;
            }
        };

        let (choice, best_visits, best_mean_reward) = match most_visited(root) {
            Some(i) => (
                root.children[i]
                    .choice()
                    .expect("child has a choice")
                    .clone(),
                root.children[i].visits(),
                root.children[i].mean_reward(),
            ),
            // Cancelled or timed out before a single iteration completed.
            None => {
                choices.clear();
                state.choices_into(ctx, choices);
                let k = below(rng, choices.len() as u64) as usize;
                (choices[k].clone(), 0, 0.0)
            }
        };

        *tree_is_current = false;

        SearchResult {
            choice,
            iterations_used,
            reused_iterations,
            stop_reason,
            root_visits: root.visits(),
            best_visits,
            best_mean_reward,
        }
    }
}

/// First child with the highest visit count, or `None` if none were visited.
/// First rather than last, so ties break deterministically.
fn most_visited<C>(root: &Node<C>) -> Option<usize> {
    let mut best = None;
    let mut best_visits = 0;
    for (i, child) in root.children().iter().enumerate() {
        if child.visits() > best_visits {
            best_visits = child.visits();
            best = Some(i);
        }
    }
    best
}

/// One selection-expansion-simulation-backpropagation pass.
///
/// The descent records the child index taken at each level rather than
/// recursing, so tree depth costs a `u32` per level instead of a stack frame and
/// a deep game cannot overflow the stack. Backpropagation walks the same path
/// back down from the root.
#[allow(clippy::too_many_arguments)]
fn run_iteration<G: Game, R: Rng + ?Sized>(
    root: &mut Node<G::Choice>,
    state: &mut G,
    ctx: &G::Context,
    side: &mut G::Side,
    choices: &mut Vec<G::Choice>,
    avail: &mut Vec<bool>,
    root_avail: &mut Vec<bool>,
    path: &mut Vec<u32>,
    cfg: &Config,
    perspective: u8,
    rng: &mut R,
) -> G::Rewards {
    path.clear();

    let rewards: G::Rewards;
    let mut node = &mut *root;

    loop {
        let player = match state.status(ctx) {
            Status::Terminal(terminal_rewards) => {
                rewards = terminal_rewards;
                break;
            }
            Status::Active { player } => player,
        };

        // An already-expanded root whose choice set cannot vary needs neither
        // enumeration nor expansion, which is the difference between paying for
        // move generation once per search and once per iteration.
        let reuse_root = G::ROOT_CHOICES_INVARIANT && node.is_root() && !node.children.is_empty();
        let available: &[bool] = if reuse_root {
            root_avail.clear();
            root_avail.resize(node.children.len(), true);
            root_avail
        } else {
            choices.clear();
            state.choices_into(ctx, choices);
            node.expand(choices, player, G::CHILD_INDEX_THRESHOLD, avail, rng);
            avail
        };

        let Some(i) = select(
            node,
            available,
            cfg.exploration_constant,
            cfg.progressive_bias_weight,
        ) else {
            rewards = G::Rewards::zero();
            break;
        };

        path.push(i as u32);
        let choice = node.children[i].choice().expect("child has a choice");
        state.apply_choice(ctx, choice, rng);
        state.advance(ctx, side, perspective, rng);

        if node.children[i].visits() == 0 {
            if cfg.progressive_bias_weight != 0.0 {
                node.children[i].heuristic_bias = state.heuristic_bias(ctx, perspective);
            }
            rewards = state.rollout(ctx, rng);
            break;
        }

        node = &mut node.children[i];
    }

    let mut node = &mut *root;
    node.record(rewards.reward(node.player));
    for &i in path.iter() {
        node = &mut node.children[i as usize];
        node.record(rewards.reward(node.player));
    }

    rewards
}
