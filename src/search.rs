#[cfg(debug_assertions)]
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(debug_assertions)]
use hashbrown::HashTable;

use rand_core::Rng;

use crate::budget::Deadline;
use crate::duct;
use crate::early_stop;
use crate::game::{Game, JointChoices, PlayerSet, Rewards, SimultaneousPolicy, Status};
use crate::node::{JointKey, Node};
use crate::rank::{leader_of, Candidate};
use crate::select::select;
use crate::util::below;
#[cfg(debug_assertions)]
use crate::util::hash_of;

/// How often the wall clock is consulted, in iterations.
const DEADLINE_CHECK_MASK: u32 = 31;

/// How the returned move is read off a simultaneous root. Ignored at a
/// sequential root, which always answers with its leading child.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RootPolicy {
    /// Sample the perspective player's mixed strategy, renormalized over the
    /// actions legal in the real position.
    ///
    /// The default, because a game with a mixed equilibrium has no single best
    /// move and returning one makes you exploitable by exactly as much as the
    /// equilibrium mixes. This consumes one draw from the `rng` passed to
    /// [`Searcher::search`], after the search loop — so a fixed seed still
    /// reproduces the whole run, but the returned move is not the argmax of
    /// anything. Under [`crate::SimultaneousPolicy::Duct`] the strategy is
    /// one-hot, so this is identical to [`RootPolicy::MostVisited`].
    #[default]
    Sampled,
    /// Return the perspective player's leading action: the one carrying the
    /// most weight under the policy in use, which is not the same thing as the
    /// most-selected one.
    ///
    /// An action is selectable only on the iterations that offered it, so raw
    /// selections rank by legality rate as much as by value; both policies
    /// divide that out, and [`crate::Marginals::most_visited`] is the raw count
    /// if you want it.
    ///
    /// Deterministic and reproducible, and exploitable wherever the equilibrium
    /// is mixed. Worth choosing when you need pinned output, or when you are
    /// deliberately exploiting an opponent you know does not best-respond. Never
    /// an argmax over joint successors: your action appears in one joint child
    /// per opponent action, so the best *pair* is not the best action.
    MostVisited,
}

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
    ///
    /// No effect at a simultaneous node, where the prior is never evaluated:
    /// `heuristic_bias` describes a state, so at a simultaneous node it
    /// describes a joint successor, and selection there scores one player's
    /// arms. A game that tunes progressive bias and then adds a simultaneous
    /// node loses the term at exactly those nodes, silently.
    pub progressive_bias_weight: f64,
    /// Stop early once the remaining iterations provably cannot change the
    /// chosen move.
    ///
    /// The answer is the best mean reward among the candidates the search
    /// selected enough times to trust, and a mean moves with the *rewards*, so
    /// almost nothing about it can be proved from the counts in hand — the
    /// rewards this crate accumulates are whatever the game returned, not
    /// values clamped into [`Config::min_reward`]`..=`[`Config::max_reward`].
    /// What is still proved is a claim about evidence: every rival is so far
    /// from being sampled enough to be trusted that the iterations left cannot
    /// get it there. That is a narrow case. Where two candidates are both well
    /// sampled the search now spends its whole budget and reports
    /// [`StopReason::Budget`]; turning this on costs a branch per iteration and
    /// buys nothing there. A branch and not a scan: while more iterations are
    /// left than the evidence bar is high, no rival can be out of reach and the
    /// pass over the root's candidates is skipped without being made. Only applies when `iterations` is non-zero, and a
    /// search that spends its whole budget reports [`StopReason::Budget`]
    /// rather than [`StopReason::Proven`].
    pub early_termination: bool,
    /// Bounds of the reward scale. Set these to your game's actual range.
    ///
    /// They are load-bearing at a simultaneous node, and in a way worth
    /// spelling out because the default range is `[0, 1]` and a zero-sum game
    /// paying in `[-1, 1]` is the likeliest thing to meet it. Regret matching
    /// *clamps* payoffs into `[0, 1]` before touching a regret. It does not
    /// rescale anything else: its strategy is invariant under any positive
    /// rescaling of every regret, so the range reaches the search through the
    /// clamp and nowhere else. Left at `[0, 1]`, a game paying `-1` for a loss
    /// and `0` for a draw hands regret matching the same number for both, at
    /// every budget, with no other symptom. Debug builds assert that observed
    /// rewards fall inside the declared range at simultaneous nodes.
    pub max_reward: f64,
    /// See [`Config::max_reward`].
    pub min_reward: f64,
    /// The knobs that apply only at a [`Status::Simultaneous`] node.
    ///
    /// Grouped rather than flattened so that `Config`'s own fields are the ones
    /// live for *every* game. A purely sequential game leaves this at its
    /// default forever and never has to work out which of ten knobs reach it;
    /// a game with simultaneous nodes gets the three that do, in one place,
    /// next to each other's measurements.
    pub simultaneous: SimultaneousConfig,
}

/// Knobs that take effect only at a [`Status::Simultaneous`] node.
///
/// Every field here is dead for a game that never returns
/// [`Status::Simultaneous`], which is the reason they are not on [`Config`]
/// beside the knobs that always apply.
// `Copy` because every field is, and because `Config` is routinely built with
// `..base` update syntax: a non-`Copy` field there would move out of `base` and
// partially invalidate a `Config` callers expect to keep using.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimultaneousConfig {
    /// UCB1's exploration constant at a simultaneous node, under
    /// [`crate::SimultaneousPolicy::Duct`].
    ///
    /// Same units and same scale as [`Config::exploration_constant`] — a
    /// multiplier on `sqrt(ln availability / visits)`, on the reward scale — and
    /// a separate field only because a simultaneous node's bandits are coupled.
    /// At a sequential node the constant sets how fast the bandit concentrates
    /// and the limit does not depend on it; at a simultaneous node the cycle two
    /// decoupled UCB1 players settle into is a function of their constants, so
    /// the value is part of the answer rather than part of the schedule.
    ///
    /// Measured on biased rock-paper-scissors, 16 seeds times both slots, mean
    /// and worst exploitability of the *visit* distribution at 50k iterations:
    /// 0.209 / 0.500 at `c = 0.05`, 0.042 / 0.235 at `c = 0.1`, 0.032 / 0.043
    /// at `c = 0.75`, 0.014 / 0.019 at `c = 2.0`. Anything from about 0.5 upward
    /// behaves; below 0.2 decoupled UCB1 collapses onto a pure pair and an
    /// opponent takes half the game's value off you. The 0.1 this field's
    /// predecessor defaulted to is inside that bad band, so splitting the field
    /// fixed a `Duct` default as well as a regret matching one. The default
    /// matches
    /// [`Config::exploration_constant`] rather than the RPS optimum on purpose:
    /// `Duct` is the policy to pick when you believe your simultaneous nodes
    /// have *pure* equilibria, and tuning it on a game with a mixed one would
    /// buy a number that only looks good where the policy is the wrong choice.
    pub duct_exploration: f64,
    /// The uniform-exploration probability a
    /// [`crate::SimultaneousPolicy::RegretMatching`] node mixes into its
    /// sampling distribution on its first visit.
    ///
    /// A probability, not a rate on the reward scale, which is why it is not
    /// the same field as [`SimultaneousConfig::duct_exploration`]: the two natural values
    /// differ by an order of magnitude and share no usable range. The floor
    /// decays from here as `gamma_0 * t^(-1/4)` in that node's own visit count
    /// and is clamped into `[0.005, 0.5]`, so this is the starting value rather
    /// than the value.
    ///
    /// Raising it explores harder and leaves a larger residual exploitability
    /// on the way; lowering it inflates the importance weight backup divides by,
    /// which the floor bounds at `arms / gamma_t`. The default sits in the band
    /// every published SM-MCTS study tunes over, and `duct.rs`'s
    /// `exploration_floor` carries the measurements behind both the value and
    /// the exponent — including why the `1 / sqrt(t)` schedule this crate first
    /// shipped does not converge at all.
    pub regret_matching_exploration: f64,
    /// Whether a simultaneous root returns a sampled move or the leading one.
    pub root_policy: RootPolicy,
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
            simultaneous: SimultaneousConfig::default(),
        }
    }
}

impl Default for SimultaneousConfig {
    fn default() -> Self {
        Self {
            duct_exploration: 0.75,
            regret_matching_exploration: 0.4,
            root_policy: RootPolicy::Sampled,
        }
    }
}

impl SimultaneousConfig {
    /// Whichever exploration field `policy` reads.
    ///
    /// The two are separate fields because a UCB1 constant and a sampling
    /// probability are not interchangeable numbers, and this is the one place
    /// that has to choose between them.
    #[inline]
    pub(crate) fn exploration(&self, policy: SimultaneousPolicy) -> f64 {
        match policy {
            SimultaneousPolicy::Duct => self.duct_exploration,
            SimultaneousPolicy::RegretMatching => self.regret_matching_exploration,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Why a search stopped.
pub enum StopReason {
    /// Only one legal choice; no search was run.
    SingleChoice,
    /// The iteration budget was reached, or a `RootParallel` merge over more
    /// than one worker declined to inherit a worker's proof — in which case
    /// `iterations_used` is below the pooled budget.
    Budget,
    /// The wall-clock budget expired.
    Deadline,
    /// The cancellation flag was set.
    Cancelled,
    /// The remaining iterations could not have changed the answer: every rival
    /// is too far from being sampled enough to be trusted for the iterations
    /// left to get it there, and an untrusted candidate never outranks a
    /// trusted one. Nothing weaker is stamped with this — the answer is a mean
    /// reward, and no count bounds a mean, so a root whose rivals are all well
    /// sampled runs to its budget. A search that exhausted its budget reports
    /// `Budget`, and a `RootParallel` merge over more than one worker never
    /// reports this — the proof is about one worker's tree, and the merged
    /// answer is read off statistics that proof never saw.
    Proven,
}

#[derive(Clone, Debug)]
/// What a search decided, and how it got there.
pub struct SearchResult<C> {
    /// The chosen root move.
    ///
    /// Always a choice legal in the `state` the search was called on, whatever
    /// the determinizations offered.
    ///
    /// At a sequential root, the leading root choice: the best mean reward among
    /// the choices the search sampled enough times to trust, with the selection
    /// rate — visits over the iterations that offered the choice — breaking
    /// ties. At a simultaneous root, the *perspective*
    /// player's own action — drawn from their mixed strategy under
    /// [`RootPolicy::Sampled`] — never a joint tuple and never read off a joint
    /// successor.
    pub choice: C,
    /// Iterations run by this call, excluding any inherited from a reused tree.
    pub iterations_used: u32,
    /// Iterations inherited from a retained subtree.
    pub reused_iterations: u32,
    /// Why the search stopped.
    pub stop_reason: StopReason,
    /// Total visits at the root, including reused ones.
    pub root_visits: u32,
    /// Visits to the chosen child, or to the chosen marginal at a simultaneous
    /// root.
    pub best_visits: u32,
    /// Mean reward of the chosen child, for the player who moved there. At a
    /// simultaneous root, of the chosen marginal — whose player is
    /// unambiguously the perspective player.
    pub best_mean_reward: f64,
}

/// One simultaneous level of the descent that actually selected a joint action.
///
/// Backup needs each participant's sampling probability as an importance
/// weight, and that probability cannot be recovered from the tree: it depends on
/// the whole distribution at the moment of selection and on a decaying floor
/// that has moved by the time backup runs. Recomputing it would rebuild an
/// `O(sum |A_i|)` distribution a second time per level and would silently
/// disagree with the value actually sampled.
///
/// The arm *indices* are not recorded — backup reads them out of the parent's
/// `joint_keys`, which is the whole payoff of keying joint edges by marginal
/// index. `path` therefore stays `Vec<u32>`, one word per level, exactly as it
/// is today.
///
/// Frames are pushed in descent order and consumed in backup order, so backup
/// keeps one cursor and asks `frames[cursor].depth == depth` — a compare against
/// an integer, not a field read on the node. For a purely sequential game
/// `frames` is empty and that is one length compare per level.
#[derive(Clone, Copy)]
struct SimFrame {
    /// Index into `path` of the level this describes.
    depth: u32,
    arity: u32,
    /// Start of this frame's `arity` entries in `sim_probs`.
    prob_start: u32,
    /// The node's visit count when the descent stamped its arms, so backup can
    /// prove it credits them before `Node::record` moves it. Debug only; in
    /// release a frame stays 12 bytes.
    #[cfg(debug_assertions)]
    epoch: u32,
}

/// Every reusable allocation one iteration needs.
///
/// Grouped into a struct because `run_iteration` was already at twelve arguments
/// with an `allow(clippy::too_many_arguments)` and would otherwise reach
/// seventeen. The fields split-borrow, so `&mut s.state` and `&*s.picks`
/// coexist — which is what lets `JointChoices` point into the node while the
/// game mutates the state.
struct Scratch<G: Game> {
    /// The single state every iteration determinizes into.
    state: G,
    /// At a simultaneous node this holds **one participant's** list at a time.
    ///
    /// No two participants' lists are ever live at once — enumerate slot `s`,
    /// expand slot `s`'s arms, clear, next — so no flat-buffer-plus-spans
    /// machinery is needed and the existing single buffer suffices.
    /// `JointChoices` borrows the chosen actions from the node's `arm_choices`,
    /// not from here.
    choices: Vec<G::Choice>,
    avail: Vec<bool>,
    root_avail: Vec<bool>,
    /// One child index per level, sequential or joint.
    path: Vec<u32>,
    /// One arm index per participant at the current level. At most eight;
    /// consumed before the descent moves on, so one buffer serves every level,
    /// and it doubles as `JointKey::pack`'s input so there is no separate key
    /// scratch.
    picks: Vec<u32>,
    sim_frames: Vec<SimFrame>,
    /// `arity` sampling probabilities per frame. `f32` is ample: the floor keeps
    /// a probability well inside f32's precision, and it halves the buffer.
    sim_probs: Vec<f32>,
    /// Which of the root's candidates — a sequential root's children, or a
    /// simultaneous root's arms for the perspective player — are legal in the
    /// *real* position. Filled once per search, not per iteration.
    root_legal: Vec<bool>,
    /// Whether every action the real position offers already has a candidate in
    /// the tree, so no future determinization can introduce a new one. The
    /// early-termination proof needs this and the mask together.
    root_legal_complete: bool,
    /// Whether the root has had a full expansion pass.
    ///
    /// A root re-rooted from a subtree has children, but only the handful
    /// progressive expansion happened to create at its old position. "Has
    /// children" is a different question, and answering it that way lets the
    /// search pick a choice that is no longer legal. At a simultaneous root this
    /// means every participant's arms are fully enumerated, not that any joint
    /// successor exists.
    root_fully_expanded: bool,
}

/// Owns every reusable allocation and the retained tree.
///
/// Keep one per player per game and call `search` repeatedly: the scratch state,
/// choice buffer and availability buffers are all reused, so a steady-state
/// iteration allocates nothing beyond growing the tree itself.
pub struct Searcher<G: Game> {
    scratch: Scratch<G>,
    side: G::Side,
    root: Option<Node<G::Choice>>,
    /// Whether the retained tree describes the position about to be searched.
    ///
    /// A search consumes this on entry; only `reuse_subtree` and `reuse_joint`
    /// set it. So carrying a tree forward is something you ask for, and
    /// forgetting to leaves you with a correct search rather than one rooted at
    /// last turn's position.
    tree_is_current: bool,
}

impl<G: Game> Searcher<G> {
    /// `template` is only used to size the scratch state; its contents are
    /// overwritten by the first determinization.
    pub fn new(template: &G) -> Self {
        Self {
            scratch: Scratch {
                state: template.new_buffer(),
                choices: Vec::new(),
                avail: Vec::new(),
                root_avail: Vec::new(),
                path: Vec::new(),
                picks: Vec::new(),
                sim_frames: Vec::new(),
                sim_probs: Vec::new(),
                root_legal: Vec::new(),
                root_legal_complete: false,
                root_fully_expanded: false,
            },
            side: Default::default(),
            root: None,
            tree_is_current: false,
        }
    }

    /// The retained tree, for inspection or display.
    pub fn tree(&self) -> Option<&Node<G::Choice>> {
        self.root.as_ref()
    }

    /// Take the retained tree, leaving the searcher with none.
    ///
    /// For handing a finished tree to something that outlives the searcher —
    /// an inspector or a UI thread. The next search starts fresh.
    pub fn take_tree(&mut self) -> Option<Node<G::Choice>> {
        self.scratch.root_fully_expanded = false;
        self.tree_is_current = false;
        self.root.take()
    }

    /// Discard the retained tree.
    pub fn clear_tree(&mut self) {
        self.root = None;
        self.scratch.root_fully_expanded = false;
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
    ///
    /// A simultaneous root is a caller error rather than a miss, and the two
    /// build profiles report it differently: debug panics on a `debug_assert!`,
    /// release returns false and drops the tree. The successor there is
    /// determined by every participant's action, not by one player's, so use
    /// [`Searcher::reuse_joint`] — and in a game with both kinds of node,
    /// branch on [`Node::simultaneous_players`] over [`Searcher::tree`] rather
    /// than on this return value, which does not survive a debug build.
    pub fn reuse_subtree(&mut self, choice: &G::Choice) -> bool {
        let Some(root) = self.root.as_mut() else {
            return false;
        };
        self.scratch.root_fully_expanded = false;
        debug_assert!(
            root.simultaneous_players().is_none(),
            "mcts: reuse_subtree was called on a simultaneous root, whose successor \
             is determined by every participant's action rather than by one player's. \
             Use Searcher::reuse_joint, branching on Node::simultaneous_players if the \
             game has both kinds of node. A release build drops the tree and returns \
             false here instead of panicking."
        );
        if root.simultaneous_players().is_none() && root.reroot_at(choice) {
            self.tree_is_current = true;
            true
        } else {
            self.root = None;
            self.tree_is_current = false;
            false
        }
    }

    /// Re-root the retained tree at the joint action a simultaneous round
    /// actually resolved to.
    ///
    /// `played` must name every participant exactly once; order does not matter,
    /// and the entries are matched to slots by player index.
    ///
    /// A root that is not simultaneous is a caller error rather than a miss, and
    /// the two build profiles report it differently: debug panics on a
    /// `debug_assert!`, release returns false and drops the tree. Use
    /// [`Searcher::reuse_subtree`] there, branching on
    /// [`Node::simultaneous_players`] over [`Searcher::tree`] rather than on
    /// this return value.
    ///
    /// **Expect this to miss, and often.** A simultaneous node has up to
    /// `prod(|A_i|)` successors and materializes at most one per iteration, so
    /// unless the search concentrated its visits on the tuple that was actually
    /// played, the subtree simply is not there. A miss drops the tree and
    /// returns false, which costs a fresh search rather than a wrong one — the
    /// same price a [`Searcher::reuse_subtree`] miss has always cost.
    ///
    /// The promoted node keeps its own marginal statistics, availability
    /// included. A node's `availability` counts iterations at the parent that
    /// was just discarded, which is why re-rooting resets it; an arm's counts
    /// iterations at *this* node, which are still valid and are still the
    /// denominator of its exploration term. Zeroing them would leave a
    /// carried-forward arm with hundreds of visits and no exploration bonus,
    /// frozen at whatever its mean happened to be.
    pub fn reuse_joint(&mut self, played: &[(u8, G::Choice)]) -> bool {
        let Some(root) = self.root.as_mut() else {
            return false;
        };
        self.scratch.root_fully_expanded = false;
        debug_assert!(
            root.simul().is_some(),
            "mcts: reuse_joint was called on a root that is not simultaneous, whose \
             successor is determined by one player's action. Use \
             Searcher::reuse_subtree, branching on Node::simultaneous_players if the \
             game has both kinds of node. A release build drops the tree and returns \
             false here instead of panicking."
        );

        let key = root.simul().and_then(|simul| {
            let players = simul.players;
            assert_eq!(
                played.len(),
                players.len(),
                "mcts: reuse_joint was given {} actions for {} participants",
                played.len(),
                players.len()
            );
            let picks = &mut self.scratch.picks;
            picks.clear();
            picks.resize(players.len(), 0);

            let mut seen: u64 = 0;
            let mut hit = true;
            for (player, choice) in played {
                assert!(
                    players.contains(*player),
                    "mcts: reuse_joint names player {player}, who does not act at this node"
                );
                let slot = players
                    .slot_of(*player)
                    .expect("membership was just asserted");
                assert!(
                    seen & (1 << slot) == 0,
                    "mcts: reuse_joint names player {player} twice"
                );
                seen |= 1 << slot;
                match simul.find_arm(slot, choice) {
                    Some(arm) => picks[slot] = (arm - simul.starts[slot] as usize) as u32,
                    None => hit = false,
                }
            }
            hit.then(|| JointKey::pack(picks, simul.key_bits))
        });

        if key.is_some_and(|key| root.reroot_at_joint(key)) {
            self.tree_is_current = true;
            true
        } else {
            self.root = None;
            self.tree_is_current = false;
            false
        }
    }

    /// Write `player`'s mixed strategy at a simultaneous root into `out`, as
    /// `(choice, probability)` pairs summing to 1 over the actions legal in
    /// `state`.
    ///
    /// Returns false if there is no retained tree, if the root is not
    /// simultaneous, if `player` does not act there, if `player` has no legal
    /// action in `state`, or if nothing has yet enumerated `player`'s actions at
    /// that root — a search that completed no iterations, or a forced move,
    /// leaves a root that names its participants and holds no arms. `out` is
    /// cleared first and is a caller buffer, so repeated calls do not grow it.
    ///
    /// On `true` the pairs are never empty: where the tree's arms and this
    /// position's legal actions are disjoint, they are a uniform distribution
    /// over the latter, which is the same fallback [`Searcher::search`] draws
    /// its answer from.
    ///
    /// Poll this once a turn, not once an iteration: renormalizing needs this
    /// position's own legal set, and `&self` means enumerating it into a
    /// temporary rather than into the search's scratch.
    ///
    /// This is [`crate::Marginals::policy_into`] renormalized against the real
    /// position rather than against the union of every determinization, which is
    /// the only sound extraction when a player's legal set varies between
    /// worlds.
    pub fn root_policy_into(
        &self,
        state: &G,
        ctx: &G::Context,
        player: u8,
        out: &mut Vec<(G::Choice, f64)>,
    ) -> bool {
        out.clear();
        let Some(simul) = self.root.as_ref().and_then(|root| root.simul()) else {
            return false;
        };
        let Some(slot) = simul.players.slot_of(player) else {
            return false;
        };
        // A root can name its participants and hold no arms for them: every
        // zero-iteration exit installs the block so `simultaneous_players` is
        // truthful, and only an iteration expands it. There is no strategy to
        // report from one, and reporting `true` with an empty `out` would break
        // the sum-to-1 promise above.
        if simul.slot_len(slot) == 0 {
            return false;
        }

        let mut choices = Vec::new();
        state.choices_for_into(ctx, player, &mut choices);
        if choices.is_empty() {
            return false;
        }
        let start = simul.starts[slot] as usize;
        let mut legal = vec![false; simul.slot_len(slot)];
        for choice in &choices {
            if let Some(arm) = simul.find_arm(slot, choice) {
                legal[arm - start] = true;
            }
        }

        let mut weights = Vec::new();
        duct::root_strategy_into(simul, slot, &legal, G::SIMULTANEOUS_POLICY, &mut weights);
        for (arm, &weight) in weights.iter().enumerate() {
            if legal[arm] {
                out.push((simul.arm_choices[start + arm].clone(), weight));
            }
        }
        if out.is_empty() {
            // Every arm the tree holds for this slot is unplayable here, so the
            // search has nothing to say about the position — which is a real
            // state at low budgets, not an error. `Searcher::search` answers it
            // with a uniform draw over this player's own list; answering with
            // the distribution that draw comes from keeps the two extractions
            // the same thing, and keeps the sum-to-1 promise above.
            let share = 1.0 / choices.len() as f64;
            out.extend(choices.into_iter().map(|choice| (choice, share)));
        }
        true
    }

    /// Search from `state` on behalf of `perspective` and return the most
    /// visited root choice.
    ///
    /// Panics if `state` is terminal, has no legal choices, or if the config
    /// specifies neither an iteration nor a time budget.
    ///
    /// [`Game::advance`] is called on each determinization of `state` and must
    /// not fast-forward past the decision being searched — the answer list and
    /// the root player are read from `state` itself, and the tree is built on
    /// the advanced state. Debug builds assert both on every iteration, which
    /// leaves one gap: a root offering a single choice returns it before the
    /// first determinization, so a game that breaks the contract is not caught
    /// there. The answer is still that root's own, having been read from
    /// `state` before any `advance` ran.
    ///
    /// At a simultaneous root, `result.choice` is the perspective player's own
    /// action, drawn from their mixed strategy under [`RootPolicy::Sampled`],
    /// and this call panics if `perspective` is not one of the acting players.
    /// [`Config::early_termination`] has no effect at a simultaneous root under
    /// [`crate::SimultaneousPolicy::RegretMatching`] and [`StopReason::Proven`]
    /// cannot occur there: the proof is about the leading candidate, and for a
    /// sampled answer proving that candidate cannot be overtaken proves nothing
    /// about what comes back.
    pub fn search<R: Rng + ?Sized>(
        &mut self,
        state: &G,
        ctx: &G::Context,
        perspective: u8,
        cfg: &Config,
        cancel: Option<&AtomicBool>,
        rng: &mut R,
    ) -> SearchResult<G::Choice> {
        // Consumed here rather than cleared at the end, so an exit that never
        // reaches the end — the forced-move return, a preamble panic, a panic
        // out of the game — cannot leave the previous position's tree armed.
        let reuse = core::mem::take(&mut self.tree_is_current);

        assert!(
            cfg.iterations != 0 || cfg.time_limit_ms.is_some(),
            "mcts: Config has neither an iteration nor a time budget"
        );

        let (root_player, root_players) = match state.status(ctx) {
            Status::Active { player } => (player, None),
            Status::Simultaneous { players } => {
                assert!(
                    !players.is_empty(),
                    "mcts: Status::Simultaneous with no players"
                );
                assert!(
                    players.len() <= PlayerSet::MAX_SIMULTANEOUS,
                    "mcts: Status::Simultaneous names {} players; at most {} may act at once",
                    players.len(),
                    PlayerSet::MAX_SIMULTANEOUS
                );
                assert!(
                    players.contains(perspective),
                    "mcts: search called on behalf of player {perspective}, who is not one of \
                     the players acting at this simultaneous state"
                );
                (perspective, Some(players))
            }
            Status::Terminal(_) => panic!("mcts: search called on a terminal state"),
        };

        // Anything retained from an earlier search describes an earlier
        // position unless `reuse_subtree` re-rooted it since. Dropped above the
        // forced-move return, so that return cannot leave one behind either;
        // below the asserts above, so a rejected call keeps the caller's tree.
        if !reuse {
            self.root = None;
        }

        self.scratch.choices.clear();
        match root_players {
            None => state.choices_into(ctx, &mut self.scratch.choices),
            Some(_) => state.choices_for_into(ctx, perspective, &mut self.scratch.choices),
        }
        match self.scratch.choices.len() {
            0 => panic!(
                "mcts: search called on a state where player {perspective} has no legal choices"
            ),
            // With one action, nothing the opponents do changes which action you
            // take, so this is sound at a simultaneous root too. It returns
            // before the first determinization, so `RootCheck` never sees this
            // exit and an `advance` that consumes the root decision goes
            // unreported here; `search`'s doc says so.
            1 => {
                // This return builds no root, so a retained one has to describe
                // the position by itself. It does describe it — `self.root` is
                // Some here only because the caller asked to reuse it — but a
                // node promoted out of a joint child that was never descended
                // into carries no marginals, and `reuse_joint` reads those.
                if let (Some(players), Some(root)) = (root_players, self.root.as_mut()) {
                    root.ensure_simul(players);
                }
                return SearchResult {
                    choice: self.scratch.choices[0].clone(),
                    iterations_used: 0,
                    reused_iterations: 0,
                    stop_reason: StopReason::SingleChoice,
                    root_visits: 0,
                    best_visits: 0,
                    best_mean_reward: 0.0,
                };
            }
            _ => {}
        }

        if self.root.is_none() {
            self.root = Some(Node::new_root(root_player));
            self.scratch.root_fully_expanded = false;
        }
        state.init_side(ctx, &mut self.side);

        let Self {
            scratch,
            side,
            root,
            ..
        } = self;
        let root = root.as_mut().expect("root was just created");

        // A re-rooted root carries the *previous* mover's index, so what its
        // cumulative reward means would otherwise depend on whether the tree was
        // reused. Both cases now accumulate the player to act.
        root.player = root_player;
        root.cumulative_reward = 0.0;

        // Installed here rather than on the first iteration, so a search that
        // completes none — cancelled, or already past its deadline — still
        // leaves a root that answers `Node::simultaneous_players` truthfully
        // and sends `reuse_joint` down its ordinary miss path. The forced-move
        // return above is the third such exit and does its own.
        if let Some(players) = root_players {
            root.ensure_simul(players);
        }

        // The mask outlives both the search and the tree, and
        // `refresh_root_legal` caches it on candidate count alone: a mask left
        // by an earlier position with the same number of candidates would pass
        // that test.
        scratch.root_legal.clear();
        scratch.root_legal_complete = false;
        let target = cfg.iterations;
        let deadline = Deadline::new(cfg.time_limit_ms);
        let reused_iterations = root.visits;
        let mut iterations_used: u32 = 0;
        #[cfg(debug_assertions)]
        let mut root_check = RootCheck::<G::Choice>::new();

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
            state.determinize_into(&mut scratch.state, ctx, perspective, rng);
            #[cfg(debug_assertions)]
            let before = root_check.read::<G>(&scratch.state, ctx, root_players, perspective);
            scratch.state.advance(ctx, side, perspective, rng);
            #[cfg(debug_assertions)]
            root_check.check::<G>(&scratch.state, ctx, root_players, perspective, before);

            let rewards = run_iteration::<G, R>(root, scratch, ctx, side, cfg, perspective, rng);

            G::credit_iteration(side, &rewards);
            iterations_used += 1;

            if cfg.early_termination && target != 0 {
                refresh_root_legal(root, state, ctx, scratch, root_players, perspective);
                if early_stop::settled::<G>(
                    root,
                    root_players,
                    perspective,
                    target,
                    &scratch.root_legal,
                    scratch.root_legal_complete,
                ) {
                    break StopReason::Proven;
                }
            }
        };

        // The tree holds every candidate any determinization offered, and the
        // answer has to be one this position actually holds. One mask, built
        // once: both extractions and the proof above rank the same set, which is
        // the whole point of the filter.
        refresh_root_legal(root, state, ctx, scratch, root_players, perspective);
        let (choice, best_visits, best_mean_reward) = match root_players {
            None => match root_answer(root, &scratch.root_legal) {
                Some(i) => (
                    root.children[i]
                        .edge()
                        .choice()
                        .expect("a sequential root's children all carry a choice")
                        .clone(),
                    root.children[i].visits(),
                    root.children[i].mean_reward(),
                ),
                // Nothing legal here was ever visited: cancelled or timed out
                // before a single iteration completed, or every world the
                // search saw offered a different set of moves.
                None => {
                    scratch.choices.clear();
                    state.choices_into(ctx, &mut scratch.choices);
                    let k = below(rng, scratch.choices.len() as u64) as usize;
                    (scratch.choices[k].clone(), 0, 0.0)
                }
            },
            Some(_) => extract_marginal::<G, R>(root, state, ctx, scratch, perspective, cfg, rng),
        };

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

/// The root's answer: the highest-ranked child that is legal in the real
/// position, under the crate's one root ranking ([`crate::rank`]). First
/// maximum, so ties break deterministically. `None` if no legal child was ever
/// visited.
///
/// `legal` is parallel to `root.children`; a mask of the wrong length describes
/// some other candidate set and admits nothing.
fn root_answer<C>(root: &Node<C>, legal: &[bool]) -> Option<usize> {
    if legal.len() != root.children().len() {
        return None;
    }
    leader_of(
        root.children()
            .iter()
            .enumerate()
            .filter(|(i, _)| legal[*i])
            .map(|(i, child)| {
                (
                    i,
                    Candidate::new(child.visits, child.availability, child.cumulative_reward),
                )
            }),
    )
    .map(|(i, _)| i)
}

/// A root decision, as much of one as `Status` carries.
#[cfg(debug_assertions)]
#[derive(PartialEq, Eq, Debug)]
enum RootDecision {
    Active(u8),
    Simultaneous(PlayerSet),
    Terminal,
}

#[cfg(debug_assertions)]
fn root_decision<G: Game>(state: &G, ctx: &G::Context) -> RootDecision {
    match state.status(ctx) {
        Status::Active { player } => RootDecision::Active(player),
        Status::Simultaneous { players } => RootDecision::Simultaneous(players),
        Status::Terminal(_) => RootDecision::Terminal,
    }
}

#[cfg(debug_assertions)]
fn root_choices_into<G: Game>(
    state: &G,
    ctx: &G::Context,
    root_players: Option<PlayerSet>,
    perspective: u8,
    out: &mut Vec<G::Choice>,
) {
    out.clear();
    match root_players {
        None => state.choices_into(ctx, out),
        Some(_) => state.choices_for_into(ctx, perspective, out),
    }
}

/// The answer lists either side of the root `advance`.
///
/// Held across the whole search and cleared rather than rebuilt: the lists are
/// re-read on every iteration, and a check that allocated per iteration would
/// cost more than the enumeration it exists to police — and would break the
/// saturated-search allocation guarantee in every debug build.
#[cfg(debug_assertions)]
struct RootCheck<C> {
    before: Vec<C>,
    after: Vec<C>,
    /// Positions in `before`, built only when the lists disagree on order. An
    /// unmoved decision settles on the elementwise compare, so a conforming
    /// game never allocates this.
    index: HashTable<u32>,
}

#[cfg(debug_assertions)]
impl<C: Eq + Hash> RootCheck<C> {
    fn new() -> Self {
        RootCheck {
            before: Vec::new(),
            after: Vec::new(),
            index: HashTable::new(),
        }
    }

    /// The root decision this determinization presents, and its answer list.
    fn read<G: Game<Choice = C>>(
        &mut self,
        state: &G,
        ctx: &G::Context,
        root_players: Option<PlayerSet>,
        perspective: u8,
    ) -> RootDecision {
        let decision = root_decision(state, ctx);
        self.before.clear();
        if decision == RootDecision::Terminal {
            return decision;
        }
        root_choices_into(state, ctx, root_players, perspective, &mut self.before);
        decision
    }

    /// The root `advance` may resolve decisions the tree does not model, but
    /// not the one being searched: the answer list and the root player were
    /// read from the state the caller handed in, and everything after this
    /// point is built on the advanced one.
    ///
    /// Read across `advance` on one determinized state rather than against the
    /// caller's, so a game whose determinizations legitimately differ from the
    /// real position — up to and including one that samples a world already
    /// over — is not accused of an `advance` it may never have written.
    ///
    /// Every iteration, not just the first: an `advance` that consumes the root
    /// decision in some worlds and not others is exactly what a side model
    /// resolving an opponent's hidden decision produces, and checking one
    /// determinization would clear it whenever the first world sampled happens
    /// to behave.
    fn check<G: Game<Choice = C>>(
        &mut self,
        state: &G,
        ctx: &G::Context,
        root_players: Option<PlayerSet>,
        perspective: u8,
        before: RootDecision,
    ) {
        let after = root_decision(state, ctx);
        assert!(
            after == before,
            "mcts: Game::advance moved the root past the decision the search was called on. \
             The determinization presented {:?} and the state advance left behind presents \
             {after:?}, so the search would read one position's legal moves and build the tree \
             on another's. An advance at the root may fast-forward past decisions the tree does \
             not model, never past the one being searched.",
            before
        );
        if after == RootDecision::Terminal {
            return;
        }

        self.after.clear();
        root_choices_into(state, ctx, root_players, perspective, &mut self.after);
        // `choices_into` promises no order, but a game that enumerates an
        // unmoved decision the same way twice settles here without hashing or a
        // scan. A merely permuted list is legal and falls through.
        if self.after == self.before {
            return;
        }
        let RootCheck {
            before,
            after,
            index,
        } = self;
        // A wide root that merely permuted its list would pay a scan per choice
        // here on every iteration, which is the quadratic shape the crate's own
        // `CHILD_INDEX_THRESHOLD` exists to avoid.
        let same_set = if before.len() > G::CHILD_INDEX_THRESHOLD {
            index.clear();
            for (i, choice) in before.iter().enumerate() {
                index.insert_unique(hash_of(choice), i as u32, |&j| hash_of(&before[j as usize]));
            }
            after.iter().all(|c| {
                index
                    .find(hash_of(c), |&j| before[j as usize] == *c)
                    .is_some()
            })
        } else {
            after.iter().all(|c| before.contains(c))
        };
        assert!(
            after.len() == before.len() && same_set,
            "mcts: Game::advance moved the root past the decision the search was called on. \
             The same player is still to act, but on a different decision: the determinization \
             offered {} choices and the state advance left behind offers {}, and they are not \
             the same set. The root's children were enumerated before this call.",
            before.len(),
            after.len()
        );
    }
}

/// Keep `scratch.root_legal` covering the root's candidates — a sequential
/// root's children, or the perspective player's arms at a simultaneous one — so
/// that both the returned move and the early-termination proof are restricted to
/// what the player actually holds rather than to every action any
/// determinization has offered. The mask is built against the **real** position
/// rather than the union of the determinizations that built the tree, because a
/// candidate the search learned about in some other world is not playable in
/// this one.
///
/// Candidates are only ever appended *within one search*, so a length match
/// means the candidate set has not moved and the mask still describes it. Move
/// generation therefore runs once per candidate created over the whole search,
/// not once per iteration. Across searches the length carries no such guarantee
/// — a new position can offer the same number of candidates as the last one and
/// a different set — which is why `search` clears the mask before its first
/// iteration.
fn refresh_root_legal<G: Game>(
    root: &Node<G::Choice>,
    state: &G,
    ctx: &G::Context,
    scratch: &mut Scratch<G>,
    root_players: Option<PlayerSet>,
    perspective: u8,
) {
    // Which kind of root this is comes from the position, as every other such
    // decision in `search` does, and never from the tree: a retained tree can
    // carry a simultaneous block into a sequential position, and a mask built
    // for that block's arms would be read by the extraction and the proof as a
    // mask over children. `run_iteration` asserts the two agree; here the
    // consequence of trusting the wrong one is a mask that describes some other
    // candidate set entirely.
    let slot = match root_players {
        None => {
            debug_assert!(
                root.simul().is_none(),
                "mcts: this position is sequential but the root's statistics were built \
                 as a simultaneous node"
            );
            None
        }
        Some(players) => Some(
            players
                .slot_of(perspective)
                .expect("the preamble asserted membership"),
        ),
    };
    let arms = slot.map(|slot| {
        let simul = root
            .simul()
            .expect("`search` installs the block before the first iteration");
        (simul, slot)
    });
    let candidates = match arms {
        Some((simul, slot)) => simul.slot_len(slot),
        None => root.children().len(),
    };
    if candidates != 0 && scratch.root_legal.len() == candidates {
        return;
    }

    scratch.choices.clear();
    match arms {
        Some(_) => state.choices_for_into(ctx, perspective, &mut scratch.choices),
        None => state.choices_into(ctx, &mut scratch.choices),
    }
    scratch.root_legal.clear();
    scratch.root_legal.resize(candidates, false);
    let mut found = 0;
    match arms {
        Some((simul, slot)) => {
            let start = simul.starts[slot] as usize;
            for choice in scratch.choices.iter() {
                if let Some(arm) = simul.find_arm(slot, choice) {
                    scratch.root_legal[arm - start] = true;
                    found += 1;
                }
            }
        }
        None => {
            for choice in scratch.choices.iter() {
                if let Some(child) = root.find_child(choice) {
                    scratch.root_legal[child] = true;
                    found += 1;
                }
            }
        }
    }
    // Every choice contributes at most one to `found`, so this is exactly "no
    // choice went unmatched". A duplicate in the enumeration matches twice and
    // is counted twice, which is why the test is against the enumeration's own
    // length rather than against how many candidates the mask marks:
    // `choices_into` promises nothing about distinctness.
    scratch.root_legal_complete = found == scratch.choices.len();
}

/// The perspective player's own action at a simultaneous root, with that
/// marginal's statistics.
///
/// Reads the mask `search` has just refreshed rather than building its own, so
/// the answer is drawn from exactly the candidates the early-termination proof
/// ranked. Two constructions of the same object is how the sequential root came
/// to be extracted unfiltered while this one was not. A mask of the wrong length
/// describes some other candidate set and admits nothing.
///
/// Never an argmax over joint successors under either policy: the perspective
/// player's action appears in one joint child per opponent action, so an argmax
/// there picks the best *pair* — the classic optimistic bug that makes a
/// decoupled agent assume the opponent plays along.
fn extract_marginal<G: Game, R: Rng + ?Sized>(
    root: &Node<G::Choice>,
    state: &G,
    ctx: &G::Context,
    scratch: &mut Scratch<G>,
    perspective: u8,
    cfg: &Config,
    rng: &mut R,
) -> (G::Choice, u32, f64) {
    let picked = root.simul().and_then(|simul| {
        let slot = simul
            .players
            .slot_of(perspective)
            .expect("the preamble asserted membership");
        let legal = &scratch.root_legal;
        if legal.len() != simul.slot_len(slot) {
            return None;
        }
        match cfg.simultaneous.root_policy {
            RootPolicy::Sampled => {
                duct::sample_root_arm(simul, slot, legal, G::SIMULTANEOUS_POLICY, rng)
            }
            RootPolicy::MostVisited => duct::best_arm(simul, slot, legal, G::SIMULTANEOUS_POLICY),
        }
        .map(|arm| {
            let marginals = simul.marginals(slot);
            (
                marginals.choice(arm).clone(),
                marginals.visits(arm),
                marginals.mean_reward(arm),
            )
        })
    });

    picked.unwrap_or_else(|| {
        // No arm legal here, or no iteration completed. Uniform over this
        // player's own list — `choices_into` may name another player's actions
        // at an asymmetric simultaneous state, and a cached mask leaves
        // `scratch.choices` describing whatever last filled it.
        scratch.choices.clear();
        state.choices_for_into(ctx, perspective, &mut scratch.choices);
        let k = below(rng, scratch.choices.len() as u64) as usize;
        (scratch.choices[k].clone(), 0, 0.0)
    })
}

/// One selection-expansion-simulation-backpropagation pass.
///
/// The descent records the child index taken at each level rather than
/// recursing, so tree depth costs a `u32` per level instead of a stack frame and
/// a deep game cannot overflow the stack. Backpropagation walks the same path
/// back down from the root.
fn run_iteration<G: Game, R: Rng + ?Sized>(
    root: &mut Node<G::Choice>,
    scratch: &mut Scratch<G>,
    ctx: &G::Context,
    side: &mut G::Side,
    cfg: &Config,
    perspective: u8,
    rng: &mut R,
) -> G::Rewards {
    let Scratch {
        state,
        choices,
        avail,
        root_avail,
        path,
        picks,
        sim_frames,
        sim_probs,
        root_fully_expanded,
        ..
    } = scratch;
    path.clear();
    sim_frames.clear();
    sim_probs.clear();

    let rewards: G::Rewards;
    let mut node = &mut *root;

    loop {
        let player = match state.status(ctx) {
            Status::Terminal(terminal_rewards) => {
                rewards = terminal_rewards;
                break;
            }
            Status::Active { player } => {
                debug_assert!(
                    node.simultaneous_players().is_none(),
                    "mcts: this node reported Status::Active {{ player: {player} }} but its \
                     statistics were built as a simultaneous node. A node's kind must not \
                     vary across determinizations; the search will read marginal bandits \
                     as children."
                );
                player
            }
            Status::Simultaneous { players } => {
                debug_assert!(
                    node.simultaneous_players().is_none()
                        || node.simultaneous_players() == Some(players),
                    "mcts: this node reported Status::Simultaneous with players {:?} but its \
                     statistics were built for {:?}. A node's kind and participant set must \
                     not vary across determinizations — determinization may change which \
                     actions are legal, never who acts. The search will read one player's \
                     arms as another's and return a move belonging to the wrong player.",
                    players,
                    node.simultaneous_players().unwrap_or(PlayerSet::EMPTY)
                );

                let arity = players.len();
                let epoch = node.visits;
                let policy = G::SIMULTANEOUS_POLICY;
                node.ensure_simul(players);

                // Marginal expansion is full on every visit, root or not; only
                // the joint successors are lazy. See `Node::expand_marginals`.
                let mut degenerate = false;
                if G::ROOT_CHOICES_INVARIANT && node.is_root() && *root_fully_expanded {
                    #[cfg(debug_assertions)]
                    {
                        for slot in 0..arity {
                            let p = players.player_at(slot).expect("a slot has a participant");
                            choices.clear();
                            state.choices_for_into(ctx, p, choices);
                            let simul = node.simul().expect("the block was just installed");
                            assert_eq!(
                                choices.len(),
                                simul.slot_len(slot),
                                "ROOT_CHOICES_INVARIANT is set, but this determinization \
                                 offers player {p} {} choices where the tree holds {} marginals",
                                choices.len(),
                                simul.slot_len(slot)
                            );
                            for choice in choices.iter() {
                                assert!(
                                    simul.find_arm(slot, choice).is_some(),
                                    "ROOT_CHOICES_INVARIANT is set, but this determinization \
                                     offers player {p} a choice the tree has no marginal for"
                                );
                            }
                        }
                    }
                    node.restamp_marginals();
                } else {
                    // Only a pass that found every arm legal has *proved* the
                    // tree's action set equals this position's. A root promoted
                    // by `reuse_joint` carries arms from determinizations
                    // reached at depth, and the fast path would restamp those
                    // as available.
                    let mut every_arm_legal = true;
                    for slot in 0..arity {
                        let p = players.player_at(slot).expect("a slot has a participant");
                        choices.clear();
                        state.choices_for_into(ctx, p, choices);
                        let legal = node.expand_marginals(
                            slot,
                            choices,
                            policy.mixes(),
                            G::CHILD_INDEX_THRESHOLD,
                        );
                        debug_assert!(
                            legal != 0,
                            "mcts: player {p} has no legal action at a simultaneous node, so \
                             no joint action exists. The iteration scores zero, which will \
                             drag this node's parent's mean down indistinguishably from a \
                             bad evaluation."
                        );
                        if legal == 0 {
                            degenerate = true;
                            break;
                        }
                        let held = node
                            .simul()
                            .expect("the block was just installed")
                            .slot_len(slot);
                        every_arm_legal &= legal == held;
                    }
                    if !degenerate && every_arm_legal && node.is_root() {
                        *root_fully_expanded = true;
                    }
                }
                if degenerate {
                    rewards = G::Rewards::zero();
                    break;
                }

                // The `k` picks are made independently; that is the definition
                // of decoupling, and the tuple they form need not already exist.
                let span = cfg.max_reward - cfg.min_reward;
                let prob_start = sim_probs.len() as u32;
                picks.clear();
                let simul = node.simul_mut().expect("the block was just installed");
                for slot in 0..arity {
                    let Some((pick, prob)) = duct::select_marginal(
                        simul,
                        slot,
                        epoch,
                        policy,
                        cfg.simultaneous.exploration(policy),
                        span,
                        rng,
                    ) else {
                        degenerate = true;
                        break;
                    };
                    picks.push(pick);
                    sim_probs.push(prob as f32);
                }
                if degenerate {
                    sim_probs.truncate(prob_start as usize);
                    rewards = G::Rewards::zero();
                    break;
                }

                let key = JointKey::pack(picks, simul.key_bits);
                let i = match simul.find_joint(key) {
                    Some(i) => i,
                    None => node.push_joint(key, perspective, G::CHILD_INDEX_THRESHOLD),
                };

                path.push(i as u32);
                sim_frames.push(SimFrame {
                    depth: (path.len() - 1) as u32,
                    arity: arity as u32,
                    prob_start,
                    #[cfg(debug_assertions)]
                    epoch,
                });

                let simul = node.simul().expect("the block was just installed");
                let joint = JointChoices::new(players, &simul.arm_choices, &simul.starts, picks);
                state.apply_joint(ctx, joint, rng);
                state.advance(ctx, side, perspective, rng);

                // The freshness test is on the joint child, never on an arm:
                // a never-selected arm can lead to an already-visited joint
                // child reached through a different partner action, and a joint
                // child can be brand new while every arm in it is well visited.
                // The joint child is the node whose value is being estimated.
                if node.children[i].visits() == 0 {
                    rewards = state.rollout(ctx, rng);
                    break;
                }

                node = &mut node.children[i];
                continue;
            }
        };

        // An already-expanded root whose choice set cannot vary needs neither
        // enumeration nor expansion, which is the difference between paying for
        // move generation once per search and once per iteration.
        let reuse_root = G::ROOT_CHOICES_INVARIANT && node.is_root() && *root_fully_expanded;
        let available: &[bool] = if reuse_root {
            // Skipping enumeration is only sound if the choice set really is
            // invariant. Check it where checking is affordable, so a game that
            // sets the flag wrongly fails loudly instead of quietly returning a
            // choice that is illegal in this determinization.
            #[cfg(debug_assertions)]
            {
                choices.clear();
                state.choices_into(ctx, choices);
                assert_eq!(
                    choices.len(),
                    node.children.len(),
                    "ROOT_CHOICES_INVARIANT is set, but this determinization has {} \
                     root choices where the tree holds {} children",
                    choices.len(),
                    node.children.len()
                );
                for choice in choices.iter() {
                    assert!(
                        node.find_child(choice).is_some(),
                        "ROOT_CHOICES_INVARIANT is set, but this determinization offers \
                         a root choice the tree has no child for"
                    );
                }
            }
            // The invariant says every root child is legal in every
            // determinization, so this iteration is an opportunity for all of
            // them. Skipping the count here would leave availability frozen at
            // whatever the first pass wrote while visits went on accumulating,
            // and the root ranking divides one by the other. The cache moves
            // with the count it caches: `select` reads the root's own visit
            // count rather than a root child's, so nothing would see the two
            // disagree today, and a stale cache waiting for the first caller
            // that does is not worth the `ln` it saves.
            for child in node.children.iter_mut() {
                child.availability += 1;
                child.ln_availability = (child.availability as f64).ln();
            }
            root_avail.clear();
            root_avail.resize(node.children.len(), true);
            root_avail
        } else {
            choices.clear();
            state.choices_into(ctx, choices);
            node.expand(choices, player, G::CHILD_INDEX_THRESHOLD, avail, rng);
            // Same proof obligation as the simultaneous branch: the fast path
            // marks every child available, so it may only be armed once a pass
            // has shown the tree holds this position's choices and nothing else.
            // A root that was never re-rooted has exactly them after the first
            // pass, so it still arms on iteration 1.
            if node.is_root()
                && choices.len() == node.children.len()
                && avail.iter().all(|&legal| legal)
            {
                *root_fully_expanded = true;
            }
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
        // The field rather than `Node::edge`: this runs once per level per
        // iteration, and the kind is already known here — `select` only ever
        // returns a child that `expand` created from a choice.
        let choice = node.children[i]
            .choice
            .as_ref()
            .expect("child has a choice");
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

    let mut backup = Backup::<G> {
        path,
        frames: sim_frames,
        probs: sim_probs,
        rewards: &rewards,
        cfg,
        cursor: 0,
    };
    let mut node = &mut *root;
    backup.credit(node, 0);
    node.record(rewards.reward(node.player));
    for (depth, &i) in path.iter().enumerate() {
        node = &mut node.children[i as usize];
        backup.credit(node, depth + 1);
        node.record(rewards.reward(node.player));
    }

    rewards
}

/// The descent's simultaneous levels, walked in step with the backup.
///
/// A struct rather than eight more arguments, and a cursor rather than a lookup:
/// frames were pushed in descent order and are consumed in backup order, so
/// matching one costs a length compare and an integer compare per level. A
/// purely sequential game has no frames and pays only the length compare.
struct Backup<'a, G: Game> {
    path: &'a [u32],
    frames: &'a [SimFrame],
    probs: &'a [f32],
    rewards: &'a G::Rewards,
    cfg: &'a Config,
    cursor: usize,
}

impl<G: Game> Backup<'_, G> {
    /// Credit this level's arms, if it selected a joint action.
    ///
    /// This must run **before** the node's own `record`. The legality test is
    /// `avail_epoch == node.visits`, so once the record bumps that count every
    /// arm reads as illegal, the regret subtraction covers nothing, and the
    /// sleeping-bandit correction that keeps a rarely-legal action playable is
    /// gone with no symptom other than a worse move.
    #[inline]
    fn credit(&mut self, node: &mut Node<G::Choice>, depth: usize) {
        let Some(frame) = self.frames.get(self.cursor).copied() else {
            return;
        };
        if frame.depth as usize != depth {
            return;
        }
        self.cursor += 1;

        let epoch = node.visits;
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            epoch, frame.epoch,
            "mcts: a simultaneous node's arms must be credited before Node::record bumps \
             its visit count — the legality test is `avail_epoch == node.visits`, and \
             after the record every arm reads as illegal, which silently disables the \
             regret correction that keeps a rarely-legal action playable."
        );

        let child = self.path[depth] as usize;
        let start = frame.prob_start as usize;
        let probs = &self.probs[start..start + frame.arity as usize];
        let simul = node
            .simul_mut()
            .expect("mcts: a joint edge was taken from a node with no marginals");
        let key = simul.joint_keys[child];
        duct::credit_marginals(
            simul,
            key,
            epoch,
            probs,
            self.rewards,
            G::SIMULTANEOUS_POLICY,
            self.cfg.min_reward,
            self.cfg.max_reward - self.cfg.min_reward,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{PlayerSet, SimultaneousPolicy};
    use crate::util::below;
    use rand_core::SeedableRng;
    use wyrand::WyRand;

    /// Rock-paper-scissors on `[0, 1]`, indexed `[player 0][player 1]`. Zero-sum
    /// and symmetric, so the unique equilibrium is uniform for both players.
    const RPS: [[f64; 3]; 3] = [[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]];

    fn rng(seed: u64) -> WyRand {
        WyRand::seed_from_u64(seed)
    }

    fn config(iterations: u32) -> Config {
        Config {
            iterations,
            early_termination: false,
            ..Config::default()
        }
    }

    /// One simultaneous ply and nothing else. The two consts pick the policy and
    /// the root fast path, because both are associated consts and each
    /// combination therefore needs its own type.
    #[derive(Clone, Default)]
    struct Rps<const MIXED: bool, const INVARIANT: bool> {
        payoff: Option<f64>,
    }

    impl<const MIXED: bool, const INVARIANT: bool> Game for Rps<MIXED, INVARIANT> {
        type Choice = u8;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        const ROOT_CHOICES_INVARIANT: bool = INVARIANT;
        const SIMULTANEOUS_POLICY: SimultaneousPolicy = if MIXED {
            SimultaneousPolicy::RegretMatching
        } else {
            SimultaneousPolicy::Duct
        };

        fn status(&self, _ctx: &()) -> Status<[f64; 2]> {
            match self.payoff {
                Some(payoff) => Status::Terminal([payoff, 1.0 - payoff]),
                None => Status::Simultaneous {
                    players: PlayerSet::first_n(2),
                },
            }
        }

        fn choices_into(&self, _ctx: &(), out: &mut Vec<u8>) {
            out.extend([0, 1, 2]);
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _ctx: &(), _choice: &u8, _rng: &mut R) {
            unreachable!("rock-paper-scissors has no sequential node")
        }

        fn apply_joint<R: Rng + ?Sized>(
            &mut self,
            _ctx: &(),
            joint: JointChoices<'_, u8>,
            _rng: &mut R,
        ) {
            self.payoff = Some(RPS[*joint.get(0) as usize][*joint.get(1) as usize]);
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _ctx: &(), _rng: &mut R) -> [f64; 2] {
            let payoff = self.payoff.expect("the joint action has been resolved");
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

    /// A sequential root over a simultaneous node, so the descent, the frame
    /// cursor and the backup all have to handle a mixed tree. Branch 0 leads to
    /// rock-paper-scissors, worth 0.5 to player 0 in equilibrium; branch 1 pays
    /// 0.25 outright.
    #[derive(Clone, Default)]
    struct SeqThenSimul {
        stage: u8,
        payoff: f64,
    }

    impl Game for SeqThenSimul {
        type Choice = u8;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        fn status(&self, _ctx: &()) -> Status<[f64; 2]> {
            match self.stage {
                0 => Status::Active { player: 0 },
                1 => Status::Simultaneous {
                    players: PlayerSet::first_n(2),
                },
                _ => Status::Terminal([self.payoff, 1.0 - self.payoff]),
            }
        }

        fn choices_into(&self, _ctx: &(), out: &mut Vec<u8>) {
            match self.stage {
                0 => out.extend([0, 1]),
                _ => out.extend([0, 1, 2]),
            }
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _ctx: &(), choice: &u8, _rng: &mut R) {
            if *choice == 0 {
                self.stage = 1;
            } else {
                self.stage = 2;
                self.payoff = 0.25;
            }
        }

        fn apply_joint<R: Rng + ?Sized>(
            &mut self,
            _ctx: &(),
            joint: JointChoices<'_, u8>,
            _rng: &mut R,
        ) {
            self.payoff = RPS[*joint.get(0) as usize][*joint.get(1) as usize];
            self.stage = 2;
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _ctx: &(), rng: &mut R) -> [f64; 2] {
            if self.stage == 1 {
                let a = below(rng, 3) as usize;
                let b = below(rng, 3) as usize;
                self.payoff = RPS[a][b];
                self.stage = 2;
            }
            [self.payoff, 1.0 - self.payoff]
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

    fn arm_visits<C>(root: &Node<C>, player: u8) -> Vec<u32> {
        let marginals = root.marginals(player).expect("this player acts here");
        (0..marginals.len()).map(|a| marginals.visits(a)).collect()
    }

    /// The whole point of the feature: a game whose equilibrium is mixed must
    /// not come back as one action played nearly always.
    #[test]
    fn a_simultaneous_root_plays_a_mixed_strategy() {
        let game = Rps::<true, false>::default();
        let mut searcher = Searcher::new(&game);
        searcher.search(&game, &(), 0, &config(20_000), None, &mut rng(7));

        let mut policy = Vec::new();
        assert!(searcher.root_policy_into(&game, &(), 0, &mut policy));
        assert_eq!(policy.len(), 3);
        let total: f64 = policy.iter().map(|&(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-9, "{policy:?}");
        for &(choice, probability) in &policy {
            assert!(
                (0.2..=0.5).contains(&probability),
                "action {choice} was played with probability {probability}"
            );
        }
    }

    /// Decoupled UCB1 converges to a pure profile. The contrast is why regret
    /// matching is the default, and it fails loudly if `Duct` starts mixing.
    #[test]
    fn duct_plays_a_pure_strategy_on_the_same_game() {
        let game = Rps::<false, false>::default();
        let mut searcher = Searcher::new(&game);
        searcher.search(&game, &(), 0, &config(20_000), None, &mut rng(7));

        let mut policy = Vec::new();
        assert!(searcher.root_policy_into(&game, &(), 0, &mut policy));
        let mass: Vec<f64> = policy.iter().map(|&(_, p)| p).collect();
        assert_eq!(mass.iter().filter(|&&p| p > 0.0).count(), 1, "{mass:?}");
    }

    /// Sampling is seeded, so it reproduces — including the extra draw
    /// `RootPolicy::Sampled` takes after the loop.
    #[test]
    fn same_seed_gives_the_same_simultaneous_search() {
        let game = Rps::<true, false>::default();
        let run = |seed| {
            let mut searcher = Searcher::new(&game);
            let result = searcher.search(&game, &(), 1, &config(3_000), None, &mut rng(seed));
            let tree = searcher.tree().expect("a search leaves a tree");
            (result.choice, arm_visits(tree, 0), arm_visits(tree, 1))
        };
        assert_eq!(run(11), run(11));
    }

    /// The returned move is the perspective player's own marginal, read off the
    /// arms and never off a joint successor.
    #[test]
    fn most_visited_returns_the_perspective_players_own_arm() {
        let game = Rps::<true, false>::default();
        let cfg = Config {
            simultaneous: SimultaneousConfig {
                root_policy: RootPolicy::MostVisited,
                ..Default::default()
            },
            ..config(2_000)
        };
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 1, &cfg, None, &mut rng(3));

        let tree = searcher.tree().expect("a search leaves a tree");
        let marginals = tree.marginals(1).expect("player 1 acts here");
        let best = marginals.most_visited().expect("something was selected");
        assert_eq!(result.choice, *marginals.choice(best));
        assert_eq!(result.best_visits, marginals.visits(best));
        assert_eq!(result.best_mean_reward, marginals.mean_reward(best));
        assert!(tree.children().len() > 1, "joint successors materialized");
    }

    /// A simultaneous node below a sequential one: the frame cursor has to match
    /// a level that is not the root, and the branch worth 0.5 in equilibrium has
    /// to beat the one worth 0.25 outright.
    #[test]
    fn a_mixed_tree_prefers_the_simultaneous_branch() {
        let game = SeqThenSimul::default();
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &config(20_000), None, &mut rng(5));
        assert_eq!(result.choice, 0);

        let tree = searcher.tree().expect("a search leaves a tree");
        let simultaneous = tree
            .children()
            .iter()
            .find(|child| child.simultaneous_players() == Some(PlayerSet::first_n(2)))
            .expect("the tree holds a simultaneous node");
        assert!((simultaneous.mean_reward() - 0.5).abs() < 0.1);
    }

    /// A joint successor that was materialized can be re-rooted at; one that was
    /// not drops the tree rather than promoting the wrong subtree.
    #[test]
    fn reuse_joint_hits_a_visited_tuple_and_misses_an_unvisited_one() {
        let game = SeqThenSimul {
            stage: 1,
            payoff: 0.0,
        };
        let mut searcher = Searcher::new(&game);
        searcher.search(&game, &(), 0, &config(2_000), None, &mut rng(2));

        let root = searcher.tree().expect("a search leaves a tree");
        let visited: Vec<(u8, u8)> = (0..root.children().len())
            .map(|child| {
                let arm = |player: u8| {
                    let slot = root.joint_arm(child, player).expect("a joint successor");
                    *root.marginals(player).expect("a participant").choice(slot)
                };
                (arm(0), arm(1))
            })
            .collect();
        assert!(!visited.is_empty());

        let (a, b) = visited[0];
        let before = arm_visits(root, 0);
        assert!(searcher.reuse_joint(&[(1, b), (0, a)]));
        assert!(searcher.tree().is_some());

        let missing = (0..3)
            .flat_map(|a| (0..3).map(move |b| (a, b)))
            .find(|pair| !visited.contains(pair));
        if let Some((a, b)) = missing {
            let mut searcher = Searcher::new(&game);
            searcher.search(&game, &(), 0, &config(2_000), None, &mut rng(2));
            assert!(!searcher.reuse_joint(&[(0, a), (1, b)]));
            assert!(searcher.tree().is_none());
        }
        assert_eq!(before.len(), 3);
    }

    /// A tuple that names a non-participant is a caller bug, not a miss.
    #[test]
    #[should_panic(expected = "who does not act at this node")]
    fn reuse_joint_rejects_a_non_participant() {
        let game = Rps::<true, false>::default();
        let mut searcher = Searcher::new(&game);
        searcher.search(&game, &(), 0, &config(100), None, &mut rng(1));
        searcher.reuse_joint(&[(0, 0), (4, 1)]);
    }

    /// Naming one player twice would leave another slot at its default arm and
    /// re-root at a tuple nobody played.
    #[test]
    #[should_panic(expected = "twice")]
    fn reuse_joint_rejects_a_repeated_player() {
        let game = Rps::<true, false>::default();
        let mut searcher = Searcher::new(&game);
        searcher.search(&game, &(), 0, &config(100), None, &mut rng(1));
        searcher.reuse_joint(&[(0, 0), (0, 1)]);
    }

    /// `StopReason::Proven` is unreachable under a mixing policy: the answer is
    /// a draw from a distribution, and the proof is about an argmax. Under
    /// `Duct` the cheap proof does run, and must still return a legal action.
    #[test]
    fn early_termination_never_proves_a_mixed_simultaneous_root() {
        let mixed = Rps::<true, false>::default();
        let cfg = Config {
            early_termination: true,
            ..config(2_000)
        };
        let mut searcher = Searcher::new(&mixed);
        let result = searcher.search(&mixed, &(), 0, &cfg, None, &mut rng(4));
        assert_eq!(result.stop_reason, StopReason::Budget);
        assert_eq!(result.root_visits, 2_000);

        let pure = Rps::<false, false>::default();
        let mut searcher = Searcher::new(&pure);
        let result = searcher.search(&pure, &(), 0, &cfg, None, &mut rng(4));
        assert!(result.choice < 3);
        assert!(result.root_visits <= 2_000);
    }

    /// The invariant fast path skips move generation but must not skip the
    /// availability stamp: an arm whose `availability` stops growing keeps a
    /// stale exploration bonus and a strategy average divided by the wrong
    /// denominator.
    #[test]
    fn an_invariant_simultaneous_root_stamps_every_arm_anyway() {
        let game = Rps::<true, true>::default();
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &config(5_000), None, &mut rng(9));

        let tree = searcher.tree().expect("a search leaves a tree");
        for player in 0..2 {
            let marginals = tree.marginals(player).expect("this player acts here");
            assert_eq!(marginals.len(), 3);
            for arm in 0..marginals.len() {
                assert_eq!(marginals.availability(arm), result.root_visits);
            }
            let selected: u32 = (0..marginals.len()).map(|a| marginals.visits(a)).sum();
            assert_eq!(selected, result.root_visits);
        }
    }

    /// A player with one action needs no search, and at a simultaneous root the
    /// opponents cannot change that.
    #[test]
    fn a_forced_simultaneous_move_skips_the_search() {
        #[derive(Clone, Default)]
        struct Forced;
        impl Game for Forced {
            type Choice = u8;
            type Rewards = [f64; 2];
            type Context = ();
            type Side = ();
            fn status(&self, _ctx: &()) -> Status<[f64; 2]> {
                Status::Simultaneous {
                    players: PlayerSet::first_n(2),
                }
            }
            fn choices_into(&self, _ctx: &(), out: &mut Vec<u8>) {
                out.push(9);
            }
            fn apply_choice<R: Rng + ?Sized>(&mut self, _c: &(), _x: &u8, _r: &mut R) {}
            fn rollout<R: Rng + ?Sized>(&mut self, _c: &(), _r: &mut R) -> [f64; 2] {
                [0.0, 0.0]
            }
            fn new_buffer(&self) -> Self {
                Self
            }
            fn determinize_into<R: Rng + ?Sized>(
                &self,
                _d: &mut Self,
                _c: &(),
                _p: u8,
                _r: &mut R,
            ) {
            }
        }

        let mut searcher = Searcher::new(&Forced);
        let result = searcher.search(&Forced, &(), 1, &config(1_000), None, &mut rng(1));
        assert_eq!(result.choice, 9);
        assert_eq!(result.stop_reason, StopReason::SingleChoice);
        assert_eq!(result.iterations_used, 0);
    }

    /// A frame is pushed once per simultaneous level of every descent, so its
    /// size is a per-iteration cost. The fourth field exists only to pin the
    /// backup ordering in debug builds and must not survive into release.
    #[test]
    fn a_descent_frame_costs_three_words() {
        let expected = if cfg!(debug_assertions) { 16 } else { 12 };
        assert_eq!(size_of::<SimFrame>(), expected);
    }
}
