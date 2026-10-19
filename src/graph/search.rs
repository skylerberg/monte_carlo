use std::sync::atomic::{AtomicBool, Ordering};

use rand_core::Rng;

use crate::budget::Deadline;
use crate::duct;
use crate::early_stop;
use crate::game::{Game, JointChoices, PlayerSet, Rewards, Status, Transposition};
use crate::graph::node::{Graph, Link, Vertex, UNRESOLVED};
use crate::node::JointKey;
use crate::rank::{leader_of, Candidate};
#[cfg(debug_assertions)]
use crate::search::RootCheck;
use crate::search::{
    debug_assert_simul_arms_match_choices, extract_marginal_from, Config, SearchResult, StopReason,
    DEADLINE_CHECK_MASK,
};
use crate::select::ExplorationCount;
use crate::util::below;

/// One simultaneous level of the descent. See the tree's `SimFrame`, which this
/// is the same record as.
#[derive(Clone, Copy)]
struct Frame {
    /// Index into `path` of the vertex the joint action was taken at.
    depth: u32,
    arity: u32,
    prob_start: u32,
    #[cfg(debug_assertions)]
    epoch: u32,
}

/// What the descent took out of `path[d]`: a link, or a joint successor.
#[derive(Clone, Copy)]
struct Step {
    child: u32,
    joint: bool,
}

struct Scratch<G: Game> {
    state: G,
    choices: Vec<G::Choice>,
    avail: Vec<bool>,
    all_available: Vec<bool>,
    /// The vertices the descent entered, root first. A cycle never re-enters
    /// one, so each appears at most once and backup credits it once.
    path: Vec<u32>,
    /// `steps[d]` is what was taken out of `path[d]`. One longer than `path`
    /// less one, except where a cycle stopped the descent: that step's link was
    /// taken, so it is counted, but its successor is already on the path.
    steps: Vec<Step>,
    picks: Vec<u32>,
    frames: Vec<Frame>,
    probs: Vec<f32>,
    root_legal: Vec<bool>,
    root_legal_complete: bool,
}

/// A search that merges transpositions: every route into a position reads and
/// grows the same vertex.
///
/// The interface is [`crate::Searcher`]'s, and so is the algorithm everywhere a
/// position is reached only one way: a game whose [`Transposition::transposition_key`]
/// never merges anything is searched move for move as the tree searches it. What
/// changes is what happens when two routes meet. The tree grows a second node
/// and splits the position's statistics between the two; this links the second
/// route to the first route's vertex, so the position is searched once and
/// every parent reads one estimate of it.
///
/// Selection at a vertex reads each successor's **shared** mean and measures
/// exploration against the **link's** own counts — how often this parent took
/// it, and how often it was legal here — which is the UCT2 rule of Childs,
/// Brodeur and Kocsis, under the same ISMCTS availability treatment the tree
/// uses. See `TRANSPOSITIONS.md` for the derivation, what is claimed, and what
/// is not.
pub struct GraphSearcher<G: Transposition> {
    scratch: Scratch<G>,
    side: G::Side,
    graph: Option<Graph<G::Choice, G::Key>>,
    /// Whether the retained graph describes the position about to be searched.
    /// Consumed on entry to every search; only the reuse calls set it.
    graph_is_current: bool,
}

impl<G: Transposition> GraphSearcher<G> {
    /// `template` is only used to size the scratch state; its contents are
    /// overwritten by the first determinization.
    pub fn new(template: &G) -> Self {
        Self {
            scratch: Scratch {
                state: template.new_buffer(),
                choices: Vec::new(),
                avail: Vec::new(),
                all_available: Vec::new(),
                path: Vec::new(),
                steps: Vec::new(),
                picks: Vec::new(),
                frames: Vec::new(),
                probs: Vec::new(),
                root_legal: Vec::new(),
                root_legal_complete: false,
            },
            side: Default::default(),
            graph: None,
            graph_is_current: false,
        }
    }

    /// The retained graph, for inspection or display.
    pub fn graph(&self) -> Option<&Graph<G::Choice, G::Key>> {
        self.graph.as_ref()
    }

    /// Take the retained graph, leaving the searcher with none. The next search
    /// starts fresh.
    pub fn take_graph(&mut self) -> Option<Graph<G::Choice, G::Key>> {
        self.graph_is_current = false;
        self.graph.take()
    }

    /// Discard the retained graph.
    pub fn clear_graph(&mut self) {
        self.graph = None;
        self.graph_is_current = false;
    }

    #[cfg(feature = "parallel")]
    /// See [`crate::Searcher`]'s `disarm`.
    pub(crate) fn disarm(&mut self) {
        self.graph_is_current = false;
    }

    /// Re-root the retained graph at `choice` so the next search inherits what
    /// it holds about the position that choice leads to. Returns false, and
    /// drops the graph, if the root never took `choice`.
    ///
    /// Everything the new root cannot reach is discarded. Unlike a tree, that
    /// can keep positions the search reached through siblings of `choice`: a
    /// position reached through another root choice *and* through this one is
    /// still ahead, and keeps everything learned about it either way.
    ///
    /// The same soundness condition as [`crate::Searcher::reuse_subtree`]: the
    /// next search must continue from that position, with no hidden
    /// information revealed in between.
    pub fn reuse_subtree(&mut self, choice: &G::Choice) -> bool {
        let Some(graph) = self.graph.as_mut() else {
            return false;
        };
        let root = graph.root_vertex();
        debug_assert!(
            root.joint().is_none(),
            "mcts: reuse_subtree was called on a simultaneous root, whose successor \
             is determined by every participant's action rather than by one player's. \
             Use GraphSearcher::reuse_joint. A release build drops the graph and \
             returns false here instead of panicking."
        );
        let target = if root.joint().is_none() {
            root.find_link(choice)
                .map(|i| root.links[i].target)
                .filter(|&target| target != UNRESOLVED)
        } else {
            None
        };
        match target {
            Some(target) => {
                graph.reroot(target);
                self.graph_is_current = true;
                true
            }
            None => {
                self.graph = None;
                self.graph_is_current = false;
                false
            }
        }
    }

    /// Re-root the retained graph at the joint action a simultaneous round
    /// resolved to. `played` names every participant exactly once, in any
    /// order. See [`crate::Searcher::reuse_joint`], including why misses are
    /// common.
    pub fn reuse_joint(&mut self, played: &[(u8, G::Choice)]) -> bool {
        let Some(graph) = self.graph.as_mut() else {
            return false;
        };
        let root = graph.root_vertex();
        debug_assert!(
            root.joint().is_some(),
            "mcts: reuse_joint was called on a root that is not simultaneous. Use \
             GraphSearcher::reuse_subtree. A release build drops the graph and returns \
             false here instead of panicking."
        );

        let target = root.joint().and_then(|joint| {
            let simul = &joint.simul;
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
            if !hit {
                return None;
            }
            let key = JointKey::pack(picks, simul.key_bits);
            simul
                .find_joint(key)
                .map(|i| joint.targets[i])
                .filter(|&target| target != UNRESOLVED)
        });

        match target {
            Some(target) => {
                graph.reroot(target);
                self.graph_is_current = true;
                true
            }
            None => {
                self.graph = None;
                self.graph_is_current = false;
                false
            }
        }
    }

    /// `player`'s mixed strategy at a simultaneous root, renormalized over the
    /// actions legal in `state`. See [`crate::Searcher::root_policy_into`],
    /// which this is the same extraction as.
    pub fn root_policy_into(
        &self,
        state: &G,
        ctx: &G::Context,
        player: u8,
        out: &mut Vec<(G::Choice, f64)>,
    ) -> bool {
        out.clear();
        let Some(joint) = self
            .graph
            .as_ref()
            .and_then(|graph| graph.root_vertex().joint())
        else {
            return false;
        };
        let simul = &joint.simul;
        let Some(slot) = simul.players.slot_of(player) else {
            return false;
        };
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
            let share = 1.0 / choices.len() as f64;
            out.extend(choices.into_iter().map(|choice| (choice, share)));
        }
        true
    }

    /// Search from `state` on behalf of `perspective`. The same contract, and
    /// the same panics, as [`crate::Searcher::search`].
    pub fn search<R: Rng + ?Sized>(
        &mut self,
        state: &G,
        ctx: &G::Context,
        perspective: u8,
        cfg: &Config,
        cancel: Option<&AtomicBool>,
        rng: &mut R,
    ) -> SearchResult<G::Choice> {
        let reuse = core::mem::take(&mut self.graph_is_current);

        cfg.validate();

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

        if !reuse {
            self.graph = None;
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
            1 => {
                if let (Some(players), Some(graph)) = (root_players, self.graph.as_mut()) {
                    graph.root_vertex_mut().ensure_joint(players);
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

        if self.graph.is_none() {
            self.graph = Some(Graph::new(root_player));
        }
        state.init_side(ctx, &mut self.side);

        let Self {
            scratch,
            side,
            graph,
            ..
        } = self;
        let graph = graph.as_mut().expect("the graph was just created");

        let root = graph.root_vertex_mut();
        // As in the tree: the root accumulates the player to act, whether or not
        // it was promoted from a vertex that accumulated its mover.
        root.player = root_player;
        root.cumulative_reward = 0.0;
        if let Some(players) = root_players {
            root.ensure_joint(players);
        }

        scratch.root_legal.clear();
        scratch.root_legal_complete = false;
        let target = cfg.iterations;
        let deadline = Deadline::new(cfg.time_limit_ms);
        let reused_iterations = graph.root_vertex().visits;
        let mut iterations_used: u32 = 0;
        #[cfg(debug_assertions)]
        let mut root_check = RootCheck::<G::Choice>::new();

        let stop_reason = loop {
            if target != 0 && graph.root_vertex().visits >= target {
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
            debug_assert_key_holds(graph.root_vertex(), &scratch.state, ctx);

            let rewards = run_iteration::<G, R>(graph, scratch, ctx, side, cfg, perspective, rng);

            G::credit_iteration(side, &rewards);
            iterations_used += 1;

            if cfg.early_termination && target != 0 {
                refresh_root_legal(graph, state, ctx, scratch, root_players, perspective);
                if settled::<G>(
                    graph,
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

        refresh_root_legal(graph, state, ctx, scratch, root_players, perspective);
        let (choice, best_visits, best_mean_reward) = match root_players {
            None => match root_answer(graph, &scratch.root_legal) {
                Some(i) => {
                    let link = &graph.root_vertex().links[i];
                    let mean = if link.visits == 0 {
                        0.0
                    } else {
                        graph.vertices[link.target as usize].mean_reward()
                    };
                    (link.choice.clone(), link.visits, mean)
                }
                None => {
                    scratch.choices.clear();
                    state.choices_into(ctx, &mut scratch.choices);
                    let k = below(rng, scratch.choices.len() as u64) as usize;
                    (scratch.choices[k].clone(), 0, 0.0)
                }
            },
            Some(_) => extract_marginal_from::<G, R>(
                graph.root_vertex().joint().map(|joint| &joint.simul),
                state,
                ctx,
                &scratch.root_legal,
                &mut scratch.choices,
                perspective,
                cfg,
                rng,
            ),
        };

        SearchResult {
            choice,
            iterations_used,
            reused_iterations,
            stop_reason,
            root_visits: graph.root_vertex().visits,
            best_visits,
            best_mean_reward,
        }
    }
}

/// A root link as a ranking candidate: the link's own selections and
/// opportunities, which are this root's evidence, against its successor's
/// shared mean, which is the best estimate of what the choice is worth.
///
/// Where the successor was reached only through this link the two visit counts
/// agree, and the successor's total is passed through untouched rather than
/// rebuilt from its mean, so a graph that never merges ranks exactly as the
/// tree does.
#[inline]
pub(crate) fn link_candidate<C, K>(vertices: &[Vertex<C, K>], link: &Link<C>) -> Candidate {
    let (visits, times_available, cumulative) = link_stats(vertices, link);
    Candidate::new(visits, times_available, cumulative)
}

/// [`link_candidate`]'s three numbers — selections, opportunities, and a total
/// reward whose mean is the successor's — for a caller that pools them.
#[inline]
pub(crate) fn link_stats<C, K>(vertices: &[Vertex<C, K>], link: &Link<C>) -> (u32, u32, f64) {
    if link.visits == 0 {
        return (0, link.times_available, 0.0);
    }
    let target = &vertices[link.target as usize];
    let cumulative = if target.visits == link.visits {
        target.cumulative_reward
    } else {
        target.mean_reward() * link.visits as f64
    };
    (link.visits, link.times_available, cumulative)
}

/// The root's answer under the crate's one ranking, over the links legal in the
/// real position. `None` if none of them was ever taken.
fn root_answer<C, K>(graph: &Graph<C, K>, legal: &[bool]) -> Option<usize> {
    let root = graph.root_vertex();
    if legal.len() != root.links.len() {
        return None;
    }
    leader_of(
        root.links
            .iter()
            .enumerate()
            .filter(|(i, _)| legal[*i])
            .map(|(i, link)| (i, link_candidate(&graph.vertices, link))),
    )
    .map(|(i, _)| i)
}

/// [`early_stop::settled`] over a graph's root.
fn settled<G: Transposition>(
    graph: &Graph<G::Choice, G::Key>,
    root_players: Option<PlayerSet>,
    perspective: u8,
    target: u32,
    legal: &[bool],
    complete: bool,
) -> bool {
    let root = graph.root_vertex();
    let remaining = target.saturating_sub(root.visits);
    if remaining == 0 || !complete {
        return false;
    }
    if root_players.is_some() {
        let Some(joint) = root.joint() else {
            return false;
        };
        return early_stop::settled_arms::<G>(&joint.simul, perspective, legal, remaining);
    }
    if legal.len() != root.links.len() {
        return false;
    }
    let vertices = &graph.vertices;
    early_stop::proven(
        root.links
            .iter()
            .zip(legal)
            .enumerate()
            .filter(|(_, (_, &legal))| legal)
            .map(|(i, (link, _))| (i, link_candidate(vertices, link))),
        remaining,
    )
}

/// Keep `scratch.root_legal` covering the root's candidates, against the real
/// position. The tree's `refresh_root_legal`, over links.
fn refresh_root_legal<G: Transposition>(
    graph: &Graph<G::Choice, G::Key>,
    state: &G,
    ctx: &G::Context,
    scratch: &mut Scratch<G>,
    root_players: Option<PlayerSet>,
    perspective: u8,
) {
    let root = graph.root_vertex();
    let slot = match root_players {
        None => {
            debug_assert!(
                root.joint().is_none(),
                "mcts: this position is sequential but the root's statistics were built \
                 as a simultaneous vertex"
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
        let joint = root
            .joint()
            .expect("`search` installs the block before the first iteration");
        (&joint.simul, slot)
    });
    let candidates = match arms {
        Some((simul, slot)) => simul.slot_len(slot),
        None => root.links.len(),
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
                if let Some(link) = root.find_link(choice) {
                    scratch.root_legal[link] = true;
                    found += 1;
                }
            }
        }
    }
    scratch.root_legal_complete = found == scratch.choices.len();
}

/// The key a vertex was merged under must be the key of every state that
/// reaches it. Debug builds recompute it on every visit; see [`Transposition`].
#[inline]
fn debug_assert_key_holds<G: Transposition>(
    vertex: &Vertex<G::Choice, G::Key>,
    state: &G,
    ctx: &G::Context,
) {
    if !cfg!(debug_assertions) {
        return;
    }
    // A fresh search's root is never merged and carries no key, so there is
    // nothing for it to hold to. A promoted root keeps the key it was merged
    // under, and is held to it like any other vertex.
    let Some(merged) = vertex.key.as_ref() else {
        return;
    };
    let now = state.transposition_key(ctx);
    assert!(
        now.as_ref() == Some(merged),
        "mcts: Transposition::transposition_key changed for a position the search \
         had already merged. A vertex is resolved once, on the first iteration that \
         reaches it, and every later iteration that takes the same route lands in it \
         whatever its determinization drew or its chance events produced — so the \
         key must read only what is fixed for all of them. This one reads something \
         a determinization resamples or a chance event decides, and positions that \
         play differently are being merged. Return None for such positions."
    );
}

/// What a link resolves to on its first traversal: the vertex already merged
/// under this state's key for `mover`, or a new one.
///
/// Returns the vertex and whether it is new — a position no iteration has
/// evaluated, which the descent stops at to roll out.
#[inline]
fn resolve<G: Transposition>(
    graph: &mut Graph<G::Choice, G::Key>,
    state: &G,
    ctx: &G::Context,
    mover: u8,
) -> (u32, bool) {
    let key = state.transposition_key(ctx);
    if let Some(found) = key.as_ref().and_then(|key| graph.lookup(key, mover)) {
        return (found, graph.vertices[found as usize].visits == 0);
    }
    (graph.insert(mover, key), true)
}

/// UCB1 at a vertex: the successor's shared mean, explored against the link's
/// own counts, with the prior decaying in the successor's evidence.
///
/// Where the successor's only parent is this vertex the two counts agree and
/// this is the tree's `ucb_raw` term for term — the same operations in the same
/// order — which is what lets a graph that never merges reproduce a tree search
/// to the last bit.
#[inline(always)]
fn ucb<C, K>(
    target: &Vertex<C, K>,
    link: &Link<C>,
    ln_total: f64,
    exploration: f64,
    bias_weight: f64,
) -> f64 {
    let evidence = target.visits as f64;
    let value =
        target.cumulative_reward / evidence + exploration * (ln_total / link.visits as f64).sqrt();
    if bias_weight == 0.0 {
        return value;
    }
    value + bias_weight * target.heuristic_bias as f64 / (1.0 + evidence)
}

/// The link to descend, or `None` if nothing is legal here. The tree's
/// `select`, reading each successor's shared statistics.
fn select<C, K>(
    vertices: &[Vertex<C, K>],
    at: usize,
    avail: &[bool],
    exploration_count: ExplorationCount,
    exploration: f64,
    bias_weight: f64,
) -> Option<usize> {
    let vertex = &vertices[at];
    debug_assert_eq!(avail.len(), vertex.links.len());

    let ln_parent_visits = match exploration_count {
        ExplorationCount::ParentVisits => (vertex.visits as f64).ln(),
        ExplorationCount::TimesAvailable => 0.0,
    };

    let mut best = None;
    let mut best_value = f64::NEG_INFINITY;
    for (i, link) in vertex.links.iter().enumerate() {
        if !avail[i] {
            continue;
        }
        if link.visits == 0 {
            return Some(i);
        }
        let ln_total = match exploration_count {
            ExplorationCount::ParentVisits => ln_parent_visits,
            ExplorationCount::TimesAvailable => link.ln_times_available,
        };
        let target = &vertices[link.target as usize];
        let value = ucb(target, link, ln_total, exploration, bias_weight);
        if value > best_value {
            best_value = value;
            best = Some(i);
        }
    }
    best
}

fn mark_every_link_available<'a, C>(
    links: &mut [Link<C>],
    all_available: &'a mut Vec<bool>,
) -> &'a [bool] {
    for link in links.iter_mut() {
        link.times_available += 1;
    }
    all_available.clear();
    all_available.resize(links.len(), true);
    all_available
}

fn debug_assert_links_match_choices<G: Transposition>(
    vertex: &Vertex<G::Choice, G::Key>,
    state: &G,
    ctx: &G::Context,
    choices: &mut Vec<G::Choice>,
) {
    if !cfg!(debug_assertions) {
        return;
    }
    choices.clear();
    state.choices_into(ctx, choices);
    let every_choice_has_a_link = choices
        .iter()
        .all(|choice| vertex.find_link(choice).is_some());
    let every_link_is_a_choice = vertex
        .links
        .iter()
        .all(|link| choices.contains(&link.choice));
    assert!(
        every_choice_has_a_link && every_link_is_a_choice,
        "mcts: CHOICES_ON_REVISIT is {:?}, but this visit to a position offers {} choices \
         that do not match the {} links an earlier visit left there. The search would \
         have selected among the earlier visit's choices.",
        G::CHOICES_ON_REVISIT,
        choices.len(),
        vertex.links.len()
    );
}

/// One selection-expansion-simulation-backpropagation pass over the graph.
///
/// The tree's `run_iteration`, with two differences. A link's successor is
/// looked up by key the first time the link is taken, and the descent rolls out
/// only at a position no iteration has evaluated — reaching an existing vertex
/// by a new route continues the descent into what is already known there. And
/// a vertex already on this iteration's path ends the descent: that is a
/// repetition, which a tree cannot represent and a graph must not loop on.
fn run_iteration<G: Transposition, R: Rng + ?Sized>(
    graph: &mut Graph<G::Choice, G::Key>,
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
        all_available,
        path,
        steps,
        picks,
        frames,
        probs,
        ..
    } = scratch;
    path.clear();
    steps.clear();
    frames.clear();
    probs.clear();

    let stamp = graph.next_stamp();
    let root = graph.root;
    graph.vertices[root as usize].path_stamp = stamp;
    path.push(root);

    let rewards: G::Rewards;
    let mut at = root as usize;

    loop {
        let player = match state.status(ctx) {
            Status::Terminal(terminal_rewards) => {
                rewards = terminal_rewards;
                break;
            }
            Status::Active { player } => {
                debug_assert!(
                    graph.vertices[at].joint().is_none(),
                    "mcts: this position reported Status::Active {{ player: {player} }} but \
                     its statistics were built as a simultaneous vertex. A position's kind \
                     must not vary across determinizations, and two positions merged under \
                     one key must be the same kind."
                );
                player
            }
            Status::Simultaneous { players } => {
                let vertex = &mut graph.vertices[at];
                debug_assert!(
                    vertex.joint().is_none()
                        || vertex.joint().map(|joint| joint.simul.players) == Some(players),
                    "mcts: this position reported Status::Simultaneous with players {:?} but \
                     its statistics were built for {:?}. A position's participants must not \
                     vary across determinizations, and two positions merged under one key \
                     must have the same participants.",
                    players,
                    vertex.joint().map(|joint| joint.simul.players)
                );

                let arity = players.len();
                let epoch = vertex.visits;
                let policy = G::SIMULTANEOUS_POLICY;
                let is_root = at == root as usize;
                let choices_already_known =
                    G::CHOICES_ON_REVISIT.are_same(is_root) && vertex.links_match_choices;
                let joint = vertex.ensure_joint(players);

                let mut degenerate = false;
                if choices_already_known {
                    debug_assert_simul_arms_match_choices::<G>(
                        &joint.simul,
                        players,
                        state,
                        ctx,
                        choices,
                    );
                    joint.simul.mark_every_arm_available(epoch, policy);
                    joint.simul.debug_check_arrays(joint.targets.len());
                } else {
                    let mut every_arm_is_a_choice = true;
                    for slot in 0..arity {
                        let p = players.player_at(slot).expect("a slot has a participant");
                        choices.clear();
                        state.choices_for_into(ctx, p, choices);
                        let legal = joint.simul.expand_slot(
                            slot,
                            choices,
                            epoch,
                            policy,
                            G::CHILD_INDEX_THRESHOLD,
                        );
                        joint.simul.debug_check_arrays(joint.targets.len());
                        debug_assert!(
                            legal != 0,
                            "mcts: player {p} has no legal action at a simultaneous node, so \
                             no joint action exists. The iteration scores the declared \
                             minimum reward, which will drag this node's parent's mean down \
                             indistinguishably from a bad evaluation."
                        );
                        if legal == 0 {
                            degenerate = true;
                            break;
                        }
                        every_arm_is_a_choice &= legal == joint.simul.slot_len(slot);
                    }
                    graph.vertices[at].links_match_choices = !degenerate && every_arm_is_a_choice;
                }
                if degenerate {
                    rewards = G::Rewards::uniform(cfg.min_reward);
                    break;
                }

                let span = cfg.max_reward - cfg.min_reward;
                let prob_start = probs.len() as u32;
                picks.clear();
                let joint = graph.vertices[at]
                    .joint_mut()
                    .expect("the block was just installed");
                for slot in 0..arity {
                    let Some((pick, prob)) = duct::select_marginal(
                        &mut joint.simul,
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
                    probs.push(prob as f32);
                }
                if degenerate {
                    probs.truncate(prob_start as usize);
                    rewards = G::Rewards::uniform(cfg.min_reward);
                    break;
                }

                let key = JointKey::pack(picks, joint.simul.key_bits);
                let i = match joint.simul.find_joint(key) {
                    Some(i) => i,
                    None => {
                        let i = joint.targets.len();
                        joint.targets.push(UNRESOLVED);
                        joint.simul.record_joint(key, i, G::CHILD_INDEX_THRESHOLD);
                        i
                    }
                };

                steps.push(Step {
                    child: i as u32,
                    joint: true,
                });
                frames.push(Frame {
                    depth: (path.len() - 1) as u32,
                    arity: arity as u32,
                    prob_start,
                    #[cfg(debug_assertions)]
                    epoch,
                });

                let joint = graph.vertices[at]
                    .joint()
                    .expect("the block was just installed");
                let joint_choices = JointChoices::new(
                    players,
                    &joint.simul.arm_choices,
                    &joint.simul.starts,
                    picks,
                );
                state.apply_joint(ctx, joint_choices, rng);
                state.advance(ctx, side, perspective, rng);

                // A joint successor has no single mover, and accumulates the
                // reward of the player the search runs for — as in the tree.
                let resolved = graph.vertices[at]
                    .joint()
                    .expect("the block was just installed")
                    .targets[i];
                let (target, fresh) = if resolved == UNRESOLVED {
                    let (target, fresh) = resolve(graph, state, ctx, perspective);
                    graph.vertices[at]
                        .joint_mut()
                        .expect("the block was just installed")
                        .targets[i] = target;
                    (target, fresh)
                } else {
                    let target = &graph.vertices[resolved as usize];
                    debug_assert_key_holds(target, state, ctx);
                    (resolved, target.visits == 0)
                };

                if graph.vertices[target as usize].path_stamp == stamp {
                    rewards = state.rollout(ctx, rng);
                    break;
                }
                graph.vertices[target as usize].path_stamp = stamp;
                path.push(target);
                if fresh {
                    rewards = state.rollout(ctx, rng);
                    break;
                }
                at = target as usize;
                continue;
            }
        };

        let is_root = at == root as usize;
        let exploration_count = if is_root || G::CHOICES_ON_REVISIT.are_same(is_root) {
            ExplorationCount::ParentVisits
        } else {
            ExplorationCount::TimesAvailable
        };
        let vertex = &mut graph.vertices[at];
        let choices_already_known =
            G::CHOICES_ON_REVISIT.are_same(is_root) && vertex.links_match_choices;
        let available: &[bool] = if choices_already_known {
            debug_assert_links_match_choices::<G>(vertex, state, ctx, choices);
            mark_every_link_available(&mut vertex.links, all_available)
        } else {
            choices.clear();
            state.choices_into(ctx, choices);
            vertex.expand(
                is_root,
                choices,
                G::CHILD_INDEX_THRESHOLD,
                exploration_count,
                avail,
                rng,
            );
            avail
        };

        let picked = select(
            &graph.vertices,
            at,
            available,
            exploration_count,
            cfg.exploration_constant,
            cfg.progressive_bias_weight,
        );
        debug_assert!(
            picked.is_some(),
            "mcts: the descent cannot continue at this position. Either player {player} has \
             no legal choice here, or every legal link scored a non-finite UCB value. The \
             iteration scores the declared minimum reward, which will drag this vertex's \
             parent's mean down indistinguishably from a bad evaluation."
        );
        let Some(i) = picked else {
            rewards = G::Rewards::uniform(cfg.min_reward);
            break;
        };

        steps.push(Step {
            child: i as u32,
            joint: false,
        });
        let link = &graph.vertices[at].links[i];
        state.apply_choice(ctx, &link.choice, rng);
        state.advance(ctx, side, perspective, rng);

        let resolved = graph.vertices[at].links[i].target;
        let (target, fresh) = if resolved == UNRESOLVED {
            let (target, fresh) = resolve(graph, state, ctx, player);
            graph.vertices[at].links[i].target = target;
            (target, fresh)
        } else {
            let target = &graph.vertices[resolved as usize];
            debug_assert_key_holds(target, state, ctx);
            (resolved, target.visits == 0)
        };

        if graph.vertices[target as usize].path_stamp == stamp {
            // A repetition: the position is already being searched on this
            // path. Taking the link again would loop, and crediting the vertex
            // twice would count one iteration twice, so the descent ends here.
            rewards = state.rollout(ctx, rng);
            break;
        }
        graph.vertices[target as usize].path_stamp = stamp;
        path.push(target);

        if fresh {
            if cfg.progressive_bias_weight != 0.0 {
                // The mover's valuation, for the reason the tree gives: `ucb`
                // adds it to a mean kept in the mover's currency.
                graph.vertices[target as usize].heuristic_bias = state.heuristic_bias(ctx, player);
            }
            rewards = state.rollout(ctx, rng);
            break;
        }

        at = target as usize;
    }

    back_propagate::<G>(graph, path, steps, frames, probs, &rewards, cfg);
    rewards
}

/// Credit the iteration to every vertex it entered and every link it took.
///
/// A simultaneous level's arms are credited **before** that vertex's record, for
/// the reason the tree's backup gives: the legality test is
/// `avail_epoch == visits`, which the record moves.
fn back_propagate<G: Transposition>(
    graph: &mut Graph<G::Choice, G::Key>,
    path: &[u32],
    steps: &[Step],
    frames: &[Frame],
    probs: &[f32],
    rewards: &G::Rewards,
    cfg: &Config,
) {
    let mut cursor = 0;
    for (depth, &id) in path.iter().enumerate() {
        let vertex = &mut graph.vertices[id as usize];
        if let Some(frame) = frames.get(cursor).copied() {
            if frame.depth as usize == depth {
                cursor += 1;
                let epoch = vertex.visits;
                #[cfg(debug_assertions)]
                debug_assert_eq!(
                    epoch, frame.epoch,
                    "mcts: a simultaneous vertex's arms must be credited before its visit \
                     count moves"
                );
                let child = steps[depth].child as usize;
                let start = frame.prob_start as usize;
                let joint = vertex
                    .joint_mut()
                    .expect("mcts: a joint step was taken from a vertex with no marginals");
                let key = joint.simul.joint_keys[child];
                duct::credit_marginals(
                    &mut joint.simul,
                    key,
                    epoch,
                    &probs[start..start + frame.arity as usize],
                    rewards,
                    G::SIMULTANEOUS_POLICY,
                    cfg.min_reward,
                    cfg.max_reward,
                );
            }
        }
        vertex.record(rewards.reward(vertex.player));
    }
    for (depth, step) in steps.iter().enumerate() {
        if !step.joint {
            graph.vertices[path[depth] as usize].links[step.child as usize].visits += 1;
        }
    }
}
