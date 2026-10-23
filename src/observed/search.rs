//! The descent, for games that implement [`crate::Observed`].
//!
//! One tree per player, advanced along the same sampled world. Each player's
//! own moves are chosen from its own tree, and every tree records the move under
//! its own perception of it, so the trees' paths stay aligned while their
//! information sets do not. The searcher's answer is read off the perspective
//! player's tree, which is the only one whose root move is a real
//! [`Game::Choice`](crate::Game::Choice) in the position being searched.

use std::sync::atomic::{AtomicBool, Ordering};

use rand_core::Rng;
#[cfg(debug_assertions)]
use rand_core::TryRng;

#[cfg(debug_assertions)]
use hashbrown::HashTable;

use crate::budget::Deadline;
use crate::game::{Game, Observed, Rewards, Status};
use crate::observed::node::{select, ObservedNode};
use crate::rank::{leader_of, Candidate};
#[cfg(debug_assertions)]
use crate::search::RootCheck;
use crate::search::{Config, SearchResult, StopReason, DEADLINE_CHECK_MASK};
use crate::util::below;
#[cfg(debug_assertions)]
use crate::util::hash_of;

/// Owns every reusable allocation and one retained tree per player.
///
/// Keep one per game (not per player — the trees cover all of them) and call
/// `search` repeatedly: the scratch state, the choice and key buffers, the
/// per-observer paths and the trees themselves are all reused.
///
/// # Zero cost when unused
///
/// [`Game`] gains nothing: [`Observed`] is a separate trait, so a game that
/// does not conceal anything never names a key type and never implements a
/// method it will not call. Nor does [`crate::Searcher`] consult any of this —
/// the node, the descent and the backup are the ones they always were.
///
/// ```compile_fail
/// # use mcts::rand_core::Rng;
/// # use mcts::{Game, ObservedSearcher, Status};
/// # #[derive(Clone)] struct Plain;
/// # impl Game for Plain {
/// #     type Choice = ();
/// #     type Rewards = [f64; 2];
/// #     type Context = ();
/// #     type Side = ();
/// #     fn status(&self, _: &()) -> Status<[f64; 2]> { Status::Active { player: 0 } }
/// #     fn choices_into(&self, _: &(), out: &mut Vec<()>) { out.push(()); }
/// #     fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &(), _: &mut R) {}
/// #     fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] { [0.5, 0.5] }
/// #     fn new_buffer(&self) -> Self { Plain }
/// #     fn determinize_into<R: Rng + ?Sized>(&self, _: &mut Self, _: &(), _: u8, _: &mut R) {}
/// # }
/// let game = Plain;
/// // `Plain` does not implement `Observed`, so it cannot be searched with this
/// // searcher — and `mcts::Searcher` never asks it to.
/// let _ = ObservedSearcher::new(&game, 1);
/// ```
pub struct ObservedSearcher<G: Game + Observed> {
    scratch: Scratch<G>,
    /// One per player, in player order. A placeholder root between searches:
    /// every search installs the real one, and the entries hold node capacity
    /// for the trees beyond that.
    trees: Vec<ObservedNode<G::Key>>,
    side: G::Side,
    players: u8,
    /// Whether the retained trees describe the position about to be searched.
    /// Only [`ObservedSearcher::reuse_subtree`] sets it.
    tree_is_current: bool,
}

struct Scratch<G: Game + Observed> {
    state: G,
    /// The acting player's legal moves this step.
    choices: Vec<G::Choice>,
    /// Their keys, in the acting player's own tree.
    keys: Vec<G::Key>,
    /// Every other observer's key for the chosen move.
    other_keys: Vec<G::Key>,
    avail: Vec<bool>,
    /// Per tree, the child indices taken to reach this iteration's node.
    paths: Vec<Vec<u32>>,
    debug: DebugScratch<G>,
}

/// Debug-only: a second state buffer and a generator of its own.
///
/// The generator is separate on purpose. A check that drew from the caller's
/// `rng` would change the world each iteration samples, and a debug build would
/// then search a different tree from a release build of the same code — which is
/// the one thing the crate's debug checks must never do.
#[cfg(debug_assertions)]
struct DebugScratch<G: Game> {
    probe: G,
    rng: SplitMix64,
}

#[cfg(not(debug_assertions))]
struct DebugScratch<G: Game>(core::marker::PhantomData<G>);

impl<G: Game + Observed> ObservedSearcher<G> {
    /// `template` is only used to size the scratch states; its contents are
    /// overwritten by the first determinization.
    ///
    /// `players` is how many seats the game has, and player indices must be
    /// `0..players`. Passing it rather than discovering it keeps every player's
    /// tree on the same path: a tree that appeared only when its owner first
    /// acted would have missed the edges above that point.
    pub fn new(template: &G, players: u8) -> Self {
        assert!(
            players >= 1,
            "mcts: ObservedSearcher needs at least one player"
        );
        assert!(
            players as usize <= crate::game::PlayerSet::MAX_PLAYER as usize + 1,
            "mcts: ObservedSearcher was built for {players} players; the crate numbers players \
             0..={}",
            crate::game::PlayerSet::MAX_PLAYER
        );
        Self {
            scratch: Scratch {
                state: template.new_buffer(),
                choices: Vec::new(),
                keys: Vec::new(),
                other_keys: Vec::new(),
                avail: Vec::new(),
                paths: (0..players).map(|_| Vec::new()).collect(),
                debug: DebugScratch::new(template),
            },
            trees: (0..players).map(|_| ObservedNode::new_root(0)).collect(),
            side: Default::default(),
            players,
            tree_is_current: false,
        }
    }

    /// The number of seats this searcher holds trees for.
    pub fn players(&self) -> u8 {
        self.players
    }

    /// `player`'s retained tree, for inspection or display. The root's
    /// `visits` is the search's iteration count.
    pub fn tree(&self, player: u8) -> Option<&ObservedNode<G::Key>> {
        self.trees.get(player as usize)
    }

    /// Discard every retained tree.
    pub fn clear_tree(&mut self) {
        for tree in &mut self.trees {
            *tree = ObservedNode::new_root(0);
        }
        self.tree_is_current = false;
    }

    /// Re-root every tree at `choice`, so the next search inherits their
    /// statistics. Returns false, and drops the trees, if any observer's tree
    /// has no such edge.
    ///
    /// `state` and `ctx` are the position the retained trees were built from —
    /// the same `state` that was passed to [`Self::search`], before `choice` was
    /// applied. The keys that re-root each tree are read from it, because a
    /// perception is a fact about the position the move was made in.
    ///
    /// Only sound when the next search continues from that position with no
    /// hidden information revealed in between. The caller owns that judgement,
    /// as with [`crate::Searcher::reuse_subtree`].
    pub fn reuse_subtree(&mut self, state: &G, ctx: &G::Context, choice: &G::Choice) -> bool {
        let actor = match state.status(ctx) {
            Status::Active { player } => player,
            _ => {
                self.clear_tree();
                return false;
            }
        };
        let mut keys = Vec::with_capacity(self.trees.len());
        for observer in 0..self.players {
            keys.push(state.observed_key(ctx, choice, actor, observer));
        }
        let mut rerooted = keys.len() == self.trees.len();
        for (tree, key) in self.trees.iter_mut().zip(keys.iter()) {
            if !tree.reroot_at(key) {
                rerooted = false;
                break;
            }
        }
        if rerooted {
            self.tree_is_current = true;
        } else {
            self.clear_tree();
        }
        rerooted
    }

    /// Search `state` on behalf of `perspective`, whose turn it must be.
    ///
    /// Every iteration samples one world consistent with `perspective`'s
    /// knowledge and then lets each player choose from its own tree, so a
    /// simulated opponent answers what it observed rather than what the sample
    /// knows. See [`crate::Observed`] for the key that defines "observed".
    ///
    /// # Panics
    ///
    /// On a terminal state, at a [`Status::Simultaneous`] node, on behalf of a
    /// player who is not the one to act, or for a player index outside
    /// `0..players`.
    pub fn search<R: Rng + ?Sized>(
        &mut self,
        state: &G,
        ctx: &G::Context,
        perspective: u8,
        cfg: &Config,
        cancel: Option<&AtomicBool>,
        rng: &mut R,
    ) -> SearchResult<G::Choice> {
        // Consumed on entry, as in `Searcher`: an exit that never reaches the
        // end cannot leave the previous position's trees armed.
        let reuse = core::mem::take(&mut self.tree_is_current);

        cfg.validate();

        let root_player = match state.status(ctx) {
            Status::Active { player } => player,
            Status::Simultaneous { .. } => panic!(
                "mcts: ObservedSearcher descended into a Status::Simultaneous state. Concurrent \
                 moves are not modelled by per-observer trees — their choices are not conditioned \
                 on one another's observations in the way a tree per player expresses. Search a \
                 game with concurrent moves with Searcher, and do not mix the two within one \
                 position."
            ),
            Status::Terminal(_) => panic!("mcts: search called on a terminal state"),
        };
        assert_eq!(
            perspective, root_player,
            "mcts: ObservedSearcher was asked for player {perspective}'s move, but the position's \
             decision belongs to player {root_player}. The tree is keyed by the perspective \
             player's observations and its answer is read off that tree, so the two must agree."
        );
        assert!(
            (root_player as usize) < self.trees.len(),
            "mcts: ObservedSearcher was built for {} players but the state has player \
             {root_player} to act",
            self.trees.len()
        );

        let Self {
            scratch,
            trees,
            side,
            ..
        } = self;

        scratch.choices.clear();
        state.choices_into(ctx, &mut scratch.choices);
        match scratch.choices.len() {
            0 => panic!(
                "mcts: search called on a state where player {perspective} has no legal choices"
            ),
            // Nothing the opponents do changes which move this is, in any
            // observation, so it is the answer without a search.
            1 => {
                return SearchResult {
                    choice: scratch.choices[0].clone(),
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

        if !reuse {
            for tree in trees.iter_mut() {
                *tree = ObservedNode::new_root(root_player);
            }
        } else {
            // A re-rooted tree carries the previous mover's index, so what its
            // cumulative reward means would otherwise depend on whether the
            // trees were reused. Same rule as `Searcher`.
            for tree in trees.iter_mut() {
                tree.player = root_player;
                tree.cumulative_reward = 0.0;
            }
        }
        for path in scratch.paths.iter_mut() {
            path.clear();
        }
        state.init_side(ctx, side);

        let target = cfg.iterations;
        let deadline = Deadline::new(cfg.time_limit_ms, cfg.clock);
        let reused_iterations = trees[perspective as usize].visits();
        let mut iterations_used: u32 = 0;
        #[cfg(debug_assertions)]
        let mut root_check = RootCheck::<G::Choice>::new();

        let stop_reason = loop {
            if target != 0 && trees[perspective as usize].visits() >= target {
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
            let before = root_check.read::<G>(&scratch.state, ctx, None, perspective);
            scratch.state.advance(ctx, side, perspective, rng);
            #[cfg(debug_assertions)]
            root_check.check::<G>(&scratch.state, ctx, None, perspective, before);

            let rewards = run_iteration::<G, R>(trees, scratch, ctx, side, cfg, perspective, rng);
            G::credit_iteration(side, &rewards);
            iterations_used += 1;
        };

        // The answer is this position's move, not a key: read the real choices
        // back and keep the children the search built under their keys. A key
        // for the acting player is injective, so the match is one to one.
        let root = &trees[perspective as usize];
        scratch.choices.clear();
        state.choices_into(ctx, &mut scratch.choices);
        scratch.keys.clear();
        for choice in scratch.choices.iter() {
            scratch
                .keys
                .push(state.observed_key(ctx, choice, root_player, root_player));
        }
        let leader = leader_of(root.children().iter().enumerate().filter_map(|(i, child)| {
            let key = child.observed_key()?;
            if !scratch.keys.iter().any(|k| k == key) {
                return None;
            }
            Some((
                i,
                Candidate::new(
                    child.visits(),
                    child.times_available(),
                    child.cumulative_reward(),
                ),
            ))
        }));
        let (choice, best_visits, best_mean_reward) = match leader {
            Some((i, _)) => {
                let key = root.children()[i]
                    .observed_key()
                    .expect("a child of the root carries its key")
                    .clone();
                let ci = scratch
                    .keys
                    .iter()
                    .position(|k| *k == key)
                    .expect("the child was kept because it matched one of these keys");
                (
                    scratch.choices[ci].clone(),
                    root.children()[i].visits(),
                    root.children()[i].mean_reward(),
                )
            }
            // Nothing legal here was ever visited: cancelled or timed out
            // before a single iteration completed.
            None => {
                let k = below(rng, scratch.choices.len() as u64) as usize;
                (scratch.choices[k].clone(), 0, 0.0)
            }
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

impl<G: Game> DebugScratch<G> {
    #[cfg(debug_assertions)]
    fn new(template: &G) -> Self {
        Self {
            probe: template.new_buffer(),
            rng: SplitMix64(0x5EED_0B5E_4FED_0001),
        }
    }

    #[cfg(not(debug_assertions))]
    fn new(_template: &G) -> Self {
        Self(core::marker::PhantomData)
    }
}

/// The node at the end of `path` in `tree`.
fn node_at_mut<'a, K>(tree: &'a mut ObservedNode<K>, path: &[u32]) -> &'a mut ObservedNode<K> {
    let mut node = tree;
    for &i in path {
        node = &mut node.children[i as usize];
    }
    node
}

/// One iteration: the caller has already determinized — descend, and back the
/// sampled payoff up every observer's path.
fn run_iteration<G: Game + Observed, R: Rng + ?Sized>(
    trees: &mut [ObservedNode<G::Key>],
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
        keys,
        other_keys,
        avail,
        paths,
        debug,
    } = scratch;
    #[cfg(not(debug_assertions))]
    let _ = debug;

    for path in paths.iter_mut() {
        path.clear();
    }

    let rewards: G::Rewards;
    let mut depth = 0usize;

    loop {
        let actor = match state.status(ctx) {
            Status::Terminal(terminal) => {
                rewards = terminal;
                break;
            }
            Status::Active { player } => player,
            Status::Simultaneous { .. } => panic!(
                "mcts: ObservedSearcher descended into a Status::Simultaneous state. Concurrent \
                 moves are not modelled by per-observer trees; use Searcher for a game that has \
                 them, and do not mix the two within one position."
            ),
        };
        assert!(
            (actor as usize) < trees.len(),
            "mcts: ObservedSearcher was built for {} players but the state has player {actor} \
             acting",
            trees.len()
        );
        debug_assert!(
            paths.iter().all(|path| path.len() == depth),
            "mcts: an observer's tree is out of step with the descent"
        );

        // The actor's move, chosen from the actor's own tree.
        choices.clear();
        state.choices_into(ctx, choices);
        if choices.is_empty() {
            // As in `Searcher`: a determinization may deal a world where this
            // node's owner has nothing to do, and the declared floor is what an
            // iteration with no move to make is worth.
            rewards = G::Rewards::uniform(cfg.min_reward);
            break;
        }

        let ti = actor as usize;
        keys.clear();
        for choice in choices.iter() {
            let key = state.observed_key(ctx, choice, actor, actor);
            #[cfg(debug_assertions)]
            debug_check_key::<G>(state, debug, ctx, choice, actor, actor, &key);
            keys.push(key);
        }
        #[cfg(debug_assertions)]
        debug_assert_keys_are_injective(keys, actor);

        let selected: G::Choice;
        let leaf: bool;
        let chosen: usize;
        {
            let node = node_at_mut(&mut trees[ti], &paths[ti]);
            node.expand(keys, actor, G::CHILD_INDEX_THRESHOLD, avail, rng);
            let Some(i) = select(
                node,
                avail,
                cfg.exploration_constant,
                cfg.progressive_bias_weight,
            ) else {
                // The declared floor rather than a zero, as in `Searcher`: a
                // fabricated payoff outside the declared range lands in every
                // accumulator on the path.
                rewards = G::Rewards::uniform(cfg.min_reward);
                break;
            };
            let key = node.children[i]
                .observed_key()
                .expect("a child carries the key it was reached by")
                .clone();
            let ci = keys
                .iter()
                .position(|k| *k == key)
                .expect("the selected child was created from one of these keys");
            selected = choices[ci].clone();
            leaf = node.children[i].visits() == 0;
            chosen = i;
        }
        paths[ti].push(chosen as u32);

        // Every other observer's key for the same move, read before the move
        // changes the position the perception is taken in.
        other_keys.clear();
        for observer in 0..trees.len() as u8 {
            if observer == actor {
                continue;
            }
            let key = state.observed_key(ctx, &selected, actor, observer);
            #[cfg(debug_assertions)]
            debug_check_key::<G>(state, debug, ctx, &selected, actor, observer, &key);
            other_keys.push(key);
        }

        state.apply_choice(ctx, &selected, rng);
        state.advance(ctx, side, perspective, rng);

        let mut next = 0;
        for observer in 0..trees.len() {
            if observer == ti {
                continue;
            }
            let key = other_keys[next].clone();
            next += 1;
            let index = {
                let node = node_at_mut(&mut trees[observer], &paths[observer]);
                node.find_or_push(key, actor, G::CHILD_INDEX_THRESHOLD)
            };
            paths[observer].push(index as u32);
        }
        depth += 1;

        if leaf {
            if cfg.progressive_bias_weight != 0.0 {
                // The successor's prior, on the mover's scale: the child's mean
                // is stamped with `rewards.reward(child.player)`, which is the
                // player who moved here.
                let node = node_at_mut(&mut trees[ti], &paths[ti]);
                node.heuristic_bias = state.heuristic_bias(ctx, actor);
            }
            rewards = state.rollout(ctx, rng);
            break;
        }
    }

    // Back up every tree's path with the mover's payoff at each step, as
    // `Searcher` does: a node's owner is the player who moved into it, not the
    // observer whose tree holds it.
    for (observer, path) in paths.iter().enumerate() {
        let mut node = &mut trees[observer];
        let player = node.player;
        node.record(rewards.reward(player));
        for &i in path.iter() {
            node = &mut node.children[i as usize];
            let player = node.player;
            node.record(rewards.reward(player));
        }
    }

    rewards
}

/// Debug-only checks.
#[cfg(debug_assertions)]
fn debug_check_key<G: Game + Observed>(
    state: &G,
    debug: &mut DebugScratch<G>,
    ctx: &G::Context,
    choice: &G::Choice,
    actor: u8,
    observer: u8,
    key: &G::Key,
) {
    let DebugScratch { probe, rng } = debug;
    state.determinize_into(probe, ctx, observer, rng);
    let again = probe.observed_key(ctx, choice, actor, observer);
    assert!(
        again == *key,
        "mcts: Observed::observed_key is not a function of what player {observer} knows. The key \
         for a move by player {actor} changed after determinizing from player {observer}'s own \
         perspective, which by definition leaves everything {observer} knows untouched. A key \
         that reads hidden state splits one of {observer}'s information sets into many, and the \
         simulated player then chooses at each of them with knowledge it does not have — the \
         strategy fusion this searcher exists to prevent. Key the move, and what {observer} \
         knows, and nothing else."
    );
}

#[cfg(debug_assertions)]
fn debug_assert_keys_are_injective<K: Clone + Eq + core::hash::Hash>(keys: &[K], actor: u8) {
    if keys.len() > 64 {
        let mut table: HashTable<u32> = HashTable::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            let hash = hash_of(key);
            assert!(
                table.find(hash, |&j| &keys[j as usize] == key).is_none(),
                "mcts: Observed::observed_key gave two distinct legal moves of player {actor} \
                 the same key in that player's own tree. A player knows which move it is making, \
                 so in its own tree the key has to be injective — otherwise the searcher cannot \
                 say which move the root led to."
            );
            let _ = table.insert_unique(hash, i as u32, |&j| hash_of(&keys[j as usize]));
        }
        return;
    }
    for (i, key) in keys.iter().enumerate() {
        for earlier in keys.iter().take(i) {
            assert!(
                key != earlier,
                "mcts: Observed::observed_key gave two distinct legal moves of player {actor} \
                 the same key in that player's own tree. A player knows which move it is making, \
                 so in its own tree the key has to be injective — otherwise the searcher cannot \
                 say which move the root led to."
            );
        }
    }
}

/// A generator for the debug checks alone.
///
/// Never `pub`, never seeded from anything the caller can see, and never used by
/// the descent: the worlds the debug checks draw must not disturb the sequence
/// the real search draws, or a debug build and a release build of the same code
/// would search different trees.
#[cfg(debug_assertions)]
struct SplitMix64(u64);

#[cfg(debug_assertions)]
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(debug_assertions)]
impl TryRng for SplitMix64 {
    type Error = core::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok((self.next() >> 32) as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.next())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        for chunk in dst.chunks_mut(8) {
            let bytes = self.next().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        Ok(())
    }
}
