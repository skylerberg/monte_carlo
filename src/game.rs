use core::hash::Hash;

use rand_core::Rng;

/// Per-player payoffs, produced once per iteration at the leaf.
///
/// Implemented for `[f64; N]`, which is what both known consumers use. Backup is
/// max^n: each node accumulates only the payoff of the player who moved there,
/// so nothing here assumes two players or zero sum.
pub trait Rewards: Copy {
    /// This player's payoff.
    fn reward(&self, player: u8) -> f64;

    /// The same payoff for every player, used for the degenerate node with no
    /// legal choices.
    ///
    /// The search builds that node's score out of
    /// [`crate::Config::min_reward`], so the value it fabricates is inside the
    /// range the caller declared. Fabricating a zero instead put a payoff the
    /// game never produced into every accumulator on the path — out of range
    /// for any game not paying through zero, and blamed on the game by the
    /// reward-range assertion at a simultaneous node.
    ///
    /// `Default` would not do even for the zero it replaces: std only
    /// implements it for arrays up to length 32, and requiring it would cap the
    /// player count for no reason.
    fn uniform(value: f64) -> Self;
}

impl<const N: usize> Rewards for [f64; N] {
    #[inline(always)]
    fn reward(&self, player: u8) -> f64 {
        self[player as usize]
    }

    #[inline(always)]
    fn uniform(value: f64) -> Self {
        [value; N]
    }
}

/// A set of players, as a bitmask over player indices `0..=63`.
///
/// A bitmask rather than a slice because [`Status`] is `Copy` and returned by
/// value from the hottest call in the crate: a `&[u8]` would put a lifetime on
/// every `fn status` signature in every game, and a `Vec<u8>` would allocate
/// once per node visit. The second reason matters more. A bitmask cannot
/// express order, so participant order is defined once, here, ascending — a
/// game physically cannot hand back `[1, 0]` on one determinization and
/// `[0, 1]` on the next and silently permute a node's marginal statistics.
///
/// The 64 is a cap on player *indices* that may appear here, not on how many
/// players a game has: [`Rewards::reward`] still takes any `u8`. A single
/// simultaneous node is separately capped at [`PlayerSet::MAX_SIMULTANEOUS`]
/// participants.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct PlayerSet(u64);

impl PlayerSet {
    /// Highest player index representable in a set.
    pub const MAX_PLAYER: u8 = 63;

    /// Most players that may act at one simultaneous node.
    ///
    /// Eight, because a joint successor is keyed by packing every
    /// participant's marginal index into one `u64`. Past eight the joint space
    /// is `prod(|A_i|)` with at least eight factors and the decoupled layer is
    /// the wrong data structure rather than merely a slow one.
    pub const MAX_SIMULTANEOUS: usize = 8;

    /// The set with no members.
    pub const EMPTY: Self = PlayerSet(0);

    /// The set holding exactly `players`.
    ///
    /// Panics if any player exceeds [`PlayerSet::MAX_PLAYER`].
    pub fn of(players: impl IntoIterator<Item = u8>) -> Self {
        let mut set = Self::EMPTY;
        for player in players {
            set = set.with(player);
        }
        set
    }

    /// The set `{0, 1, .., n - 1}`. `PlayerSet::first_n(2)` is the two-player
    /// case.
    ///
    /// Panics if `n` exceeds [`PlayerSet::MAX_PLAYER`] plus one.
    pub const fn first_n(n: u8) -> Self {
        assert!(
            n <= Self::MAX_PLAYER + 1,
            "mcts: PlayerSet holds player indices 0..=63"
        );
        if n == 0 {
            Self::EMPTY
        } else {
            PlayerSet(u64::MAX >> (64 - n as u32))
        }
    }

    /// The set whose members are the set bits of `bits`.
    pub const fn from_bits(bits: u64) -> Self {
        PlayerSet(bits)
    }

    /// The members, as a bitmask.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// This set with `player` added.
    ///
    /// Panics if `player` exceeds [`PlayerSet::MAX_PLAYER`].
    pub const fn with(self, player: u8) -> Self {
        assert!(
            player <= Self::MAX_PLAYER,
            "mcts: PlayerSet holds player indices 0..=63"
        );
        PlayerSet(self.0 | 1u64 << player)
    }

    /// Whether `player` is a member.
    pub const fn contains(self, player: u8) -> bool {
        player <= Self::MAX_PLAYER && (self.0 >> player) & 1 == 1
    }

    /// Number of members.
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Whether nobody is in the set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Position of `player` among the members in ascending order, or `None`.
    ///
    /// This is the **slot** index, and it is the one order everything else
    /// uses: marginal storage, joint keys, [`JointChoices`] and
    /// `Searcher::reuse_joint`. Nothing anywhere depends on any other
    /// ordering, so the four never have to be kept in sync by hand.
    pub const fn slot_of(self, player: u8) -> Option<usize> {
        if !self.contains(player) {
            return None;
        }
        Some((self.0 & ((1u64 << player) - 1)).count_ones() as usize)
    }

    /// The player occupying `slot`, or `None`.
    pub const fn player_at(self, slot: usize) -> Option<u8> {
        let mut bits = self.0;
        let mut remaining = slot;
        while bits != 0 {
            if remaining == 0 {
                return Some(bits.trailing_zeros() as u8);
            }
            bits &= bits - 1;
            remaining -= 1;
        }
        None
    }

    /// Members in ascending player index. Allocation-free.
    pub fn iter(self) -> PlayerSetIter {
        PlayerSetIter(self.0)
    }
}

/// Ascending iterator over a [`PlayerSet`].
#[derive(Clone, Copy, Debug)]
pub struct PlayerSetIter(u64);

impl Iterator for PlayerSetIter {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.0 == 0 {
            return None;
        }
        let player = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(player)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.0.count_ones() as usize;
        (remaining, Some(remaining))
    }
}

/// Whether a state needs a decision, or is finished and worth `Rw`.
///
/// Games fold their own cutoffs into `Terminal` — a round horizon with a
/// heuristic evaluation is reported here exactly like a real game end.
#[derive(Clone, Copy, Debug)]
pub enum Status<Rw> {
    /// `player` must choose.
    Active {
        /// The player to act.
        player: u8,
    },
    /// Every player in `players` chooses at the same time, none of them seeing
    /// the others' choice, and the tuple of their actions determines the
    /// successor.
    ///
    /// Enumerate each participant's own options with [`Game::choices_for_into`]
    /// and resolve the tuple with [`Game::apply_joint`]. The set must be
    /// non-empty and hold at most [`PlayerSet::MAX_SIMULTANEOUS`] players.
    ///
    /// A node's *kind* and its exact participant set must not vary across
    /// determinizations — determinization may change which actions are legal,
    /// never who acts. An information set whose kind depends on hidden state is
    /// not one information set. Debug builds check this on every visit; in
    /// release the search reads one node's statistics as the other kind's and
    /// returns a choice belonging to the wrong player, which is a wrong move
    /// rather than a slow search.
    Simultaneous {
        /// The players choosing concurrently.
        players: PlayerSet,
    },
    /// Finished, or cut off by the game's own horizon.
    Terminal(Rw),
}

/// One action per participant at a simultaneous node, borrowed from the tree.
///
/// A borrowing view rather than a slice of owned choices: resolving a joint
/// action must not clone one `Choice` per participant per iteration, and the
/// marginal that owns each choice already lives in the node. The three slices
/// it holds are disjoint from the `&mut self` the game applies them to, so it
/// costs no scratch buffer and no clone.
pub struct JointChoices<'a, C> {
    players: PlayerSet,
    /// Every participant's actions, concatenated in slot order.
    arm_choices: &'a [C],
    /// Where each slot's actions begin in `arm_choices`.
    starts: &'a [u32],
    /// Each slot's chosen action, as an index within that slot.
    picks: &'a [u32],
}

// Hand-written: `derive` would demand `C: Copy`, and the whole point of the
// view is that it copies without touching a `Choice` at all.
impl<C> Clone for JointChoices<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for JointChoices<'_, C> {}

impl<'a, C> JointChoices<'a, C> {
    pub(crate) fn new(
        players: PlayerSet,
        arm_choices: &'a [C],
        starts: &'a [u32],
        picks: &'a [u32],
    ) -> Self {
        debug_assert_eq!(
            players.len(),
            picks.len(),
            "mcts: a joint action must name exactly one choice per participant"
        );
        Self {
            players,
            arm_choices,
            starts,
            picks,
        }
    }

    /// The players who chose.
    pub fn players(&self) -> PlayerSet {
        self.players
    }

    /// Number of participants.
    pub fn len(&self) -> usize {
        self.picks.len()
    }

    /// Whether nobody chose. Never true at a well-formed simultaneous node.
    pub fn is_empty(&self) -> bool {
        self.picks.is_empty()
    }

    /// The player occupying `slot`, in ascending player order.
    ///
    /// Panics if `slot >= len()`.
    pub fn player_at(&self, slot: usize) -> u8 {
        self.players
            .player_at(slot)
            .expect("mcts: joint action slot is out of range")
    }

    /// The action chosen in `slot`. Panics if `slot >= len()`.
    pub fn get(&self, slot: usize) -> &'a C {
        let pick = self.picks[slot] as usize;
        &self.arm_choices[self.starts[slot] as usize + pick]
    }

    /// This player's action, or `None` if they are not a participant.
    pub fn for_player(&self, player: u8) -> Option<&'a C> {
        self.players.slot_of(player).map(|slot| self.get(slot))
    }

    /// `(player, action)` in ascending player order.
    pub fn iter(&self) -> impl Iterator<Item = (u8, &'a C)> + 'a {
        let joint = *self;
        joint
            .players
            .iter()
            .enumerate()
            .map(move |(slot, player)| (player, joint.get(slot)))
    }
}

impl<C> core::ops::Index<usize> for JointChoices<'_, C> {
    type Output = C;

    fn index(&self, slot: usize) -> &C {
        self.get(slot)
    }
}

/// Which bandit each player runs over their own actions at a simultaneous node.
///
/// Sequential nodes are always UCB1; this never touches them. It is an
/// associated const on [`Game`] rather than a `Config` field because which
/// equilibria a game has is a property of the game, like
/// [`Game::ROOT_CHOICES_INVARIANT`] — and because a const folds the whole
/// unchosen branch away.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SimultaneousPolicy {
    /// Regret matching on bandit feedback, with a decaying exploration floor
    /// and the availability correction determinization requires.
    ///
    /// The default. Each player's time-averaged strategy converges to the
    /// coarse correlated equilibrium set — to a Nash equilibrium only in
    /// two-player zero-sum games, and only at a single simultaneous ply. It
    /// needs exactly the information [`SimultaneousPolicy::Duct`] needs: no
    /// joint-payoff matrix, no counterfactual row, no extra hash probes. The
    /// price is variance, because the regret estimate is importance-weighted
    /// rather than bootstrapped from the joint children.
    ///
    /// See the crate docs for what determinization does to that claim; the
    /// short version is that the equilibrium reached is one of the
    /// determinization-averaged game, never of the true imperfect-information
    /// game.
    #[default]
    RegretMatching,
    /// Decoupled UCB1 over each player's own actions, on the availability
    /// count, with ties broken at random inside a tolerance.
    ///
    /// The cheapest option and the strongest raw performer in most published
    /// comparisons — and it **provably does not converge** in games whose
    /// equilibrium is mixed. On biased rock-paper-scissors it settles into a
    /// stable cycle whose resting point is a function of the exploration
    /// constants rather than of the game, and an opponent who notices the skew
    /// exploits it. Randomized tie-breaking, which this crate applies at
    /// simultaneous nodes and nowhere else, takes measured exploitability there
    /// from 0.5-0.8 down to 0.01-0.05; it does not make the policy converge.
    /// Pick this only when you know your simultaneous nodes have pure
    /// equilibria, or when you have measured it winning on your game.
    Duct,
}

impl SimultaneousPolicy {
    /// Whether this policy plays a mixed strategy.
    ///
    /// A mixed root cannot be proven settled: the answer is a draw from a
    /// distribution, so "the leader cannot be overtaken" does not imply "the
    /// returned action cannot change".
    pub const fn mixes(self) -> bool {
        matches!(self, Self::RegretMatching)
    }
}

/// The interface a game implements to be searched.
///
/// Players are `u8` indices so that rewards can be a plain array. Keep your own
/// player type and convert at the boundary.
pub trait Game: Sized {
    /// An edge in the tree.
    ///
    /// `Eq` drives the linear child scan; `Hash` is used only once a node grows
    /// past [`Game::CHILD_INDEX_THRESHOLD`] children.
    type Choice: Clone + Eq + Hash;

    /// Usually `[f64; N]` for `N` players.
    type Rewards: Rewards;

    /// Immutable data shared by every iteration of one search: evaluation
    /// tables, tuned parameters, precomputed scores, move-filtering flags.
    ///
    /// This exists so such data does not have to live in the game state, where
    /// it would be copied on every determinization. Use `()` if you have none.
    type Context;

    /// Mutable state that belongs to the search but not to the tree, such as a
    /// flat bandit over decisions the tree does not model. Use `()` if unused.
    type Side: Default;

    /// Set only when the root's legal choices provably do not vary across
    /// determinizations. Then the root is enumerated once per search instead of
    /// once per iteration.
    ///
    /// **Off by default, and turn it on only with evidence.** It is tempting to
    /// reason that determinization reshuffles only what is hidden from the
    /// searching player, so their own options cannot change — that argument is
    /// easy to make and hard to make correctly. When it is wrong, the search
    /// selects a choice that is illegal in the current determinization, and
    /// what you get is a wrong move rather than a slow search.
    ///
    /// Debug builds verify the claim on every iteration and panic if it breaks,
    /// so run your integration tests in debug before trusting it in release.
    ///
    /// At a simultaneous root this is restated per participant: for every
    /// player acting there, that player's own marginal set does not vary across
    /// determinizations. It says nothing about joint successors, which are
    /// created only as they are visited and are never "fully expanded".
    const ROOT_CHOICES_INVARIANT: bool = false;

    /// Child count above which a node builds a hash index for lookup.
    ///
    /// Below it, `expand` finds children by linear scan, which wins for small
    /// branching factors and cheap `Choice` equality. Above it the scan is
    /// quadratic in the branching factor and loses badly.
    ///
    /// The default is where `benches/child_storage.rs` measures the crossover:
    /// scanning is up to 1.17x faster at 4 children and break-even at 16, while
    /// indexing is 2.4x faster at 64 and 36x faster at 1024. A `Choice` that
    /// owns heap data crosses over sooner — around 8 — because both equality
    /// and hashing chase a pointer, so lower this if yours does.
    const CHILD_INDEX_THRESHOLD: usize = 16;

    /// The bandit each player runs at a [`Status::Simultaneous`] node. Ignored
    /// entirely by a game that never reports one.
    ///
    /// Leave it alone unless you have measured. The default converges to a
    /// mixed strategy; [`SimultaneousPolicy::Duct`] is cheaper per iteration
    /// but converges to a pure one, which a game like rock-paper-scissors
    /// punishes by exactly as much as its equilibrium is mixed.
    const SIMULTANEOUS_POLICY: SimultaneousPolicy = SimultaneousPolicy::RegretMatching;

    // ---- hot path: called at every node of every iteration ----

    /// Whether this state needs a decision, and from whom.
    fn status(&self, ctx: &Self::Context) -> Status<Self::Rewards>;

    /// Fill `out` with the legal choices. `out` is cleared by the caller and
    /// reused across iterations, so this should not allocate in steady state.
    fn choices_into(&self, ctx: &Self::Context, out: &mut Vec<Self::Choice>);

    /// Fill `out` with `player`'s own legal choices at a
    /// [`Status::Simultaneous`] state. Same contract as [`Game::choices_into`]:
    /// `out` is cleared by the caller and reused across calls and iterations,
    /// so this must not allocate in steady state.
    ///
    /// The default assumes every participant chooses from the same list, which
    /// is true of most simultaneous games — bidding, simultaneous card
    /// selection, rock-paper-scissors — and false of an asymmetric phase.
    /// Getting it wrong hands a player an action they do not own, and
    /// [`Game::apply_joint`] is where you find out.
    fn choices_for_into(&self, ctx: &Self::Context, player: u8, out: &mut Vec<Self::Choice>) {
        let _ = player;
        self.choices_into(ctx, out)
    }

    /// Play `choice`. Resolve any chance events with `rng`.
    fn apply_choice<R: Rng + ?Sized>(
        &mut self,
        ctx: &Self::Context,
        choice: &Self::Choice,
        rng: &mut R,
    );

    /// Resolve one simultaneous round: every participant's action at once.
    /// Resolve any chance events with `rng`.
    ///
    /// This is not [`Game::apply_choice`] in a loop. Applying the actions one
    /// after another is *serialization*, which hands the second mover a best
    /// response they do not have — it is a bound on the true value of the node,
    /// not an approximation of it, and a searcher built on it plays defensively
    /// as first mover and optimistically as second. There is deliberately no
    /// working default for that reason.
    fn apply_joint<R: Rng + ?Sized>(
        &mut self,
        _ctx: &Self::Context,
        _joint: JointChoices<'_, Self::Choice>,
        _rng: &mut R,
    ) {
        unimplemented!(
            "mcts: this game returns Status::Simultaneous but does not implement \
             Game::apply_joint"
        )
    }

    /// Play out from here and return the payoffs.
    ///
    /// The whole rollout, not one step: the game owns step granularity, its own
    /// step budget, its cutoff, and the evaluation applied at that cutoff. A
    /// per-move callback would forbid policies that resolve several game steps
    /// at once.
    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &Self::Context, rng: &mut R) -> Self::Rewards;

    // ---- cold path: called once per search, or once per iteration ----

    /// Build the single scratch state that every iteration determinizes into.
    /// Its contents are irrelevant; only its shape and capacity matter.
    fn new_buffer(&self) -> Self;

    /// Overwrite `dest` with a determinization of `self` from `perspective`.
    ///
    /// Writing into a caller-owned buffer (rather than returning `Self`) is what
    /// lets an implementation reuse its inner allocations via `clone_from`. For
    /// a perfect-information game this is `dest.clone_from(self)`.
    ///
    /// A determinization may change which actions are legal, and it may sample
    /// a world that is already over. It must not change *who* is to act: the
    /// tree records one node's statistics across every determinization that
    /// reaches it, so a node whose kind or participant set varies would have
    /// one player's arms read as another's, and debug builds assert against it
    /// during the descent.
    fn determinize_into<R: Rng + ?Sized>(
        &self,
        dest: &mut Self,
        ctx: &Self::Context,
        perspective: u8,
        rng: &mut R,
    );

    // ---- optional hooks ----

    /// Fast-forward past decisions the tree does not model, such as opponent
    /// moves resolved by a side model.
    ///
    /// Called once after each determinization and again after every applied
    /// choice during descent, including after [`Game::apply_joint`].
    ///
    /// The call on the root state must leave the decision being searched
    /// intact: the same kind of decision, for the same player or the same
    /// participants, offering the same choices. [`crate::Searcher::search`]
    /// reads the root player and the answer list from the state it was handed
    /// and builds the tree on the advanced one, so an `advance` that consumes
    /// the root's own decision returns a move belonging to somebody else — or
    /// one the position does not offer at all. Debug builds compare the root
    /// decision and its answer list across this call on every determinization,
    /// since an `advance` driven by a side model can consume the root's
    /// decision in some worlds and leave it standing in others.
    ///
    /// A game whose `advance` resolves opponent moves through a side model is
    /// doing something that overlaps semantically with simultaneous play, and
    /// it should pick one: modelling the opponents' concurrent choice here and
    /// *also* reporting [`Status::Simultaneous`] credits the same decision
    /// twice.
    fn advance<R: Rng + ?Sized>(
        &mut self,
        _ctx: &Self::Context,
        _side: &mut Self::Side,
        _perspective: u8,
        _rng: &mut R,
    ) {
    }

    /// Reset the side model at the start of a search.
    fn init_side(&self, _ctx: &Self::Context, side: &mut Self::Side) {
        *side = Default::default();
    }

    /// Start of an iteration, before determinization.
    fn begin_iteration(_side: &mut Self::Side) {}

    /// End of an iteration, after backpropagation. Credit anything the side
    /// model recorded during the descent.
    fn credit_iteration(_side: &mut Self::Side, _rewards: &Self::Rewards) {}

    /// Prior for progressive bias, evaluated once when a child is first visited.
    /// Only called when `Config::progressive_bias_weight` is non-zero.
    ///
    /// `self` is the successor state and `mover` is the player who chose the
    /// move that reached it — **not** the player the search runs for. Return a
    /// number on the same scale as [`Rewards::reward`]`(mover)`, because that
    /// is what it is added to: the crate scores every child by the reward of
    /// the player who moved into it (max^n backup), so a prior stated on any
    /// other player's scale is negated at every opponent node and the search
    /// spends its budget helping the opponent play badly. `mover` names no one
    /// at a terminal successor and cannot be recovered from the state in a game
    /// where one player moves twice, which is why it is passed rather than
    /// inferred.
    ///
    /// Never called at a simultaneous node. The prior is evaluated at the
    /// *successor* state, so it can only ever describe a joint child — and
    /// selection at a simultaneous node scores each player's own marginals, not
    /// the joint children. `Config::progressive_bias_weight` therefore has no
    /// effect there.
    fn heuristic_bias(&self, _ctx: &Self::Context, _mover: u8) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_set_has_no_members() {
        let set = PlayerSet::EMPTY;
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
        assert_eq!(set.iter().next(), None);
        assert_eq!(set.player_at(0), None);
        for player in 0..=PlayerSet::MAX_PLAYER {
            assert!(!set.contains(player));
            assert_eq!(set.slot_of(player), None);
        }
    }

    #[test]
    fn single_member_sets_round_trip() {
        for player in [0, 1, 7, 62, PlayerSet::MAX_PLAYER] {
            let set = PlayerSet::of([player]);
            assert_eq!(set.len(), 1);
            assert!(!set.is_empty());
            assert!(set.contains(player));
            assert_eq!(set.slot_of(player), Some(0));
            assert_eq!(set.player_at(0), Some(player));
            assert_eq!(set.player_at(1), None);
            assert_eq!(set.iter().collect::<Vec<_>>(), vec![player]);
        }
    }

    #[test]
    fn non_contiguous_set_round_trips() {
        let set = PlayerSet::of([0, 3, 7]);
        assert_eq!(set.len(), 3);
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![0, 3, 7]);
        assert_eq!(set.slot_of(0), Some(0));
        assert_eq!(set.slot_of(3), Some(1));
        assert_eq!(set.slot_of(7), Some(2));
        assert_eq!(set.player_at(0), Some(0));
        assert_eq!(set.player_at(1), Some(3));
        assert_eq!(set.player_at(2), Some(7));
        assert_eq!(set.player_at(3), None);
        for absent in [1, 2, 4, 5, 6, 8, PlayerSet::MAX_PLAYER] {
            assert!(!set.contains(absent));
            assert_eq!(set.slot_of(absent), None);
        }
    }

    #[test]
    fn slot_and_player_are_inverses_in_ascending_order() {
        for set in [
            PlayerSet::EMPTY,
            PlayerSet::of([0, 3, 7]),
            PlayerSet::of([PlayerSet::MAX_PLAYER]),
            PlayerSet::of([1, 2, 62, PlayerSet::MAX_PLAYER]),
            PlayerSet::first_n(8),
            PlayerSet::first_n(64),
        ] {
            assert_eq!(set.iter().count(), set.len());
            let mut previous = None;
            for (slot, player) in set.iter().enumerate() {
                assert_eq!(set.slot_of(player), Some(slot));
                assert_eq!(set.player_at(slot), Some(player));
                assert!(set.contains(player));
                assert!(previous.is_none_or(|last| last < player));
                previous = Some(player);
            }
            assert_eq!(set.player_at(set.len()), None);
        }
    }

    #[test]
    fn first_n_and_bits_agree() {
        assert_eq!(PlayerSet::first_n(0), PlayerSet::EMPTY);
        assert_eq!(PlayerSet::first_n(2), PlayerSet::of([0, 1]));
        assert_eq!(PlayerSet::first_n(64).len(), 64);
        assert_eq!(PlayerSet::first_n(64).bits(), u64::MAX);
        let set = PlayerSet::of([0, 3, 7]);
        assert_eq!(PlayerSet::from_bits(set.bits()), set);
        assert_eq!(PlayerSet::EMPTY.with(3).with(3), PlayerSet::of([3]));
    }

    #[test]
    #[should_panic(expected = "player indices 0..=63")]
    fn of_rejects_a_player_past_the_maximum() {
        PlayerSet::of([PlayerSet::MAX_PLAYER + 1]);
    }

    #[test]
    #[should_panic(expected = "player indices 0..=63")]
    fn with_rejects_a_player_past_the_maximum() {
        PlayerSet::EMPTY.with(64);
    }

    #[test]
    #[should_panic(expected = "player indices 0..=63")]
    fn first_n_rejects_more_players_than_fit() {
        PlayerSet::first_n(65);
    }

    #[test]
    fn joint_choices_views_the_arms_of_each_slot() {
        let players = PlayerSet::of([0, 3, 7]);
        // Slot 0 owns "a".."c", slot 1 owns "d".."e", slot 2 owns "f".
        let arm_choices = ["a", "b", "c", "d", "e", "f"];
        let starts = [0u32, 3, 5, 6];
        let picks = [2u32, 0, 0];
        let joint = JointChoices::new(players, &arm_choices, &starts, &picks);

        assert_eq!(joint.players(), players);
        assert_eq!(joint.len(), 3);
        assert!(!joint.is_empty());
        assert_eq!(joint.player_at(1), 3);
        assert_eq!(*joint.get(0), "c");
        assert_eq!(joint[1], "d");
        assert_eq!(joint.for_player(7), Some(&"f"));
        assert_eq!(joint.for_player(1), None);
        assert_eq!(
            joint.iter().collect::<Vec<_>>(),
            vec![(0, &"c"), (3, &"d"), (7, &"f")]
        );
    }
}
