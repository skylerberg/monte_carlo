use core::hash::Hash;
use core::ops::Range;

use hashbrown::HashTable;
use rand_core::Rng;

use crate::game::{PlayerSet, SimultaneousPolicy};
use crate::util::{below, hash_of};

/// A node in the search tree.
///
/// Children are owned inline in a contiguous `Vec`, which is what `select`
/// scans. `extra` is built lazily and holds whichever side table this node
/// needs: a hash index from a choice's hash to a position in `children`, so a
/// `Choice` is never stored twice, or the marginal bandits of a simultaneous
/// node. A node is never both, so the two share one allocation.
pub struct Node<C> {
    pub(crate) cumulative_reward: f64,
    /// Cached `ln(availability)`, refreshed only when `availability` changes —
    /// and only at a non-root parent, the only place `select` reads it.
    pub(crate) ln_availability: f64,
    pub(crate) children: Vec<Node<C>>,
    pub(crate) extra: Option<Box<Extra<C>>>,
    pub(crate) visits: u32,
    /// Iterations in which this choice was legal at its parent. The exploration
    /// term uses this rather than the parent's visit count, so a choice that is
    /// rarely legal is not penalised for being rarely tried.
    pub(crate) availability: u32,
    pub(crate) heuristic_bias: f32,
    /// The player whose reward `cumulative_reward` accumulates.
    pub(crate) player: u8,
    pub(crate) kind: NodeKind,
    /// `Some` only when `kind == NodeKind::Choice`.
    pub(crate) choice: Option<C>,
}

/// Which of the three things a node is.
///
/// Stored rather than inferred, because `choice.is_none()` used to mean "the
/// root" and a joint successor also has no single choice. Overloading one field
/// to answer two questions is what would make those two cases
/// indistinguishable. It costs nothing: measured, `Node<C>` is 64 / 72 / 80 /
/// 88 bytes for `C = u8 / u16 / usize / Vec<u16>` both with this byte and
/// without it — it lands in tail padding at every size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum NodeKind {
    /// The root. `choice` is `None` and there is no incoming edge.
    Root = 0,
    /// Reached by one player's single `Choice`, which `choice` holds.
    Choice = 1,
    /// Reached by one action per participant of a simultaneous parent. `choice`
    /// is `None`; the marginal indices live in the parent — see
    /// [`Node::joint_arm`].
    Joint = 2,
}

/// What leads to a [`Node`]: the incoming edge, with whatever it carries.
///
/// This exists because the obvious signature does not work. An
/// `Option<&Choice>` has one `None` and this has two distinct cases that carry
/// no choice — the root, which has no incoming edge, and a joint successor,
/// which is reached by one action per participant of a simultaneous parent.
/// Collapsing them means code that walks `children()` and skips `None`
/// silently skips every joint successor while reading as though it handled
/// them. Naming the cases makes that skip something a reader writes on purpose.
pub enum Edge<'a, C> {
    /// The root. There is no incoming edge.
    Root,
    /// Reached by one player's choice.
    Choice(&'a C),
    /// Reached by one action per participant of a simultaneous parent. The
    /// marginal indices live in the parent — see [`Node::joint_arm`].
    Joint,
}

// Hand-written rather than derived: `Edge` holds a reference, so it is `Copy`
// for every `C`, and deriving would bound the impls on `C: Copy` / `C: Clone`
// and take the trait away from exactly the heap-owning `Choice` types this
// crate is careful never to clone.
impl<C> Clone for Edge<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for Edge<'_, C> {}

impl<C: core::fmt::Debug> core::fmt::Debug for Edge<'_, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Root => f.write_str("Root"),
            Self::Joint => f.write_str("Joint"),
            Self::Choice(choice) => f.debug_tuple("Choice").field(choice).finish(),
        }
    }
}

impl<C: PartialEq> PartialEq for Edge<'_, C> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Root, Self::Root) | (Self::Joint, Self::Joint) => true,
            (Self::Choice(a), Self::Choice(b)) => a == b,
            _ => false,
        }
    }
}

impl<C: Eq> Eq for Edge<'_, C> {}

impl<'a, C> Edge<'a, C> {
    /// The choice, if this node was reached by exactly one.
    ///
    /// Deliberately not the way to walk `children()`: this `None` still merges
    /// the root and a joint successor, which is the conflation [`Edge`] exists
    /// to undo. Reach for it where you already know the kind, and match on the
    /// variants otherwise.
    pub fn choice(self) -> Option<&'a C> {
        match self {
            Self::Choice(choice) => Some(choice),
            Self::Root | Self::Joint => None,
        }
    }
}

/// A node's lazily-built side table. A node is either sequential or
/// simultaneous, never both, so the two possibilities share one allocation and
/// one pointer.
pub(crate) enum Extra<C> {
    /// Hash index over `children`, keyed by `Choice`, built once a sequential
    /// node grows past `Game::CHILD_INDEX_THRESHOLD` children.
    Index(ChildIndex),
    /// Present exactly at simultaneous nodes.
    Simul(Box<Simul<C>>),
}

pub(crate) struct ChildIndex {
    table: HashTable<u32>,
}

/// The marginal bandits and joint successors of a simultaneous node.
pub(crate) struct Simul<C> {
    /// Structure-of-arrays, deliberately. Selection reads only `arm_stats` and
    /// never touches a `Choice`; expansion matches `Choice`s and never touches
    /// the stats. At 32 bytes an arm's statistics are exactly half a cache
    /// line, so a pass over a 16-arm bandit touches 8 lines — an
    /// array-of-structs layout carrying the `Choice` inline would touch 16, and
    /// would stride unpredictably for a `Choice` that owns heap data.
    ///
    /// Every participant's arms, concatenated in slot (ascending player) order.
    /// Arms are only ever inserted at the end of their own slot, which is what
    /// makes a slot-relative arm index stable for the life of the node — the
    /// same argument that makes `path`'s child indices safe.
    pub(crate) arm_stats: Vec<ArmStats>,
    pub(crate) arm_choices: Vec<C>,
    /// Parallel to `arm_stats`. **Empty** under `SimultaneousPolicy::Duct`,
    /// which needs no policy state, so `Duct` pays 32 bytes an arm rather than
    /// 48.
    pub(crate) arm_policy: Vec<ArmPolicy>,
    /// Keys of the materialized joint successors, index-parallel to `children`.
    ///
    /// Held here rather than on each child so a linear scan walks a dense `u64`
    /// array — 8 candidates per cache line at any arity — instead of striding
    /// 64 to 88 bytes into `children` to reach a field.
    pub(crate) joint_keys: Vec<JointKey>,
    /// One index over *all* arms, hashed on `(slot, choice)` and holding an
    /// [`ArmRef`], built once the total arm count passes
    /// `Game::CHILD_INDEX_THRESHOLD`. One index rather than one per
    /// participant, because arms live in one flat array and a slot-tagged hash
    /// distinguishes them for free.
    pub(crate) arm_index: Option<ChildIndex>,
    pub(crate) joint_index: Option<ChildIndex>,
    pub(crate) players: PlayerSet,
    /// `starts[s] .. starts[s + 1]` is slot `s`'s arm range. `starts[arity]` is
    /// the total arm count. A fixed array rather than a `Vec` because arity is
    /// capped at `PlayerSet::MAX_SIMULTANEOUS`.
    pub(crate) starts: [u32; 9],
    pub(crate) arity: u8,
    /// Bits per participant in a `JointKey`: `64 / arity`.
    pub(crate) key_bits: u8,
}

/// One arm of one player's marginal bandit.
#[derive(Clone, Copy)]
pub(crate) struct ArmStats {
    /// Cumulative reward for this arm's own player, on the game's raw scale, so
    /// `Marginals::mean_reward` and `Duct` read a real reward.
    pub(crate) cumulative_reward: f64,
    /// Cached `ln(availability)`, the arm's ISMCTS exploration denominator.
    /// Maintained only under `SimultaneousPolicy::Duct`, which is the only
    /// reader; a mixing policy leaves it at zero rather than paying an `ln` per
    /// arm per visit. The field itself stays either way — dropping it would
    /// take `ArmStats` off its 32-byte stride, where `avail_epoch` rides the
    /// same line as the statistics.
    pub(crate) ln_availability: f64,
    pub(crate) visits: u32,
    /// Iterations at this node in which this action was legal for this player.
    pub(crate) availability: u32,
    /// The node's visit count at the last iteration in which this arm was
    /// legal.
    ///
    /// `avail_epoch == node.visits` *is* the legality test, so arms need no
    /// parallel availability buffer at all: the test rides on a word in the
    /// same 32-byte line as the statistics being read, instead of a second
    /// memory stream. Initialized to `u32::MAX`, which no reachable visit count
    /// equals.
    ///
    /// The test is only sound while `node.visits` still holds the value stamped
    /// during the descent, so backup **must credit a simultaneous node's arms
    /// before calling `Node::record` on the node itself**. See
    /// `back_propagate`.
    pub(crate) avail_epoch: u32,
}

impl Default for ArmStats {
    fn default() -> Self {
        Self {
            cumulative_reward: 0.0,
            ln_availability: 0.0,
            visits: 0,
            availability: 0,
            avail_epoch: u32::MAX,
        }
    }
}

/// Regret-matching state for one arm, allocated only under
/// `SimultaneousPolicy::RegretMatching`.
#[derive(Clone, Copy, Default)]
pub(crate) struct ArmPolicy {
    /// Cumulative regret for this arm, on the normalized `[0, 1]` reward scale.
    pub(crate) regret: f64,
    /// Sum, over iterations in which this arm was legal, of the probability the
    /// policy assigned it *before* the exploration floor was mixed in.
    ///
    /// Accumulated at selection time rather than reconstructed from visit
    /// counts afterwards, which makes exploration removal exact instead of an
    /// approximate subtract-and-renormalize. The published recipe subtracts
    /// `gamma * T / |A|`, which is only correct for a constant exploration rate
    /// over a constant legal set — here the rate decays and determinization
    /// varies the legal set, so the subtraction would be wrong in exactly the
    /// setting this crate runs in.
    pub(crate) strategy_sum: f64,
}

/// A joint successor edge, as the tuple of marginal indices that produced it,
/// packed at a fixed `64 / arity` bits per participant.
///
/// Fixed rather than derived from the live arm counts: arm lists grow as
/// determinizations reveal new actions, and a radix that grows with them makes
/// every previously-stored key silently alias a different tuple. A fixed radix
/// is stable for the same reason `path`'s child indices are — arms are only
/// ever appended to their own slot.
///
/// Eight bytes per joint edge and one FxHasher multiply to hash. Making the
/// joint action a `Choice` instead would clone one `Choice` per participant per
/// edge — for a choice that owns heap data, that is `arity` allocations and an
/// order of magnitude more bytes — and would force every game with both node
/// kinds into an `enum Choice`, inflating it at every *sequential* edge in the
/// tree as well.
///
/// Arms per participant are therefore capped by arity, and the cap is asserted
/// where an arm is created rather than where a key is packed, so the failure
/// names the player and the action count:
///
/// | participants | bits each | max arms per player |
/// |---:|---:|---:|
/// | 1 | 64 | 4 294 967 295 |
/// | 2 | 32 | 4 294 967 295 |
/// | 3 | 21 | 2 097 152 |
/// | 4 | 16 | 65 536 |
/// | 5 | 12 | 4 096 |
/// | 6 | 10 | 1 024 |
/// | 7 | 9 | 512 |
/// | 8 | 8 | 256 |
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub(crate) struct JointKey(u64);

impl JointKey {
    /// Largest arm index representable in one field.
    #[inline(always)]
    const fn field_mask(key_bits: u8) -> u64 {
        if key_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << key_bits) - 1
        }
    }

    /// Arms per participant at this arity: one more than the largest index the
    /// field holds. `u32::MAX` where the field is 32 bits or wider, since an
    /// arm index is a `u32` regardless.
    #[inline(always)]
    pub(crate) const fn capacity(key_bits: u8) -> u32 {
        if key_bits >= 32 {
            u32::MAX
        } else {
            1u32 << key_bits
        }
    }

    /// `picks[s]` is slot `s`'s **slot-relative** arm index, `0 .. slot_len(s)`.
    #[inline(always)]
    pub(crate) fn pack(picks: &[u32], key_bits: u8) -> Self {
        let mask = Self::field_mask(key_bits);
        let mut bits = 0u64;
        for (slot, &pick) in picks.iter().enumerate() {
            debug_assert!(
                pick as u64 <= mask,
                "mcts: arm index {pick} does not fit in {key_bits} bits"
            );
            bits |= (pick as u64) << (slot * key_bits as usize);
        }
        Self(bits)
    }

    /// The **slot-relative** arm index for `slot`. The global position in
    /// `Simul::arm_stats` is `starts[slot] + arm(slot, key_bits)`; that
    /// conversion is the only place the two numberings meet.
    #[inline(always)]
    pub(crate) fn arm(self, slot: usize, key_bits: u8) -> u32 {
        ((self.0 >> (slot * key_bits as usize)) & Self::field_mask(key_bits)) as u32
    }
}

/// What the arm index stores: an arm's slot, and its position **within** that
/// slot.
///
/// A global position would not survive `Simul::grow_slot`, which inserts in the
/// middle of the flat arm arrays and moves every later arm along — the same
/// reason a [`JointKey`] holds slot-relative indices. Three bits of slot,
/// because arity is capped at [`PlayerSet::MAX_SIMULTANEOUS`]; the remaining 29
/// are more than a `JointKey` can address at arity 3 and above, which
/// `grow_slot` asserts, and more arms at arity 1 or 2 than the arm array could
/// hold in memory — 2^29 of them is 16 GB of `ArmStats` alone.
#[derive(Clone, Copy)]
struct ArmRef(u32);

impl ArmRef {
    const SLOT_BITS: u32 = 3;
    const SLOT_MASK: u32 = (1 << Self::SLOT_BITS) - 1;

    #[inline(always)]
    fn new(slot: usize, relative: usize) -> Self {
        debug_assert!(slot < PlayerSet::MAX_SIMULTANEOUS);
        debug_assert!(relative < (1 << (32 - Self::SLOT_BITS)));
        Self(((relative as u32) << Self::SLOT_BITS) | slot as u32)
    }

    #[inline(always)]
    fn slot(self) -> usize {
        (self.0 & Self::SLOT_MASK) as usize
    }

    #[inline(always)]
    fn relative(self) -> usize {
        (self.0 >> Self::SLOT_BITS) as usize
    }
}

/// The key the arm index hashes: an arm is identified by its slot and its
/// choice, so one index serves every participant.
#[inline]
fn arm_hash<C: Hash>(slot: usize, choice: &C) -> u64 {
    hash_of(&(slot as u32, choice))
}

impl<C> Node<C> {
    pub(crate) fn new_root(player: u8) -> Self {
        Self::new(player, NodeKind::Root, None)
    }

    fn new(player: u8, kind: NodeKind, choice: Option<C>) -> Self {
        Self {
            cumulative_reward: 0.0,
            ln_availability: 0.0,
            children: Vec::new(),
            extra: None,
            visits: 0,
            availability: 1,
            heuristic_bias: 0.0,
            player,
            kind,
            choice,
        }
    }

    #[inline(always)]
    pub(crate) fn is_root(&self) -> bool {
        matches!(self.kind, NodeKind::Root)
    }

    /// Times this node was visited.
    pub fn visits(&self) -> u32 {
        self.visits
    }

    /// Iterations at the parent that offered the choice leading here, which is
    /// the denominator [`Node::visits`] is a numerator of.
    ///
    /// A child can only be selected on an iteration whose determinization made
    /// it legal, so `visits <= availability` always and the ratio — not the
    /// visit count — is how often the search took the choice given the chance.
    /// That ratio is how a root's answer is ranked, and this is what lets a
    /// caller reproduce or log the ranking from [`crate::Searcher::tree`].
    /// Read it on a child. A root has no parent to have been offered by, so its
    /// own count is either the 1 it was created with or whatever it carried up
    /// from the node it was promoted from.
    pub fn availability(&self) -> u32 {
        self.availability
    }

    /// Mean reward for the player this node accumulates for, or 0.0 if never
    /// visited. See [`Node::reward_player`].
    pub fn mean_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.cumulative_reward / self.visits as f64
        }
    }

    /// Total reward accumulated for [`Node::reward_player`].
    pub fn cumulative_reward(&self) -> f64 {
        self.cumulative_reward
    }

    /// What leads to this node.
    ///
    /// Three cases, named, because the two that carry no choice are not the
    /// same case: the root has no incoming edge at all, and a joint successor
    /// is reached by one action per participant rather than by any single
    /// choice. Use [`Edge::choice`] only where you already know which kind you
    /// hold.
    pub fn edge(&self) -> Edge<'_, C> {
        match self.kind {
            NodeKind::Root => Edge::Root,
            NodeKind::Joint => Edge::Joint,
            NodeKind::Choice => Edge::Choice(
                self.choice
                    .as_ref()
                    .expect("a node reached by a choice carries it"),
            ),
        }
    }

    /// This node's expanded children. At a simultaneous node these are the
    /// joint successors that have actually been visited, not every possible
    /// one.
    pub fn children(&self) -> &[Node<C>] {
        &self.children
    }

    /// Nodes in this subtree, including this one. Marginal bandit arms are
    /// statistics, not nodes, so they are not counted here or by
    /// [`Node::max_depth`].
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(Node::node_count).sum::<usize>()
    }

    /// Longest root-to-leaf path in this subtree, in nodes.
    pub fn max_depth(&self) -> usize {
        1 + self.children.iter().map(Node::max_depth).max().unwrap_or(0)
    }

    /// Which of the three things this node is.
    pub fn kind(&self) -> NodeKind {
        self.kind
    }

    /// The player whose reward [`Node::cumulative_reward`] accumulates.
    ///
    /// For a node reached by one player's choice that is the player who moved
    /// here, which is the max^n rule. A joint successor has no single mover, so
    /// it accumulates the reward of the player the search is running on behalf
    /// of. The root has no incoming edge at all and accumulates the reward of
    /// the player **to act**: the perspective player at a simultaneous root,
    /// where the search requires the two to coincide, but at a sequential root
    /// whichever player the position says is on turn — which is a different
    /// player from the perspective one whenever the search is running an
    /// opponent model.
    pub fn reward_player(&self) -> u8 {
        self.player
    }

    /// The participants, if this is a simultaneous decision node.
    pub fn simultaneous_players(&self) -> Option<PlayerSet> {
        self.simul().map(|simul| simul.players)
    }

    /// `player`'s bandit over their own actions here, or `None` if this is not
    /// a simultaneous node or `player` does not act at it.
    pub fn marginals(&self, player: u8) -> Option<Marginals<'_, C>> {
        let simul = self.simul()?;
        let slot = simul.players.slot_of(player)?;
        Some(simul.marginals(slot))
    }

    /// The index, into `self.marginals(player)`, of the action `player` took to
    /// reach `self.children()[child]`.
    ///
    /// `None` if this is not a simultaneous node, if `player` does not act
    /// here, or if `child` is out of range.
    pub fn joint_arm(&self, child: usize, player: u8) -> Option<usize> {
        let simul = self.simul()?;
        let slot = simul.players.slot_of(player)?;
        let key = *simul.joint_keys.get(child)?;
        Some(key.arm(slot, simul.key_bits) as usize)
    }

    #[inline(always)]
    pub(crate) fn record(&mut self, reward: f64) {
        self.cumulative_reward += reward;
        self.visits += 1;
    }

    #[inline(always)]
    pub(crate) fn simul(&self) -> Option<&Simul<C>> {
        match self.extra.as_deref() {
            Some(Extra::Simul(simul)) => Some(simul),
            _ => None,
        }
    }

    #[inline(always)]
    pub(crate) fn simul_mut(&mut self) -> Option<&mut Simul<C>> {
        match self.extra.as_deref_mut() {
            Some(Extra::Simul(simul)) => Some(simul),
            _ => None,
        }
    }

    /// The marginal block of a node that reported
    /// [`crate::Status::Simultaneous`], created empty on the first visit.
    ///
    /// `kind` is left alone: it describes the *incoming* edge, and simultaneity
    /// is a property of this block's presence, which is why [`NodeKind`] has no
    /// simultaneous variant.
    pub(crate) fn ensure_simul(&mut self, players: PlayerSet) -> &mut Simul<C> {
        if self.simul().is_none() {
            debug_assert!(
                self.extra.is_none(),
                "mcts: this node's Choice-keyed child index is being replaced by \
                 marginal bandits. A node's kind must not vary across \
                 determinizations."
            );
            self.extra = Some(Box::new(Extra::Simul(Box::new(Simul::new(players)))));
        }
        self.simul_mut()
            .expect("the simultaneous block was just installed")
    }

    /// The invariants that let selection and backup index the arm arrays
    /// without bounds reasoning. Checked after every expansion pass, the way
    /// `select` checks `avail` against `children`.
    #[inline]
    pub(crate) fn debug_check_arms(&self) {
        if let Some(simul) = self.simul() {
            debug_assert_eq!(
                simul.arm_stats.len(),
                simul.arm_choices.len(),
                "mcts: arm statistics and arm choices are out of step"
            );
            debug_assert!(
                simul.arm_policy.is_empty() || simul.arm_policy.len() == simul.arm_stats.len(),
                "mcts: arm policy state is out of step with arm statistics"
            );
            debug_assert_eq!(
                simul.starts[simul.arity as usize] as usize,
                simul.arm_stats.len(),
                "mcts: slot boundaries do not cover the arm array"
            );
            debug_assert_eq!(
                simul.joint_keys.len(),
                self.children.len(),
                "mcts: joint keys and joint children are out of step"
            );
        }
    }

    /// Bump every arm's availability without enumerating a single move.
    ///
    /// The `Game::ROOT_CHOICES_INVARIANT` fast path at a simultaneous root:
    /// move generation is what the invariant saves, and the stamp is
    /// `sum(|A_i|)` word writes, the same order as the sequential path's
    /// `root_avail.resize` — which it would not be if it issued a libm call per
    /// arm on top, so the cached logarithm is refreshed only for the policy
    /// that reads it.
    pub(crate) fn restamp_marginals(&mut self, policy: SimultaneousPolicy) {
        let epoch = self.visits;
        let caches_ln = policy.reads_ln_availability();
        let simul = self
            .simul_mut()
            .expect("mcts: only a simultaneous node has marginals to restamp");
        for arm in &mut simul.arm_stats {
            arm.availability += 1;
            if caches_ln {
                arm.ln_availability = (arm.availability as f64).ln();
            }
            arm.avail_epoch = epoch;
        }
        self.debug_check_arms();
    }

    /// Push a joint successor for `key` and return its child index.
    ///
    /// A joint successor is an ordinary node with no incoming `Choice`; the
    /// tuple of marginal indices that reached it stays in the parent, where a
    /// lookup scans it densely.
    pub(crate) fn push_joint(&mut self, key: JointKey, player: u8, threshold: usize) -> usize {
        let position = self.children.len();
        self.children.push(Node::new(player, NodeKind::Joint, None));

        let simul = self
            .simul_mut()
            .expect("mcts: a joint successor belongs to a simultaneous node");
        simul.joint_keys.push(key);

        if simul.joint_index.is_none() {
            if simul.joint_keys.len() > threshold {
                simul.build_joint_index();
            }
            return position;
        }

        let Simul {
            joint_keys,
            joint_index,
            ..
        } = simul;
        if let Some(index) = joint_index {
            let _ = index
                .table
                .insert_unique(hash_of(&key), position as u32, |&i| {
                    hash_of(&joint_keys[i as usize])
                });
        }
        position
    }

    /// Re-root at the joint successor reached by `key`, keeping that subtree
    /// and discarding the rest. Returns false and leaves `self` untouched if no
    /// such successor was ever materialized.
    ///
    /// The promoted node keeps its own marginals: it is the next simultaneous
    /// decision, and its arms are statistics about actions the players have not
    /// taken yet.
    pub(crate) fn reroot_at_joint(&mut self, key: JointKey) -> bool {
        let Some(simul) = self.simul() else {
            return false;
        };
        let Some(i) = simul.find_joint(key) else {
            return false;
        };
        let mut child = self.children.swap_remove(i);
        child.kind = NodeKind::Root;
        child.choice = None;
        child.availability = 1;
        child.ln_availability = 0.0;
        *self = child;
        true
    }
}

impl<C: Clone + Eq + Hash> Node<C> {
    #[inline]
    pub(crate) fn find_child(&self, choice: &C) -> Option<usize> {
        match self.child_index() {
            None => self
                .children
                .iter()
                .position(|child| child.choice.as_ref() == Some(choice)),
            Some(index) => {
                let hash = hash_of(choice);
                index
                    .table
                    .find(hash, |&i| {
                        self.children[i as usize].choice.as_ref() == Some(choice)
                    })
                    .map(|&i| i as usize)
            }
        }
    }

    #[inline(always)]
    fn child_index(&self) -> Option<&ChildIndex> {
        match self.extra.as_deref() {
            Some(Extra::Index(index)) => Some(index),
            _ => None,
        }
    }

    fn push_child(&mut self, choice: C, player: u8, threshold: usize) {
        debug_assert!(
            self.simul().is_none(),
            "mcts: the Choice-keyed child index is only ever built on a sequential \
             node; a joint successor has no single choice to key on"
        );
        let hash = hash_of(&choice);
        let position = self.children.len() as u32;
        self.children
            .push(Node::new(player, NodeKind::Choice, Some(choice)));

        if self.extra.is_none() {
            if self.children.len() > threshold {
                self.build_index();
            }
            return;
        }

        let Self {
            children, extra, ..
        } = self;
        if let Some(Extra::Index(index)) = extra.as_deref_mut() {
            let _ = index.table.insert_unique(hash, position, |&i| {
                hash_of(
                    children[i as usize]
                        .choice
                        .as_ref()
                        .expect("a sequential node's children all carry a choice"),
                )
            });
        }
    }

    fn build_index(&mut self) {
        debug_assert!(
            self.simul().is_none(),
            "mcts: the Choice-keyed child index is only ever built on a sequential \
             node; a joint successor has no single choice to key on"
        );
        let children: &[Node<C>] = &self.children;
        let mut table = HashTable::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            let hash = hash_of(
                child
                    .choice
                    .as_ref()
                    .expect("a sequential node's children all carry a choice"),
            );
            let _ = table.insert_unique(hash, i as u32, |&j| {
                hash_of(
                    children[j as usize]
                        .choice
                        .as_ref()
                        .expect("a sequential node's children all carry a choice"),
                )
            });
        }
        self.extra = Some(Box::new(Extra::Index(ChildIndex { table })));
    }

    /// Record this iteration's legal choices, and grow the tree by one child.
    ///
    /// Every already-present child that is legal now has its availability count
    /// bumped, and `avail[i]` is set for it. The root creates a child for every
    /// legal choice; other nodes create exactly one, reservoir-sampled uniformly
    /// from the choices they have not seen yet. Sampling rather than shuffling
    /// matters: shuffling a five-figure choice list on every visit costs more
    /// than the rest of the iteration put together.
    pub(crate) fn expand<R: Rng + ?Sized>(
        &mut self,
        choices: &[C],
        player: u8,
        threshold: usize,
        avail: &mut Vec<bool>,
        rng: &mut R,
    ) {
        avail.clear();
        avail.resize(self.children.len(), false);

        let is_root = self.is_root();
        if is_root && self.children.is_empty() {
            self.children.reserve(choices.len());
        }

        let mut unseen: u64 = 0;
        let mut sampled = usize::MAX;

        for (i, choice) in choices.iter().enumerate() {
            match self.find_child(choice) {
                // A duplicate within `choices` must not be counted twice.
                Some(k) if avail[k] => {}
                Some(k) => {
                    let child = &mut self.children[k];
                    child.availability += 1;
                    // Only a non-root parent scores its children against their
                    // own availability: at a root `select` uses the root's
                    // visit count, and `reroot_at` resets the field on the one
                    // promotion that could make a root child read it again. So
                    // a root child pays the increment, which the root ranking
                    // divides by, and not the `ln` for a denominator nothing
                    // reads.
                    if !is_root {
                        child.ln_availability = (child.availability as f64).ln();
                    }
                    avail[k] = true;
                }
                None if is_root => {
                    self.push_child(choice.clone(), player, threshold);
                    avail.push(true);
                }
                None => {
                    unseen += 1;
                    if below(rng, unseen) == 0 {
                        sampled = i;
                    }
                }
            }
        }

        if !is_root && sampled != usize::MAX {
            self.push_child(choices[sampled].clone(), player, threshold);
            avail.push(true);
        }
    }

    /// Record this iteration's legal actions for one participant, creating an
    /// arm for every action this node has not seen before. Returns how many of
    /// this slot's arms are legal now; zero means no joint action exists.
    ///
    /// **Full, on every visit, root or not.** Joint successors stay strictly
    /// lazy, but the marginals never are. Progressive expansion exists because
    /// a node with a five-figure *choice list* makes full expansion quadratic;
    /// under a decoupled bandit the quantity that explodes is `prod(|A_i|)` —
    /// the joint successors — and the quantity that is cheap is `sum(|A_i|)`,
    /// which is the entire reason to decouple. Expanding the arms progressively
    /// would optimize the term that was already cheap, and it would cost
    /// correctness twice: an arm created at visit `v` has `availability == 1`,
    /// so `ln_availability == 0` and no exploration bonus at all, while its
    /// siblings compete on a full history — and every mixed policy normalizes
    /// over a support, so a support that keeps growing makes the time-averaged
    /// strategy an average over incomparable supports, which is not an
    /// approximate equilibrium over anything.
    pub(crate) fn expand_marginals(
        &mut self,
        slot: usize,
        choices: &[C],
        policy: SimultaneousPolicy,
        threshold: usize,
    ) -> usize {
        let epoch = self.visits;
        let policy_in_use = policy.mixes();
        let caches_ln = policy.reads_ln_availability();
        let simul = self
            .simul_mut()
            .expect("mcts: marginals belong to a simultaneous node");

        // `grow_slot` is the sole builder, and holds the invariant: an arm set
        // past the threshold has an index. Building here as well would be a
        // second copy of the threshold test that can never fire — a node holds
        // no arms before its first growth, and any growth that crosses the
        // threshold leaves an index behind.
        debug_assert!(
            simul.arm_index.is_some() || simul.arm_choices.len() <= threshold,
            "mcts: an arm set past the index threshold has no index"
        );
        if simul.slot_len(slot) == 0 {
            simul.reserve_slot(choices.len(), policy_in_use);
        }

        let mut legal = 0;
        for choice in choices {
            match simul.find_arm(slot, choice) {
                // A duplicate within the enumeration must not be counted twice.
                Some(a) if simul.arm_stats[a].avail_epoch == epoch => {}
                Some(a) => {
                    let arm = &mut simul.arm_stats[a];
                    arm.availability += 1;
                    if caches_ln {
                        arm.ln_availability = (arm.availability as f64).ln();
                    }
                    arm.avail_epoch = epoch;
                    legal += 1;
                }
                None => {
                    simul.grow_slot(slot, choice, epoch, policy_in_use, threshold);
                    legal += 1;
                }
            }
        }

        self.debug_check_arms();
        legal
    }
}

impl<C> Simul<C> {
    fn new(players: PlayerSet) -> Self {
        let arity = players.len();
        debug_assert!(
            arity > 0 && arity <= PlayerSet::MAX_SIMULTANEOUS,
            "mcts: a simultaneous node has between 1 and {} participants",
            PlayerSet::MAX_SIMULTANEOUS
        );
        Self {
            arm_stats: Vec::new(),
            arm_choices: Vec::new(),
            arm_policy: Vec::new(),
            joint_keys: Vec::new(),
            arm_index: None,
            joint_index: None,
            players,
            starts: [0; 9],
            arity: arity as u8,
            key_bits: (64 / arity) as u8,
        }
    }

    /// Where slot `s`'s arms live in the flat arm arrays.
    #[inline(always)]
    pub(crate) fn slot_range(&self, slot: usize) -> Range<usize> {
        self.starts[slot] as usize..self.starts[slot + 1] as usize
    }

    /// How many actions this participant has been seen to have.
    #[inline(always)]
    pub(crate) fn slot_len(&self, slot: usize) -> usize {
        (self.starts[slot + 1] - self.starts[slot]) as usize
    }

    /// One participant's bandit, as the borrowing view callers see.
    pub(crate) fn marginals(&self, slot: usize) -> Marginals<'_, C> {
        let range = self.slot_range(slot);
        let policy = if self.arm_policy.is_empty() {
            &[][..]
        } else {
            &self.arm_policy[range.clone()]
        };
        Marginals {
            stats: &self.arm_stats[range.clone()],
            choices: &self.arm_choices[range],
            policy,
            mixes: !self.arm_policy.is_empty(),
        }
    }

    /// The joint successor reached by `key`, if it has been materialized.
    pub(crate) fn find_joint(&self, key: JointKey) -> Option<usize> {
        match &self.joint_index {
            None => self.joint_keys.iter().position(|&stored| stored == key),
            Some(index) => index
                .table
                .find(hash_of(&key), |&i| self.joint_keys[i as usize] == key)
                .map(|&i| i as usize),
        }
    }

    fn build_joint_index(&mut self) {
        let keys: &[JointKey] = &self.joint_keys;
        let mut table = HashTable::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            let _ = table.insert_unique(hash_of(key), i as u32, |&j| hash_of(&keys[j as usize]));
        }
        self.joint_index = Some(ChildIndex { table });
    }

    /// Make room for a slot's first enumeration, so the arrays never regrow for
    /// arms that were present on the first visit.
    fn reserve_slot(&mut self, arms: usize, policy_in_use: bool) {
        self.arm_stats.reserve_exact(arms);
        self.arm_choices.reserve_exact(arms);
        if policy_in_use {
            self.arm_policy.reserve_exact(arms);
        }
    }
}

impl<C: Clone + Eq + Hash> Simul<C> {
    /// The global position an [`ArmRef`] names.
    #[inline(always)]
    fn arm_at(&self, arm: ArmRef) -> usize {
        self.starts[arm.slot()] as usize + arm.relative()
    }

    /// The global position of `slot`'s arm for `choice`, if it has one.
    pub(crate) fn find_arm(&self, slot: usize, choice: &C) -> Option<usize> {
        let range = self.slot_range(slot);
        match &self.arm_index {
            None => {
                let start = range.start;
                self.arm_choices[range]
                    .iter()
                    .position(|stored| stored == choice)
                    .map(|i| start + i)
            }
            Some(index) => index
                .table
                .find(arm_hash(slot, choice), |&packed| {
                    let arm = ArmRef(packed);
                    arm.slot() == slot && &self.arm_choices[self.arm_at(arm)] == choice
                })
                .map(|&packed| self.arm_at(ArmRef(packed))),
        }
    }

    fn build_arm_index(&mut self) {
        let Simul {
            arm_choices,
            arm_index,
            starts,
            arity,
            ..
        } = self;
        let choices: &[C] = arm_choices;
        let mut table = HashTable::with_capacity(choices.len());
        for slot in 0..*arity as usize {
            let start = starts[slot] as usize;
            for (relative, choice) in choices[start..starts[slot + 1] as usize].iter().enumerate() {
                let _ = table.insert_unique(
                    arm_hash(slot, choice),
                    ArmRef::new(slot, relative).0,
                    |&packed| {
                        let other = ArmRef(packed);
                        arm_hash(
                            other.slot(),
                            &choices[starts[other.slot()] as usize + other.relative()],
                        )
                    },
                );
            }
        }
        *arm_index = Some(ChildIndex { table });
    }

    /// Append an arm to slot `slot`, in the middle of the flat arrays.
    ///
    /// The new arm takes the next **slot-relative** index, and every other
    /// slot's arms keep theirs, so no stored [`JointKey`] has to be remapped —
    /// which is the whole reason a key holds slot-relative indices rather than
    /// global positions.
    fn grow_slot(
        &mut self,
        slot: usize,
        choice: &C,
        epoch: u32,
        policy_in_use: bool,
        threshold: usize,
    ) {
        let at = self.starts[slot + 1] as usize;
        let new_len = at - self.starts[slot] as usize + 1;
        let capacity = JointKey::capacity(self.key_bits);
        assert!(
            new_len <= capacity as usize,
            "mcts: player {} has {new_len} legal actions at a simultaneous node with \
             {} participants, where at most {capacity} fit in a joint key. Narrow the \
             action set, or split the decision — with {} players the joint successor \
             space is already unreachable.",
            self.players.player_at(slot).unwrap_or(u8::MAX),
            self.arity,
            self.arity
        );

        self.arm_stats.insert(
            at,
            ArmStats {
                availability: 1,
                avail_epoch: epoch,
                ..Default::default()
            },
        );
        self.arm_choices.insert(at, choice.clone());
        if policy_in_use {
            self.arm_policy.insert(at, ArmPolicy::default());
        }
        let arity = self.arity as usize;
        for start in &mut self.starts[slot + 1..=arity] {
            *start += 1;
        }

        // The index survives the insert: every payload is slot-relative, this
        // slot's existing arms keep their positions and the new one takes the
        // next, so only the new arm has to be added. Discarding the index
        // instead would make a node's expansion quadratic in its arm count —
        // the first visit of a wide node rebuilds it once per action and scans
        // linearly in between, and under determinization every action a later
        // determinization reveals does it again.
        if self.arm_index.is_none() {
            // The only place the index is ever built. A node holds no arms
            // until its first expansion pass grows them, so growth is the only
            // thing that can take an arm set past the threshold.
            if self.arm_choices.len() > threshold {
                self.build_arm_index();
            }
            return;
        }
        let Simul {
            arm_choices,
            arm_index,
            starts,
            ..
        } = self;
        if let Some(index) = arm_index {
            let choices: &[C] = arm_choices;
            let _ = index.table.insert_unique(
                arm_hash(slot, &choices[at]),
                ArmRef::new(slot, at - starts[slot] as usize).0,
                |&packed| {
                    let other = ArmRef(packed);
                    arm_hash(
                        other.slot(),
                        &choices[starts[other.slot()] as usize + other.relative()],
                    )
                },
            );
        }
    }
}

impl<C: Clone + Eq + Hash> Node<C> {
    /// Re-root at `choice`, keeping that subtree and discarding the rest.
    /// Returns false and leaves `self` untouched if `choice` is not a child.
    pub(crate) fn reroot_at(&mut self, choice: &C) -> bool {
        let Some(i) = self.find_child(choice) else {
            return false;
        };
        let mut child = self.children.swap_remove(i);
        child.kind = NodeKind::Root;
        child.choice = None;
        child.availability = 1;
        child.ln_availability = 0.0;
        *self = child;
        true
    }
}

/// One player's bandit over their own actions at a simultaneous node.
///
/// Arms are statistics, not nodes: an arm owns no children and no successor of
/// its own, and storing them as `Node`s would silently change what
/// [`Node::node_count`] and [`Node::max_depth`] report.
pub struct Marginals<'a, C> {
    stats: &'a [ArmStats],
    choices: &'a [C],
    /// Empty unless the game runs a mixing policy, which is also what `mixes`
    /// reports.
    policy: &'a [ArmPolicy],
    mixes: bool,
}

// Hand-written: `derive` would demand `C: Copy`, and the view holds nothing but
// shared slices.
impl<C> Clone for Marginals<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for Marginals<'_, C> {}

impl<'a, C> Marginals<'a, C> {
    /// Number of actions this player has been seen to have here.
    pub fn len(&self) -> usize {
        self.stats.len()
    }

    /// Whether this player has no actions here. Never true at a well-formed
    /// simultaneous node.
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }

    /// The action this arm plays.
    pub fn choice(&self, arm: usize) -> &'a C {
        &self.choices[arm]
    }

    /// Iterations in which this arm was selected.
    pub fn visits(&self, arm: usize) -> u32 {
        self.stats[arm].visits
    }

    /// Iterations at this node in which this action was legal for this player.
    ///
    /// This, and not the node's visit count, is what an arm is measured
    /// against, so an action that is rarely legal is not mistaken for an
    /// under-explored one: [`crate::SimultaneousPolicy::Duct`] takes its
    /// exploration denominator from it, and the average strategy
    /// [`Marginals::policy_into`] reports divides by it. It holds at every
    /// simultaneous node, a root included — which is where an arm parts company
    /// with a child, since a root's children are scored against the root's own
    /// visit count and only the levels below it use this rule.
    pub fn availability(&self, arm: usize) -> u32 {
        self.stats[arm].availability
    }

    /// Mean reward of this arm for its own player, or 0.0 if never selected.
    pub fn mean_reward(&self, arm: usize) -> f64 {
        let arm = &self.stats[arm];
        if arm.visits == 0 {
            0.0
        } else {
            arm.cumulative_reward / arm.visits as f64
        }
    }

    /// Index of the arm this player's search settled on, first maximum so ties
    /// break deterministically. `None` if no arm has been selected yet.
    ///
    /// The best mean reward, not the most selections: an arm is selectable only
    /// on the iterations it was legal, so raw counts rank an arm the
    /// determinizations rarely dealt below one they dealt constantly, however
    /// much better it is. An arm the search sampled too few times for its mean
    /// to mean anything is ranked below every arm that cleared the bar, and the
    /// selection rate breaks ties between equal means.
    /// [`Marginals::most_visited`] is the raw count.
    ///
    /// The right answer only where the equilibrium is pure. See
    /// [`Marginals::policy_into`].
    pub fn leader(&self) -> Option<usize> {
        crate::duct::leader_over(self.stats, None)
    }

    /// Index of the most-selected arm, first maximum so ties break
    /// deterministically. `None` if no arm has been selected yet.
    ///
    /// The raw count, for inspection. It is *not* what a root move is read off:
    /// see [`Marginals::leader`].
    pub fn most_visited(&self) -> Option<usize> {
        let mut best = None;
        let mut best_visits = 0;
        for (i, arm) in self.stats.iter().enumerate() {
            if arm.visits > best_visits {
                best_visits = arm.visits;
                best = Some(i);
            }
        }
        best
    }

    /// Write this player's mixed strategy into `out`, parallel to the arms and
    /// summing to 1. `out` is cleared first, and is a caller buffer so polling
    /// this every turn costs nothing after the first.
    ///
    /// Under [`crate::SimultaneousPolicy::RegretMatching`] this is the
    /// time-averaged strategy with the exploration floor already excluded and
    /// availability divided out — the only extraction with a convergence result
    /// behind it, and the one to sample. Under
    /// [`crate::SimultaneousPolicy::Duct`] it is one-hot at [`Marginals::leader`],
    /// deliberately: decoupled UCB1 converges to a pure policy, and its
    /// visit distribution is precisely the object an opponent exploits, so this
    /// crate will not hand it back dressed as a mixed strategy.
    ///
    /// It is averaged over every determinization in which each action was
    /// legal, which is a wider action set than any single position offers. A
    /// caller holding a narrower legal set must not zero this vector's illegal
    /// entries and renormalize: that is sound under a mixing policy and unsound
    /// under [`crate::SimultaneousPolicy::Duct`], where the vector is one-hot
    /// at the leader over *every* arm the tree holds, so an illegal leader
    /// leaves nothing but zeros to divide by. Use
    /// [`Marginals::policy_masked_into`], which re-runs the extraction over the
    /// legal arms instead of rescaling this one — the same thing
    /// [`crate::Searcher::root_policy_into`] does at a root, where it can also
    /// build the mask from the position itself.
    pub fn policy_into(&self, out: &mut Vec<f64>) {
        crate::duct::strategy_into(self.stats, self.policy, None, self.mixes, out);
    }

    /// [`Marginals::policy_into`] restricted to the arms legal in the real
    /// position, and normalized over them.
    ///
    /// `legal` must be parallel to the arms — a mask of any other length panics
    /// — and is built by looking this player's legal actions up against
    /// [`Marginals::choice`]. `out` comes back the same length as the arms with
    /// a zero at every illegal one, summing to 1 unless nothing is legal, in
    /// which case it is all zeros and the tree has no move to offer here.
    ///
    /// This is the extraction to use whenever the arms outrun the position,
    /// which determinization makes the normal case, and it is the only sound
    /// one under [`crate::SimultaneousPolicy::Duct`]: the leader is recomputed
    /// over the legal arms rather than found first and masked away afterwards.
    /// At a root, prefer [`crate::Searcher::root_policy_into`], which
    /// enumerates the mask for you and hands back `(choice, probability)`
    /// pairs; below one, this is the whole extraction. The two run the same
    /// body over the same arms, so they are one extraction rather than two that
    /// agree.
    pub fn policy_masked_into(&self, legal: &[bool], out: &mut Vec<f64>) {
        assert_eq!(
            legal.len(),
            self.stats.len(),
            "mcts: the legality mask and this player's arms are out of step"
        );
        crate::duct::strategy_into(self.stats, self.policy, Some(legal), self.mixes, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;
    use wyrand::WyRand;

    const RM: SimultaneousPolicy = SimultaneousPolicy::RegretMatching;
    const DUCT: SimultaneousPolicy = SimultaneousPolicy::Duct;

    fn simul_of(players: PlayerSet, slots: &[&[u16]], policy_in_use: bool) -> Simul<u16> {
        let mut simul = Simul::new(players);
        for (slot, choices) in slots.iter().enumerate() {
            for choice in *choices {
                simul.grow_slot(slot, choice, 0, policy_in_use, usize::MAX);
            }
        }
        simul
    }

    /// The sizes every doc comment in this file cites. A `Node` that grew is a
    /// cache-line regression the benches would only report as noise.
    #[test]
    fn node_layout_is_unchanged() {
        assert_eq!(size_of::<Node<u8>>(), 64);
        assert_eq!(size_of::<Node<u16>>(), 72);
        assert_eq!(size_of::<Node<u32>>(), 72);
        assert_eq!(size_of::<Node<usize>>(), 80);
        assert_eq!(size_of::<Node<Vec<u16>>>(), 88);

        assert_eq!(size_of::<ChildIndex>(), 32);
        assert_eq!(size_of::<Extra<u8>>(), 32);
        assert_eq!(size_of::<Extra<usize>>(), 32);
        assert_eq!(size_of::<Extra<Vec<u16>>>(), 32);

        assert_eq!(size_of::<Simul<u8>>(), 208);
        assert_eq!(size_of::<Simul<Vec<u16>>>(), 208);
        assert_eq!(size_of::<ArmStats>(), 32);
        assert_eq!(size_of::<ArmPolicy>(), 16);
        assert_eq!(size_of::<JointKey>(), 8);
    }

    #[test]
    fn a_joint_key_round_trips_at_every_arity() {
        for arity in 1..=PlayerSet::MAX_SIMULTANEOUS {
            let key_bits = (64 / arity) as u8;
            let cap = JointKey::capacity(key_bits);

            // Distinct per slot, so a leak between fields cannot pass.
            let picks: Vec<u32> = (0..arity as u32).map(|slot| cap - 1 - slot).collect();
            let key = JointKey::pack(&picks, key_bits);
            for (slot, &pick) in picks.iter().enumerate() {
                assert_eq!(key.arm(slot, key_bits), pick, "arity {arity} slot {slot}");
            }

            let zeros = vec![0u32; arity];
            assert_eq!(JointKey::pack(&zeros, key_bits), JointKey::default());
        }
    }

    #[test]
    fn joint_key_capacity_matches_the_field_width() {
        for arity in 1..=PlayerSet::MAX_SIMULTANEOUS {
            let key_bits = (64 / arity) as u8;
            let cap = JointKey::capacity(key_bits);
            let picks = vec![cap - 1; arity];
            let key = JointKey::pack(&picks, key_bits);
            for slot in 0..arity {
                assert_eq!(key.arm(slot, key_bits), cap - 1, "arity {arity}");
            }
        }
        assert_eq!(JointKey::capacity(64), u32::MAX);
        assert_eq!(JointKey::capacity(32), u32::MAX);
        assert_eq!(JointKey::capacity(21), 2_097_152);
        assert_eq!(JointKey::capacity(16), 65_536);
        assert_eq!(JointKey::capacity(12), 4_096);
        assert_eq!(JointKey::capacity(10), 1_024);
        assert_eq!(JointKey::capacity(9), 512);
        assert_eq!(JointKey::capacity(8), 256);
    }

    #[test]
    fn growing_a_slot_leaves_every_stored_key_meaning_what_it_meant() {
        let players = PlayerSet::first_n(3);
        let mut simul = simul_of(players, &[&[10, 11], &[20, 21, 22], &[30]], true);
        let key_bits = simul.key_bits;

        let tuples = [[0u32, 0, 0], [1, 2, 0], [0, 1, 0], [1, 0, 0]];
        let keys: Vec<JointKey> = tuples
            .iter()
            .map(|picks| JointKey::pack(picks, key_bits))
            .collect();
        let meaning = |simul: &Simul<u16>, key: JointKey| -> Vec<u16> {
            (0..3)
                .map(|slot| {
                    simul.arm_choices
                        [simul.starts[slot] as usize + key.arm(slot, key_bits) as usize]
                })
                .collect()
        };
        let before: Vec<Vec<u16>> = keys.iter().map(|&key| meaning(&simul, key)).collect();

        simul.grow_slot(0, &12, 7, true, usize::MAX);
        simul.grow_slot(1, &23, 7, true, usize::MAX);
        simul.grow_slot(0, &13, 7, true, usize::MAX);

        for (key, expected) in keys.iter().zip(&before) {
            assert_eq!(&meaning(&simul, *key), expected);
        }
        assert_eq!(simul.slot_len(0), 4);
        assert_eq!(simul.slot_len(1), 4);
        assert_eq!(simul.slot_len(2), 1);
        assert_eq!(simul.starts[3] as usize, simul.arm_stats.len());
        assert_eq!(simul.arm_policy.len(), simul.arm_stats.len());
        // The newest arm of slot 0 takes the next slot-relative index.
        assert_eq!(simul.find_arm(0, &13), Some(3));
        assert_eq!(simul.find_arm(1, &23), Some(4 + 3));
    }

    fn expanded_node(threshold: usize, policy: SimultaneousPolicy) -> Node<u16> {
        let mut node = Node::new_root(0);
        node.ensure_simul(PlayerSet::first_n(2));
        let first: Vec<u16> = (0..40).collect();
        let second: Vec<u16> = (100..130).collect();
        for _ in 0..2 {
            assert_eq!(
                node.expand_marginals(0, &first, policy, threshold),
                first.len()
            );
            assert_eq!(
                node.expand_marginals(1, &second, policy, threshold),
                second.len()
            );
            node.record(0.0);
        }
        node
    }

    #[test]
    fn indexed_and_scanned_arm_lookup_agree() {
        let scanned = expanded_node(usize::MAX, RM);
        let indexed = expanded_node(4, RM);
        let scanned = scanned.simul().unwrap();
        let indexed = indexed.simul().unwrap();
        assert!(scanned.arm_index.is_none());
        assert!(indexed.arm_index.is_some());

        for slot in 0..2 {
            for choice in 0..200u16 {
                assert_eq!(
                    scanned.find_arm(slot, &choice),
                    indexed.find_arm(slot, &choice),
                    "slot {slot} choice {choice}"
                );
            }
        }
        // A choice belonging to the other player is not this player's arm.
        assert_eq!(indexed.find_arm(0, &100), None);
        assert_eq!(indexed.find_arm(1, &0), None);
        assert_eq!(indexed.find_arm(1, &100), Some(40));
    }

    #[test]
    fn expansion_stamps_every_arm_once_per_visit() {
        let node = expanded_node(4, DUCT);
        let simul = node.simul().unwrap();
        assert_eq!(simul.arm_stats.len(), 70);
        for arm in &simul.arm_stats {
            assert_eq!(arm.availability, 2);
            assert_eq!(arm.avail_epoch, 1);
            assert_eq!(arm.ln_availability, 2.0f64.ln());
        }
    }

    /// `ln_availability` is `Duct`'s exploration denominator and nothing else
    /// reads it, so a mixing policy must not pay a libm call per arm per visit
    /// to maintain it. The availability *count* is read by every policy, so it
    /// keeps moving either way.
    #[test]
    fn only_duct_pays_for_the_availability_logarithm() {
        for (policy, expected) in [(DUCT, 2.0f64.ln()), (RM, 0.0)] {
            let mut node = expanded_node(4, policy);
            for arm in &node.simul().unwrap().arm_stats {
                assert_eq!(arm.availability, 2);
                assert_eq!(arm.ln_availability, expected);
            }

            // The `ROOT_CHOICES_INVARIANT` fast path stamps the same arms
            // without enumerating them, and has the same reason not to.
            node.record(0.0);
            node.restamp_marginals(policy);
            let expected = if policy == DUCT { 3.0f64.ln() } else { 0.0 };
            for arm in &node.simul().unwrap().arm_stats {
                assert_eq!(arm.availability, 3);
                assert_eq!(arm.ln_availability, expected);
            }
        }
    }

    /// `select` scores a root's children against the root's own visit count,
    /// so a root child's `ln_availability` has no reader — and a search at a
    /// root that enumerates its choices every iteration would otherwise pay a
    /// libm call for each of them.
    #[test]
    fn a_root_child_pays_no_availability_logarithm() {
        let choices: Vec<u16> = (0..4).collect();
        let mut avail = Vec::new();
        let mut rng = WyRand::seed_from_u64(1);

        let mut root = Node::new_root(0);
        for _ in 0..2 {
            root.expand(&choices, 0, usize::MAX, &mut avail, &mut rng);
        }
        assert_eq!(root.children.len(), choices.len());
        for child in &root.children {
            assert_eq!(child.availability, 2);
            assert_eq!(child.ln_availability, 0.0);
        }

        // The same expansion under a non-root parent, where `select` does read
        // the field, must still cache it.
        let mut inner = Node::new(0, NodeKind::Choice, Some(9u16));
        for _ in 0..5 {
            inner.expand(&choices, 0, usize::MAX, &mut avail, &mut rng);
        }
        let opened = &inner.children[0];
        assert!(opened.availability > 1);
        assert_eq!(opened.ln_availability, (opened.availability as f64).ln());
    }

    /// The arm index's payloads are slot-relative so that growing a slot moves
    /// no other arm's identity — which is what lets a newly discovered action
    /// be inserted rather than force the whole index to be thrown away and
    /// rebuilt on the next pass.
    #[test]
    fn growing_a_slot_keeps_the_arm_index() {
        let mut node = expanded_node(4, RM);
        // A threshold no arm count can cross, so nothing here can *rebuild* the
        // index: what the lookups below find is the index built before the
        // growth, or nothing.
        let unreachable = usize::MAX;
        assert_eq!(
            node.expand_marginals(0, &[40, 41], RM, unreachable),
            2,
            "two actions no earlier determinization offered"
        );
        assert_eq!(node.expand_marginals(1, &[130], RM, unreachable), 1);

        let simul = node.simul().unwrap();
        assert!(
            simul.arm_index.is_some(),
            "the index was discarded by a newly discovered action"
        );
        for (slot, present) in [(0, 0..42u16), (1, 100..131)] {
            for choice in present {
                let found = simul.find_arm(slot, &choice).expect("an arm exists");
                assert_eq!(simul.arm_choices[found], choice);
                assert!(simul.slot_range(slot).contains(&found));
            }
        }
        assert_eq!(simul.find_arm(0, &130), None);
        assert_eq!(simul.find_arm(1, &41), None);
    }

    #[test]
    fn a_mixed_policy_sums_to_one() {
        let mut simul = simul_of(PlayerSet::first_n(2), &[&[1, 2, 3], &[4, 5]], true);
        for (i, arm) in simul.arm_stats.iter_mut().enumerate() {
            arm.availability = i as u32 + 1;
        }
        for (i, arm) in simul.arm_policy.iter_mut().enumerate() {
            arm.strategy_sum = i as f64;
        }

        let mut out = Vec::new();
        simul.marginals(0).policy_into(&mut out);
        assert_eq!(out.len(), 3);
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // First arm has no strategy mass; the rest are weighted by
        // strategy_sum / availability, which is 0/1, 1/2, 2/3 here.
        assert_eq!(out[0], 0.0);
        assert!(out[2] > out[1]);

        simul.marginals(1).policy_into(&mut out);
        assert_eq!(out.len(), 2);
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // Nothing accumulated yet: uniform, not a division by zero.
        let empty = simul_of(PlayerSet::first_n(2), &[&[1, 2, 3], &[4]], true);
        empty.marginals(0).policy_into(&mut out);
        assert_eq!(out, vec![1.0 / 3.0; 3]);
    }

    #[test]
    fn a_pure_policy_is_one_hot_at_the_most_visited_arm() {
        let mut simul = simul_of(PlayerSet::first_n(2), &[&[1, 2, 3], &[4, 5]], false);
        assert!(simul.arm_policy.is_empty());

        let mut out = Vec::new();
        simul.marginals(0).policy_into(&mut out);
        assert_eq!(out, vec![1.0 / 3.0; 3]);

        // Opportunities alongside the selections: an arm cannot have been taken
        // more often than it was offered, and the ranking divides one by the
        // other.
        for arm in simul.arm_stats.iter_mut() {
            arm.availability = 8;
        }
        simul.arm_stats[1].visits = 5;
        simul.arm_stats[2].visits = 3;
        simul.marginals(0).policy_into(&mut out);
        assert_eq!(out, vec![0.0, 1.0, 0.0]);
        assert_eq!(out.iter().sum::<f64>(), 1.0);
        assert_eq!(simul.marginals(0).most_visited(), Some(1));
        assert_eq!(simul.marginals(1).most_visited(), None);
    }

    /// The masking recipe `policy_into` used to document — zero that vector's
    /// illegal entries and renormalize — normalizes `[0, 0, 0]` by 0.0 whenever
    /// a `Duct` leader is not on offer in the real position. The masked
    /// extraction re-ranks over the legal arms instead, which can name a
    /// different action and never a NaN.
    #[test]
    fn a_masked_pure_policy_re_ranks_rather_than_rescales() {
        let mut simul = simul_of(PlayerSet::first_n(2), &[&[1, 2, 3], &[4, 5]], false);
        for (arm, reward) in simul.arm_stats.iter_mut().zip([20.0, 36.0, 28.0]) {
            arm.visits = 40;
            arm.availability = 40;
            arm.cumulative_reward = reward;
        }

        let marginals = simul.marginals(0);
        let mut out = Vec::new();
        marginals.policy_into(&mut out);
        assert_eq!(out, vec![0.0, 1.0, 0.0], "one-hot at the best mean");

        marginals.policy_masked_into(&[true, false, true], &mut out);
        assert_eq!(out, vec![0.0, 0.0, 1.0]);
        assert_eq!(out.iter().sum::<f64>(), 1.0);

        // No legal arm is a tree with nothing to say about the position, which
        // is a real state at a low budget and not a division by zero.
        marginals.policy_masked_into(&[false, false, false], &mut out);
        assert_eq!(out, vec![0.0; 3]);
    }

    #[test]
    fn a_masked_mixed_policy_renormalizes_over_the_legal_arms() {
        let mut simul = simul_of(PlayerSet::first_n(2), &[&[1, 2, 3], &[4, 5]], true);
        for arm in simul.arm_stats.iter_mut() {
            arm.availability = 4;
        }
        for (i, arm) in simul.arm_policy.iter_mut().enumerate() {
            arm.strategy_sum = i as f64 + 1.0;
        }

        let mut out = Vec::new();
        simul
            .marginals(0)
            .policy_masked_into(&[true, false, true], &mut out);
        // Mass 1 and 3 over a common availability, and nothing on the arm the
        // position withholds.
        assert_eq!(out.len(), 3);
        assert_eq!(out[1], 0.0);
        assert!((out[0] - 0.25).abs() < 1e-12);
        assert!((out[2] - 0.75).abs() < 1e-12);

        // Legal arms that carry no mass fall back to uniform over them — the
        // same draw the search itself makes in that state.
        let fresh = simul_of(PlayerSet::first_n(2), &[&[1, 2, 3], &[4]], true);
        fresh
            .marginals(0)
            .policy_masked_into(&[false, true, true], &mut out);
        assert_eq!(out, vec![0.0, 0.5, 0.5]);
    }

    #[test]
    #[should_panic(expected = "out of step")]
    fn a_mask_must_be_parallel_to_the_arms() {
        let simul = simul_of(PlayerSet::first_n(2), &[&[1, 2, 3], &[4, 5]], true);
        simul
            .marginals(0)
            .policy_masked_into(&[true, true], &mut Vec::new());
    }

    #[test]
    fn joint_successors_are_found_by_key_scanned_or_indexed() {
        for threshold in [usize::MAX, 2] {
            let mut node: Node<u16> = Node::new_root(0);
            node.ensure_simul(PlayerSet::first_n(2));
            let key_bits = node.simul().unwrap().key_bits;

            let mut keys = Vec::new();
            for a in 0..4u32 {
                for b in 0..3u32 {
                    let key = JointKey::pack(&[a, b], key_bits);
                    let child = node.push_joint(key, 0, threshold);
                    assert_eq!(child, keys.len());
                    keys.push(key);
                }
            }
            node.debug_check_arms();

            let simul = node.simul().unwrap();
            assert_eq!(simul.joint_index.is_some(), threshold != usize::MAX);
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(simul.find_joint(*key), Some(i));
            }
            assert_eq!(simul.find_joint(JointKey::pack(&[9, 9], key_bits)), None);
            assert_eq!(node.children.len(), keys.len());
            assert!(node.children.iter().all(|c| c.kind() == NodeKind::Joint));
            assert_eq!(node.joint_arm(5, 0), Some(1));
            assert_eq!(node.joint_arm(5, 1), Some(2));
            assert_eq!(node.joint_arm(5, 2), None);
        }
    }

    #[test]
    fn rerooting_at_a_joint_successor_keeps_its_marginals() {
        let mut node = Node::new_root(0);
        node.ensure_simul(PlayerSet::first_n(2));
        let key_bits = node.simul().unwrap().key_bits;
        let kept = JointKey::pack(&[1, 1], key_bits);

        for a in 0..2u32 {
            for b in 0..2u32 {
                let i = node.push_joint(JointKey::pack(&[a, b], key_bits), 0, 16);
                node.children[i].record(1.0);
            }
        }
        let target = node.simul().unwrap().find_joint(kept).unwrap();
        node.children[target].ensure_simul(PlayerSet::first_n(2));
        node.children[target].expand_marginals(0, &[7u16, 8], RM, 16);

        assert!(node.reroot_at_joint(kept));
        assert_eq!(node.kind(), NodeKind::Root);
        assert!(node.is_root());
        assert_eq!(node.visits(), 1);
        assert_eq!(node.availability, 1);
        assert_eq!(node.marginals(0).unwrap().len(), 2);
        assert_eq!(*node.marginals(0).unwrap().choice(1), 8);

        assert!(!node.reroot_at_joint(JointKey::pack(&[1, 0], key_bits)));
    }

    #[test]
    fn a_sequential_node_still_reports_its_kind() {
        let mut node: Node<u16> = Node::new_root(0);
        assert_eq!(node.kind(), NodeKind::Root);
        assert!(node.simultaneous_players().is_none());
        assert!(node.marginals(0).is_none());
        assert_eq!(node.joint_arm(0, 0), None);

        node.push_child(3, 1, 16);
        assert_eq!(node.children[0].kind(), NodeKind::Choice);
        assert_eq!(node.children[0].reward_player(), 1);
        assert!(node.reroot_at(&3));
        assert_eq!(node.kind(), NodeKind::Root);
        assert_eq!(node.edge(), Edge::Root);
    }
}
