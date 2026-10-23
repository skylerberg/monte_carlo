//! One observer's tree: a node per information set that observer can
//! distinguish, keyed by [`crate::Observed::Key`] rather than by the move.
//!
//! Deliberately its own node rather than a parameter of [`crate::Node`]: the
//! edges here carry no `Choice` at all. A stored `Choice` would be replayed into
//! a later determinization in which the concealed card behind it has been
//! resampled, which is exactly the trap the `Choice` contract exists to
//! prevent — so the action applied is always re-derived from the world being
//! searched, and only its perception is stored.

use core::hash::Hash;

use hashbrown::HashTable;
use rand_core::Rng;

use crate::select::ucb_raw;
use crate::util::{below, hash_of};

/// A node in one player's tree, for [`crate::ObservedSearcher`].
///
/// Same statistics as [`crate::Node`] and the same meaning: `cumulative_reward`
/// is accumulated for [`ObservedNode::reward_player`], `times_available` is the
/// exploration denominator below the root. What differs is the edge: a child is
/// found by its key, so two moves the observer cannot tell apart share it.
pub struct ObservedNode<K> {
    pub(crate) cumulative_reward: f64,
    /// Cached `ln(times_available)`, refreshed only when `times_available`
    /// changes — and only at a non-root parent, where `select` reads it.
    pub(crate) ln_times_available: f64,
    pub(crate) children: Vec<ObservedNode<K>>,
    /// A hash index over `children`, built once a node grows past
    /// [`crate::Game::CHILD_INDEX_THRESHOLD`]. Below it the linear scan wins.
    pub(crate) index: Option<Box<HashTable<u32>>>,
    pub(crate) visits: u32,
    pub(crate) times_available: u32,
    pub(crate) heuristic_bias: f32,
    /// The player who moved into this node, whose reward
    /// `cumulative_reward` accumulates. At a root, the player to act.
    pub(crate) player: u8,
    /// What the observer saw of the move that reached this node. `None` at the
    /// root, which has no incoming edge.
    pub(crate) key: Option<K>,
}

impl<K> ObservedNode<K> {
    pub(crate) fn new_root(player: u8) -> Self {
        Self::new(player, None)
    }

    fn new(player: u8, key: Option<K>) -> Self {
        Self {
            cumulative_reward: 0.0,
            ln_times_available: 0.0,
            children: Vec::new(),
            index: None,
            visits: 0,
            times_available: 1,
            heuristic_bias: 0.0,
            player,
            key,
        }
    }

    #[inline(always)]
    pub(crate) fn is_root(&self) -> bool {
        self.key.is_none()
    }

    /// Times this node was visited.
    pub fn visits(&self) -> u32 {
        self.visits
    }

    /// Iterations in which the move leading here was available to this tree's
    /// parent, which is the denominator [`ObservedNode::visits`] is a numerator
    /// of.
    ///
    /// Read it on a child. A root has no parent to have been offered by.
    pub fn times_available(&self) -> u32 {
        self.times_available
    }

    /// Mean reward for the player this node accumulates for, or 0.0 if never
    /// visited. See [`ObservedNode::reward_player`].
    pub fn mean_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.cumulative_reward / self.visits as f64
        }
    }

    /// Total reward accumulated for [`ObservedNode::reward_player`].
    pub fn cumulative_reward(&self) -> f64 {
        self.cumulative_reward
    }

    /// The player whose reward this node accumulates: the one who moved into
    /// it, or the one to act at a root.
    pub fn reward_player(&self) -> u8 {
        self.player
    }

    /// What the observer saw of the move that reached this node, or `None` at
    /// the root.
    pub fn observed_key(&self) -> Option<&K> {
        self.key.as_ref()
    }

    /// This node's expanded children.
    pub fn children(&self) -> &[ObservedNode<K>] {
        &self.children
    }

    /// Nodes in this subtree, including this one.
    pub fn node_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(ObservedNode::node_count)
            .sum::<usize>()
    }

    /// Longest root-to-leaf path in this subtree, in nodes.
    pub fn max_depth(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(ObservedNode::max_depth)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn record(&mut self, reward: f64) {
        self.visits += 1;
        self.cumulative_reward += reward;
    }
}

impl<K: Clone + Eq + Hash> ObservedNode<K> {
    /// The child reached by `key`, if this node has one.
    #[inline]
    pub(crate) fn find_child(&self, key: &K) -> Option<usize> {
        match self.index.as_deref() {
            None => self
                .children
                .iter()
                .position(|child| child.key.as_ref() == Some(key)),
            Some(index) => {
                let hash = hash_of(key);
                index
                    .find(hash, |&i| {
                        self.children[i as usize].key.as_ref() == Some(key)
                    })
                    .map(|&i| i as usize)
            }
        }
    }

    /// The child reached by `key`, created as a scaffold if this tree has never
    /// taken that edge.
    ///
    /// Only the trees that are *not* choosing this step create children this
    /// way — the ones that merely have to keep their path aligned with the
    /// selecting tree's. The selecting tree's child comes from [`Self::expand`],
    /// which is where a child's availability is recorded; a scaffold edge
    /// carries none, and once its own player does move here, `expand` finds it
    /// like any other.
    pub(crate) fn find_or_push(&mut self, key: K, player: u8, threshold: usize) -> usize {
        match self.find_child(&key) {
            Some(i) => i,
            None => self.push_child(key, player, threshold),
        }
    }

    fn push_child(&mut self, key: K, player: u8, threshold: usize) -> usize {
        let hash = hash_of(&key);
        let position = self.children.len() as u32;
        self.children.push(ObservedNode::new(player, Some(key)));

        match self.index.as_deref_mut() {
            None if self.children.len() > threshold => self.build_index(),
            None => {}
            Some(index) => {
                let children: &[ObservedNode<K>] = &self.children;
                let _ = index.insert_unique(hash, position, |&i| {
                    hash_of(
                        children[i as usize]
                            .key
                            .as_ref()
                            .expect("an observed child carries its key"),
                    )
                });
            }
        }
        position as usize
    }

    fn build_index(&mut self) {
        let children: &[ObservedNode<K>] = &self.children;
        let mut table = HashTable::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            let _ = table.insert_unique(
                hash_of(
                    child
                        .key
                        .as_ref()
                        .expect("an observed child carries its key"),
                ),
                i as u32,
                |&j| {
                    hash_of(
                        children[j as usize]
                            .key
                            .as_ref()
                            .expect("an observed child carries its key"),
                    )
                },
            );
        }
        self.index = Some(Box::new(table));
    }

    /// Record this iteration's legal moves by their keys, and grow the tree by
    /// one child.
    ///
    /// Every already-present child that is legal now has its `times_available`
    /// bumped, and `avail[i]` is set for it. The root creates a child for every
    /// key; other nodes create exactly one, reservoir-sampled uniformly from the
    /// keys they have not seen. Mirroring [`crate::Node::expand`], including the
    /// sampling, which is what keeps a wide choice list cheap.
    pub(crate) fn expand<R: Rng + ?Sized>(
        &mut self,
        keys: &[K],
        player: u8,
        threshold: usize,
        avail: &mut Vec<bool>,
        rng: &mut R,
    ) {
        avail.clear();
        avail.resize(self.children.len(), false);

        let is_root = self.is_root();
        if is_root && self.children.is_empty() {
            self.children.reserve(keys.len());
        }

        let mut unseen: u64 = 0;
        let mut sampled = usize::MAX;

        for (i, key) in keys.iter().enumerate() {
            match self.find_child(key) {
                // A duplicate within `keys` must not be counted twice.
                Some(k) if avail[k] => {}
                Some(k) => {
                    let child = &mut self.children[k];
                    child.times_available += 1;
                    if !is_root {
                        child.ln_times_available = (child.times_available as f64).ln();
                    }
                    avail[k] = true;
                }
                None if is_root => {
                    self.push_child(key.clone(), player, threshold);
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
            self.push_child(keys[sampled].clone(), player, threshold);
            avail.push(true);
        }
    }

    /// Re-root at `key`, keeping that subtree and discarding the rest. Returns
    /// false and leaves `self` untouched if this node has no such child.
    pub(crate) fn reroot_at(&mut self, key: &K) -> bool {
        let Some(i) = self.find_child(key) else {
            return false;
        };
        let mut child = self.children.swap_remove(i);
        child.key = None;
        child.times_available = 1;
        child.ln_times_available = 0.0;
        *self = child;
        true
    }
}

/// Index of the child to descend into, or `None` if nothing is legal here.
///
/// The same rule as [`crate::select::select`] — first unvisited legal child,
/// otherwise the highest UCB1 with the availability denominator below the root
/// and the parent's visit count at it. Written out rather than shared, because
/// sharing it would mean generalizing the sequential hot path over an edge type
/// the games that never observe anything would still instantiate.
pub(crate) fn select<K>(
    node: &ObservedNode<K>,
    avail: &[bool],
    exploration: f64,
    bias_weight: f64,
) -> Option<usize> {
    debug_assert_eq!(avail.len(), node.children.len());

    let ln_parent_visits = if node.is_root() {
        (node.visits as f64).ln()
    } else {
        0.0
    };

    let mut best = None;
    let mut best_value = f64::NEG_INFINITY;

    for (i, child) in node.children.iter().enumerate() {
        if !avail[i] {
            continue;
        }
        if child.visits == 0 {
            return Some(i);
        }
        let ln_total = if node.is_root() {
            ln_parent_visits
        } else {
            child.ln_times_available
        };
        let value = ucb_raw(
            child.cumulative_reward,
            child.visits as f64,
            ln_total,
            exploration,
            bias_weight,
            child.heuristic_bias,
        );
        if value > best_value {
            best_value = value;
            best = Some(i);
        }
    }
    best
}
