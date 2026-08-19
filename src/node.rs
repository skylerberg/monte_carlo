use core::hash::Hash;

use hashbrown::HashTable;
use rand_core::Rng;

use crate::util::{below, hash_of};

/// A node in the search tree.
///
/// Children are owned inline in a contiguous `Vec`, which is what `select`
/// scans. `index` is built lazily once a node grows wide enough that the scan
/// in `expand` would dominate; it maps a choice's hash to a position in
/// `children`, so a `Choice` is never stored twice.
pub struct Node<C> {
    pub(crate) cumulative_reward: f64,
    /// Cached `ln(availability)`, refreshed only when `availability` changes.
    pub(crate) ln_availability: f64,
    pub(crate) children: Vec<Node<C>>,
    pub(crate) index: Option<Box<ChildIndex>>,
    pub(crate) visits: u32,
    /// Iterations in which this choice was legal at its parent. The exploration
    /// term uses this rather than the parent's visit count, so a choice that is
    /// rarely legal is not penalised for being rarely tried.
    pub(crate) availability: u32,
    pub(crate) heuristic_bias: f32,
    pub(crate) player: u8,
    /// `None` marks the root.
    pub(crate) choice: Option<C>,
}

pub(crate) struct ChildIndex {
    table: HashTable<u32>,
}

impl<C> Node<C> {
    pub(crate) fn new_root(player: u8) -> Self {
        Self::new(player, None)
    }

    fn new(player: u8, choice: Option<C>) -> Self {
        Self {
            cumulative_reward: 0.0,
            ln_availability: 0.0,
            children: Vec::new(),
            index: None,
            visits: 0,
            availability: 1,
            heuristic_bias: 0.0,
            player,
            choice,
        }
    }

    #[inline(always)]
    pub(crate) fn is_root(&self) -> bool {
        self.choice.is_none()
    }

    /// Times this node was visited.
    pub fn visits(&self) -> u32 {
        self.visits
    }

    /// Mean reward for this node's own player, or 0.0 if never visited.
    pub fn mean_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.cumulative_reward / self.visits as f64
        }
    }

    /// Total reward accumulated for this node's own player.
    pub fn cumulative_reward(&self) -> f64 {
        self.cumulative_reward
    }

    /// The choice leading to this node, or `None` at the root.
    pub fn choice(&self) -> Option<&C> {
        self.choice.as_ref()
    }

    /// This node's expanded children.
    pub fn children(&self) -> &[Node<C>] {
        &self.children
    }

    /// Nodes in this subtree, including this one.
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(Node::node_count).sum::<usize>()
    }

    /// Longest root-to-leaf path in this subtree, in nodes.
    pub fn max_depth(&self) -> usize {
        1 + self.children.iter().map(Node::max_depth).max().unwrap_or(0)
    }

    #[inline(always)]
    pub(crate) fn record(&mut self, reward: f64) {
        self.cumulative_reward += reward;
        self.visits += 1;
    }
}

impl<C: Clone + Eq + Hash> Node<C> {
    #[inline]
    pub(crate) fn find_child(&self, choice: &C) -> Option<usize> {
        match &self.index {
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

    fn push_child(&mut self, choice: C, player: u8, threshold: usize) {
        let hash = hash_of(&choice);
        let position = self.children.len() as u32;
        self.children.push(Node::new(player, Some(choice)));

        if self.index.is_none() {
            if self.children.len() > threshold {
                self.build_index();
            }
            return;
        }

        let Self {
            children, index, ..
        } = self;
        if let Some(index) = index {
            let _ = index.table.insert_unique(hash, position, |&i| {
                hash_of(children[i as usize].choice.as_ref().unwrap())
            });
        }
    }

    fn build_index(&mut self) {
        let children: &[Node<C>] = &self.children;
        let mut table = HashTable::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            let hash = hash_of(child.choice.as_ref().unwrap());
            let _ = table.insert_unique(hash, i as u32, |&j| {
                hash_of(children[j as usize].choice.as_ref().unwrap())
            });
        }
        self.index = Some(Box::new(ChildIndex { table }));
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
                    child.ln_availability = (child.availability as f64).ln();
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

    /// Re-root at `choice`, keeping that subtree and discarding the rest.
    /// Returns false and leaves `self` untouched if `choice` is not a child.
    pub(crate) fn reroot_at(&mut self, choice: &C) -> bool {
        let Some(i) = self.find_child(choice) else {
            return false;
        };
        let mut child = self.children.swap_remove(i);
        child.choice = None;
        child.availability = 1;
        child.ln_availability = 0.0;
        *self = child;
        true
    }
}
