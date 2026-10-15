//! The transposition graph's storage: an arena of positions, the links between
//! them, and the table that finds a position by its key.
//!
//! Positions are vertices in one `Vec` and refer to each other by index, which
//! is what lets two parents share a successor — the thing an owned tree cannot
//! express. A link carries the statistics that belong to the *edge*: how often
//! this parent took it and how often it was legal here. The successor's value
//! belongs to the position and lives on the vertex, where every parent reads it.

use core::hash::Hash;

use hashbrown::HashTable;
use rand_core::Rng;

use crate::game::PlayerSet;
use crate::node::Simul;
use crate::select::ExplorationCount;
use crate::util::{below, hash_of};

/// The target of a link no iteration has taken yet: its successor is learned on
/// the first traversal, when the key can be read off the state.
pub(crate) const UNRESOLVED: u32 = u32::MAX;

/// One position.
pub(crate) struct Vertex<C, K> {
    /// Total reward for `player` over every iteration that entered this
    /// position, by any route.
    pub(crate) cumulative_reward: f64,
    /// Outgoing links at a sequential vertex. Empty at a simultaneous one,
    /// whose successors live in [`Joint`].
    pub(crate) links: Vec<Link<C>>,
    /// A hash index over `links`, or a simultaneous vertex's marginal bandits.
    /// A vertex is never both, so the two share one allocation.
    pub(crate) extra: Option<Box<Extra<C>>>,
    /// What this vertex was merged under. `None` for a position the game keeps
    /// out of every merge, and for the root of a fresh search.
    pub(crate) key: Option<K>,
    pub(crate) visits: u32,
    /// The iteration that last entered this vertex. Equal to the graph's
    /// current stamp exactly when the vertex is on the path being descended,
    /// which is the whole cycle test.
    pub(crate) path_stamp: u32,
    pub(crate) heuristic_bias: f32,
    /// The player whose reward `cumulative_reward` accumulates: the mover into
    /// this position. Part of the merge identity, so every route in shares it.
    pub(crate) player: u8,
    /// Whether `links` — or, at a simultaneous vertex, the arms — are exactly
    /// the choices the last enumeration offered.
    pub(crate) links_match_choices: bool,
}

/// An edge: one choice at one position, and what taking it led to.
pub(crate) struct Link<C> {
    /// Cached `ln(times_available)`, the ISMCTS exploration denominator, kept
    /// only where selection reads it.
    pub(crate) ln_times_available: f64,
    pub(crate) choice: C,
    /// Index of the successor, or [`UNRESOLVED`].
    pub(crate) target: u32,
    /// Times this parent took this link. The exploration term reads this
    /// rather than the successor's visits, which also count other parents.
    pub(crate) visits: u32,
    /// Iterations at this parent in which the choice was legal.
    pub(crate) times_available: u32,
}

pub(crate) enum Extra<C> {
    Index(HashTable<u32>),
    Joint(Box<Joint<C>>),
}

/// A simultaneous vertex's marginal bandits and its joint successors.
pub(crate) struct Joint<C> {
    pub(crate) simul: Simul<C>,
    /// Parallel to `simul.joint_keys`: the successor each joint action led to.
    pub(crate) targets: Vec<u32>,
}

/// A searched game, as a graph of positions.
///
/// Held by [`crate::GraphSearcher`] and handed out by
/// [`crate::GraphSearcher::take_graph`]; read it through [`Graph::root`].
pub struct Graph<C, K> {
    pub(crate) vertices: Vec<Vertex<C, K>>,
    /// Every vertex merged under a key, found by the hash of `(key, player)`.
    /// The root is never in it: its reward player is the player to act rather
    /// than a mover, so it answers a different question than a merged vertex.
    pub(crate) table: HashTable<u32>,
    pub(crate) root: u32,
    /// The current iteration's path stamp. Never zero, which is what a vertex
    /// starts with, so a new vertex is never mistaken for one on the path.
    pub(crate) stamp: u32,
}

#[inline]
fn merge_hash<K: Hash>(key: &K, player: u8) -> u64 {
    hash_of(&(key, player))
}

impl<C, K> Vertex<C, K> {
    pub(crate) fn new(player: u8, key: Option<K>) -> Self {
        Self {
            cumulative_reward: 0.0,
            links: Vec::new(),
            extra: None,
            key,
            visits: 0,
            path_stamp: 0,
            heuristic_bias: 0.0,
            player,
            links_match_choices: false,
        }
    }

    #[inline(always)]
    pub(crate) fn record(&mut self, reward: f64) {
        self.cumulative_reward += reward;
        self.visits += 1;
    }

    #[inline(always)]
    pub(crate) fn mean_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.cumulative_reward / self.visits as f64
        }
    }

    #[inline(always)]
    pub(crate) fn joint(&self) -> Option<&Joint<C>> {
        match self.extra.as_deref() {
            Some(Extra::Joint(joint)) => Some(joint),
            _ => None,
        }
    }

    #[inline(always)]
    pub(crate) fn joint_mut(&mut self) -> Option<&mut Joint<C>> {
        match self.extra.as_deref_mut() {
            Some(Extra::Joint(joint)) => Some(joint),
            _ => None,
        }
    }

    #[inline(always)]
    fn index(&self) -> Option<&HashTable<u32>> {
        match self.extra.as_deref() {
            Some(Extra::Index(index)) => Some(index),
            _ => None,
        }
    }

    /// The marginal block of a vertex that reported a simultaneous status,
    /// created empty on the first visit.
    pub(crate) fn ensure_joint(&mut self, players: PlayerSet) -> &mut Joint<C> {
        if self.joint().is_none() {
            debug_assert!(
                self.extra.is_none(),
                "mcts: this vertex's Choice-keyed link index is being replaced by \
                 marginal bandits. A position's kind must not vary across \
                 determinizations, and two positions merged under one key must be \
                 the same kind."
            );
            self.extra = Some(Box::new(Extra::Joint(Box::new(Joint {
                simul: Simul::new(players),
                targets: Vec::new(),
            }))));
        }
        self.joint_mut()
            .expect("the simultaneous block was just installed")
    }

    /// Every successor this vertex links to, sequential or joint.
    pub(crate) fn targets(&self) -> impl Iterator<Item = u32> + '_ {
        let joint = self.joint().map(|joint| joint.targets.as_slice());
        self.links
            .iter()
            .map(|link| link.target)
            .chain(joint.unwrap_or(&[]).iter().copied())
    }

    fn targets_mut(&mut self) -> impl Iterator<Item = &mut u32> + '_ {
        let Self { links, extra, .. } = self;
        let joint = match extra.as_deref_mut() {
            Some(Extra::Joint(joint)) => joint.targets.as_mut_slice(),
            _ => &mut [],
        };
        links
            .iter_mut()
            .map(|link| &mut link.target)
            .chain(joint.iter_mut())
    }
}

impl<C: Clone + Eq + Hash, K> Vertex<C, K> {
    #[inline]
    pub(crate) fn find_link(&self, choice: &C) -> Option<usize> {
        match self.index() {
            None => self.links.iter().position(|link| link.choice == *choice),
            Some(index) => index
                .find(hash_of(choice), |&i| {
                    self.links[i as usize].choice == *choice
                })
                .map(|&i| i as usize),
        }
    }

    fn push_link(&mut self, choice: C, threshold: usize) {
        debug_assert!(
            self.joint().is_none(),
            "mcts: a link is only ever added at a sequential vertex"
        );
        let hash = hash_of(&choice);
        let position = self.links.len() as u32;
        self.links.push(Link {
            ln_times_available: 0.0,
            choice,
            target: UNRESOLVED,
            visits: 0,
            times_available: 1,
        });

        if self.extra.is_none() {
            if self.links.len() > threshold {
                self.build_index();
            }
            return;
        }
        let Self { links, extra, .. } = self;
        if let Some(Extra::Index(index)) = extra.as_deref_mut() {
            let _ = index.insert_unique(hash, position, |&i| hash_of(&links[i as usize].choice));
        }
    }

    fn build_index(&mut self) {
        let links: &[Link<C>] = &self.links;
        let mut table = HashTable::with_capacity(links.len());
        for (i, link) in links.iter().enumerate() {
            let _ = table.insert_unique(hash_of(&link.choice), i as u32, |&j| {
                hash_of(&links[j as usize].choice)
            });
        }
        self.extra = Some(Box::new(Extra::Index(table)));
    }

    /// Record this iteration's legal choices, and grow the vertex by one link.
    ///
    /// The same rule as the tree's `Node::expand`, on links instead of child
    /// nodes: every link legal now gains an opportunity and is marked in
    /// `avail`; the root opens a link for every legal choice, and any other
    /// vertex opens one, sampled uniformly from the choices it has no link for.
    /// Keeping the rule identical is what makes a graph that never merges
    /// search exactly as the tree does.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn expand<R: Rng + ?Sized>(
        &mut self,
        is_root: bool,
        choices: &[C],
        threshold: usize,
        exploration_count: ExplorationCount,
        avail: &mut Vec<bool>,
        rng: &mut R,
    ) {
        avail.clear();
        avail.resize(self.links.len(), false);

        if is_root && self.links.is_empty() {
            self.links.reserve(choices.len());
        }

        let mut unseen: u64 = 0;
        let mut sampled = usize::MAX;
        let mut legal_links = 0;

        for (i, choice) in choices.iter().enumerate() {
            match self.find_link(choice) {
                // A duplicate within `choices` must not be counted twice.
                Some(k) if avail[k] => {}
                Some(k) => {
                    legal_links += 1;
                    let link = &mut self.links[k];
                    link.times_available += 1;
                    if exploration_count == ExplorationCount::TimesAvailable {
                        link.ln_times_available = (link.times_available as f64).ln();
                    }
                    avail[k] = true;
                }
                None if is_root => {
                    legal_links += 1;
                    self.push_link(choice.clone(), threshold);
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
            legal_links += 1;
            self.push_link(choices[sampled].clone(), threshold);
            avail.push(true);
        }

        let every_choice_has_a_link = is_root || unseen <= 1;
        let every_link_is_a_choice = legal_links == self.links.len();
        self.links_match_choices = every_choice_has_a_link && every_link_is_a_choice;
    }
}

impl<C, K> Graph<C, K> {
    /// A graph holding one unmerged root.
    pub(crate) fn new(root_player: u8) -> Self {
        Self {
            vertices: vec![Vertex::new(root_player, None)],
            table: HashTable::new(),
            root: 0,
            stamp: 1,
        }
    }

    #[inline(always)]
    pub(crate) fn root_vertex(&self) -> &Vertex<C, K> {
        &self.vertices[self.root as usize]
    }

    #[inline(always)]
    pub(crate) fn root_vertex_mut(&mut self) -> &mut Vertex<C, K> {
        &mut self.vertices[self.root as usize]
    }

    /// A fresh stamp for the next iteration's path. On wrapping, every vertex's
    /// stamp is cleared first, so no stale stamp can equal a reissued one.
    pub(crate) fn next_stamp(&mut self) -> u32 {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            for vertex in &mut self.vertices {
                vertex.path_stamp = 0;
            }
            self.stamp = 1;
        }
        self.stamp
    }
}

impl<C, K: Eq + Hash> Graph<C, K> {
    /// The vertex merged under `key` for `player`, if there is one.
    #[inline]
    pub(crate) fn lookup(&self, key: &K, player: u8) -> Option<u32> {
        let vertices = &self.vertices;
        self.table
            .find(merge_hash(key, player), |&i| {
                let vertex = &vertices[i as usize];
                vertex.player == player && vertex.key.as_ref() == Some(key)
            })
            .copied()
    }

    /// Add a vertex for a position first reached by `player`, merged under
    /// `key` when there is one, and return its index.
    pub(crate) fn insert(&mut self, player: u8, key: Option<K>) -> u32 {
        let id = self.vertices.len() as u32;
        assert!(
            id != UNRESOLVED,
            "mcts: the transposition graph holds {} positions, the most a u32 index can name",
            UNRESOLVED
        );
        let hash = key.as_ref().map(|key| merge_hash(key, player));
        self.vertices.push(Vertex::new(player, key));
        if let Some(hash) = hash {
            let vertices = &self.vertices;
            self.table.insert_unique(hash, id, |&i| {
                let vertex = &vertices[i as usize];
                merge_hash(
                    vertex
                        .key
                        .as_ref()
                        .expect("only keyed vertices are in the table"),
                    vertex.player,
                )
            });
        }
        id
    }

    /// Make `new_root` the root, keeping only what it reaches.
    ///
    /// Every vertex the new root cannot reach describes a position the game
    /// has moved past, and is dropped; the survivors are renumbered in
    /// breadth-first order from the root, and the table is rebuilt over them.
    /// The root leaves the table, as a fresh search's root is never in it, and
    /// any link back into it — a cycle through the new root — is cleared, since
    /// the root accumulates the player to act and a link reads its successor in
    /// the mover's currency.
    pub(crate) fn reroot(&mut self, new_root: u32) {
        let count = self.vertices.len();
        let mut remap = vec![UNRESOLVED; count];
        let mut order = vec![new_root];
        remap[new_root as usize] = 0;
        let mut head = 0;
        while head < order.len() {
            let id = order[head] as usize;
            head += 1;
            for target in self.vertices[id].targets() {
                if target != UNRESOLVED && remap[target as usize] == UNRESOLVED {
                    remap[target as usize] = order.len() as u32;
                    order.push(target);
                }
            }
        }

        let mut old: Vec<Option<Vertex<C, K>>> = core::mem::take(&mut self.vertices)
            .into_iter()
            .map(Some)
            .collect();
        let mut vertices = Vec::with_capacity(order.len());
        for &id in &order {
            let mut vertex = old[id as usize]
                .take()
                .expect("breadth-first order visits each vertex once");
            for target in vertex.targets_mut() {
                if *target != UNRESOLVED {
                    *target = remap[*target as usize];
                }
            }
            vertices.push(vertex);
        }
        for vertex in vertices.iter_mut() {
            for link in &mut vertex.links {
                if link.target == 0 {
                    link.target = UNRESOLVED;
                    link.visits = 0;
                }
            }
            if let Some(joint) = vertex.joint_mut() {
                for target in &mut joint.targets {
                    if *target == 0 {
                        *target = UNRESOLVED;
                    }
                }
            }
        }
        // What the root's last enumeration proved described it as somebody's
        // successor; the next search proves it afresh as a root.
        vertices[0].links_match_choices = false;

        self.vertices = vertices;
        self.root = 0;
        self.table.clear();
        let Self {
            vertices, table, ..
        } = self;
        let vertices = &*vertices;
        for (id, vertex) in vertices.iter().enumerate().skip(1) {
            if let Some(key) = vertex.key.as_ref() {
                table.insert_unique(merge_hash(key, vertex.player), id as u32, |&i| {
                    let vertex = &vertices[i as usize];
                    merge_hash(
                        vertex
                            .key
                            .as_ref()
                            .expect("only keyed vertices are in the table"),
                        vertex.player,
                    )
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Link `from` to `to` by `choice`, as a resolved link taken `visits` times.
    fn link(graph: &mut Graph<u8, u32>, from: u32, choice: u8, to: u32, visits: u32) {
        graph.vertices[from as usize].links.push(Link {
            ln_times_available: 0.0,
            choice,
            target: to,
            visits,
            times_available: visits.max(1),
        });
    }

    #[test]
    fn a_wrapped_stamp_clears_every_vertex_first() {
        let mut graph: Graph<u8, u32> = Graph::new(0);
        let a = graph.insert(0, Some(1));
        graph.stamp = u32::MAX;
        graph.vertices[a as usize].path_stamp = 1;
        graph.vertices[0].path_stamp = u32::MAX;

        assert_eq!(graph.next_stamp(), 1);
        assert!(graph.vertices.iter().all(|vertex| vertex.path_stamp == 0));
        assert_eq!(graph.next_stamp(), 2);
    }

    /// Root 0 reaches 1 and 2; 1 reaches 3; 2 links back to 1 and to 3; 3 links
    /// back to 1, a cycle. Re-rooting at 1 must keep 1 and 3 — and nothing that
    /// only the old root reached — renumber them from 0, clear the cycle's link
    /// into the new root, and leave the table finding 3 but not the root.
    #[test]
    fn re_rooting_keeps_what_the_new_root_reaches_and_cuts_links_into_it() {
        let mut graph: Graph<u8, u32> = Graph::new(0);
        let one = graph.insert(1, Some(10));
        let two = graph.insert(1, Some(20));
        let three = graph.insert(0, Some(30));
        link(&mut graph, 0, 0, one, 4);
        link(&mut graph, 0, 1, two, 3);
        link(&mut graph, one, 0, three, 2);
        link(&mut graph, two, 0, one, 1);
        link(&mut graph, two, 1, three, 1);
        link(&mut graph, three, 0, one, 1);

        graph.reroot(one);

        assert_eq!(
            graph.vertices.len(),
            2,
            "the old root and its other child are gone"
        );
        assert_eq!(graph.root, 0);
        assert_eq!(graph.vertices[0].key, Some(10));
        assert_eq!(
            graph.vertices[0].links[0].target, 1,
            "the new root's link is renumbered"
        );
        assert_eq!(graph.vertices[1].key, Some(30));
        let back = &graph.vertices[1].links[0];
        assert_eq!(
            back.target, UNRESOLVED,
            "a link into the new root is a cycle and is cut"
        );
        assert_eq!(back.visits, 0);

        assert_eq!(graph.lookup(&30, 0), Some(1));
        assert_eq!(graph.lookup(&10, 1), None, "the root is never in the table");
        assert_eq!(
            graph.lookup(&20, 1),
            None,
            "a dropped position is gone from the table"
        );
    }

    #[test]
    fn a_position_is_found_by_its_key_and_its_mover() {
        let mut graph: Graph<u8, u32> = Graph::new(0);
        let a = graph.insert(0, Some(7));
        let b = graph.insert(1, Some(7));
        let unmerged = graph.insert(0, None);
        assert_ne!(a, b);
        assert_eq!(graph.lookup(&7, 0), Some(a));
        assert_eq!(graph.lookup(&7, 1), Some(b));
        assert_eq!(graph.lookup(&8, 0), None);
        assert!(graph.vertices[unmerged as usize].key.is_none());
    }
}
