//! Read-only access to a searched graph.

use crate::game::PlayerSet;
use crate::graph::node::{Graph, Link, Vertex, UNRESOLVED};
use crate::node::Marginals;

impl<C, K> Graph<C, K> {
    /// The position the search was run from.
    pub fn root(&self) -> GraphNode<'_, C, K> {
        GraphNode {
            graph: self,
            id: self.root,
        }
    }

    /// The position with this [`GraphNode::id`], if there is one.
    pub fn node(&self, id: usize) -> Option<GraphNode<'_, C, K>> {
        (id < self.vertices.len()).then_some(GraphNode {
            graph: self,
            id: id as u32,
        })
    }

    /// Positions in the graph, each counted once however many routes reach it.
    pub fn node_count(&self) -> usize {
        self.vertices.len()
    }

    /// Links some iteration has taken, sequential and joint. A link the search
    /// has opened but never taken leads nowhere yet and is not counted.
    pub fn edge_count(&self) -> usize {
        self.vertices
            .iter()
            .map(|vertex| vertex.targets().filter(|&t| t != UNRESOLVED).count())
            .sum()
    }

    /// Longest path from the root, in positions.
    ///
    /// A graph can hold repetitions, where a longest path is unbounded, so a
    /// link back into a position already on the walk is not followed: the depth
    /// is that of the graph with its back edges removed, which is the graph
    /// itself whenever the game cannot repeat a position.
    pub fn max_depth(&self) -> usize {
        let count = self.vertices.len();
        let mut depth = vec![0u32; count];
        let mut on_stack = vec![false; count];
        let mut stack: Vec<(u32, usize)> = vec![(self.root, 0)];
        on_stack[self.root as usize] = true;
        while let Some(&mut (at, ref mut next)) = stack.last_mut() {
            let vertex = &self.vertices[at as usize];
            match target_at(vertex, *next) {
                Some(target) => {
                    *next += 1;
                    if target == UNRESOLVED
                        || on_stack[target as usize]
                        || depth[target as usize] != 0
                    {
                        continue;
                    }
                    on_stack[target as usize] = true;
                    stack.push((target, 0));
                }
                None => {
                    // A successor still on the stack is a back edge and reads 0.
                    let deepest = vertex
                        .targets()
                        .filter(|&t| t != UNRESOLVED)
                        .map(|t| depth[t as usize])
                        .max()
                        .unwrap_or(0);
                    depth[at as usize] = 1 + deepest;
                    on_stack[at as usize] = false;
                    stack.pop();
                }
            }
        }
        depth[self.root as usize] as usize
    }
}

/// The `k`-th successor of `vertex`, links first and then joint successors.
fn target_at<C, K>(vertex: &Vertex<C, K>, k: usize) -> Option<u32> {
    if let Some(link) = vertex.links.get(k) {
        return Some(link.target);
    }
    let joint = vertex.joint()?;
    joint.targets.get(k - vertex.links.len()).copied()
}

/// One position in a [`Graph`].
pub struct GraphNode<'a, C, K> {
    graph: &'a Graph<C, K>,
    id: u32,
}

// Hand-written: deriving would bound the impls on `C` and `K`, and the view
// holds nothing but a reference.
impl<C, K> Clone for GraphNode<'_, C, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C, K> Copy for GraphNode<'_, C, K> {}

impl<'a, C, K> GraphNode<'a, C, K> {
    #[inline]
    fn vertex(&self) -> &'a Vertex<C, K> {
        &self.graph.vertices[self.id as usize]
    }

    /// This position's index in its graph, for [`Graph::node`]. Two views of
    /// one position always share it. Re-rooting renumbers every position.
    pub fn id(&self) -> usize {
        self.id as usize
    }

    /// Whether this is the position the search was run from.
    pub fn is_root(&self) -> bool {
        self.id == self.graph.root
    }

    /// Iterations that entered this position, by every route into it.
    pub fn visits(&self) -> u32 {
        self.vertex().visits
    }

    /// Mean reward for [`GraphNode::reward_player`] over every iteration that
    /// entered this position, or 0.0 if none has.
    pub fn mean_reward(&self) -> f64 {
        self.vertex().mean_reward()
    }

    /// Total reward accumulated for [`GraphNode::reward_player`].
    pub fn cumulative_reward(&self) -> f64 {
        self.vertex().cumulative_reward
    }

    /// The player whose reward this position accumulates: whoever moved into
    /// it — every route into a merged position shares that player — or, at the
    /// root, the player to act, as in [`crate::Node::reward_player`].
    pub fn reward_player(&self) -> u8 {
        self.vertex().player
    }

    /// The key this position was merged under, or `None` if it never was.
    pub fn key(&self) -> Option<&'a K> {
        self.vertex().key.as_ref()
    }

    /// The links out of this position, in the order the search opened them.
    /// Empty at a simultaneous position; see [`GraphNode::joint_successors`].
    pub fn edges(&self) -> impl ExactSizeIterator<Item = GraphEdge<'a, C, K>> + 'a {
        let graph = self.graph;
        self.vertex()
            .links
            .iter()
            .map(move |link| GraphEdge { graph, link })
    }

    /// The participants, if this is a simultaneous position.
    pub fn simultaneous_players(&self) -> Option<PlayerSet> {
        self.vertex().joint().map(|joint| joint.simul.players)
    }

    /// `player`'s bandit over their own actions here, or `None` if this is not
    /// a simultaneous position or `player` does not act at it.
    pub fn marginals(&self, player: u8) -> Option<Marginals<'a, C>> {
        let simul = &self.vertex().joint()?.simul;
        let slot = simul.players.slot_of(player)?;
        Some(simul.marginals(slot))
    }

    /// The successor each joint action taken here led to, in the order the
    /// joint actions were first taken. `None` for one taken before a re-root
    /// cleared it.
    pub fn joint_successors(
        &self,
    ) -> impl ExactSizeIterator<Item = Option<GraphNode<'a, C, K>>> + 'a {
        let graph = self.graph;
        let targets = self
            .vertex()
            .joint()
            .map_or(&[][..], |joint| joint.targets.as_slice());
        targets
            .iter()
            .map(move |&target| (target != UNRESOLVED).then_some(GraphNode { graph, id: target }))
    }

    /// The index, into `self.marginals(player)`, of the action `player` took
    /// to reach `self.joint_successors()[successor]`.
    pub fn joint_arm(&self, successor: usize, player: u8) -> Option<usize> {
        let simul = &self.vertex().joint()?.simul;
        let slot = simul.players.slot_of(player)?;
        let key = *simul.joint_keys.get(successor)?;
        Some(key.arm(slot, simul.key_bits) as usize)
    }
}

/// One link out of a position: a choice, and what taking it led to.
pub struct GraphEdge<'a, C, K> {
    graph: &'a Graph<C, K>,
    link: &'a Link<C>,
}

impl<C, K> Clone for GraphEdge<'_, C, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C, K> Copy for GraphEdge<'_, C, K> {}

impl<'a, C, K> GraphEdge<'a, C, K> {
    /// The choice this link plays.
    pub fn choice(&self) -> &'a C {
        &self.link.choice
    }

    /// Times the search took this link from this position — not the
    /// successor's visits, which count every route into it.
    pub fn visits(&self) -> u32 {
        self.link.visits
    }

    /// Iterations at this position in which the choice was legal.
    pub fn times_available(&self) -> u32 {
        self.link.times_available
    }

    /// The position this link leads to, or `None` if no iteration has taken it.
    pub fn target(&self) -> Option<GraphNode<'a, C, K>> {
        (self.link.target != UNRESOLVED).then_some(GraphNode {
            graph: self.graph,
            id: self.link.target,
        })
    }
}
