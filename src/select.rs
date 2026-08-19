use crate::node::Node;

/// UCB1 with an optional progressive-bias prior.
///
/// `ln_total` is the parent's visit count at the root, and the child's own
/// availability count elsewhere — the Information Set MCTS variant, which stops
/// a rarely-legal choice from looking under-explored.
///
/// Kept as one function so that `select` and the early-termination simulation
/// can never disagree about the formula.
#[inline(always)]
pub(crate) fn ucb_raw(
    cumulative_reward: f64,
    visits: f64,
    ln_total: f64,
    exploration: f64,
    bias_weight: f64,
    heuristic_bias: f32,
) -> f64 {
    cumulative_reward / visits
        + exploration * (ln_total / visits).sqrt()
        + bias_weight * heuristic_bias as f64 / (1.0 + visits)
}

#[inline]
pub(crate) fn ucb<C>(child: &Node<C>, ln_total: f64, exploration: f64, bias_weight: f64) -> f64 {
    ucb_raw(
        child.cumulative_reward,
        child.visits as f64,
        ln_total,
        exploration,
        bias_weight,
        child.heuristic_bias,
    )
}

/// Index of the child to descend into, or `None` if nothing is legal here.
///
/// Returning on the first unvisited child is equivalent to giving unvisited
/// children an infinite value and taking the first maximum, but it costs two
/// iterations instead of a full scan — which matters at nodes with thousands of
/// children, where the tree spends most of the search still opening them.
pub(crate) fn select<C>(
    node: &Node<C>,
    avail: &[bool],
    exploration: f64,
    bias_weight: f64,
) -> Option<usize> {
    debug_assert_eq!(avail.len(), node.children.len());

    let is_root = node.is_root();
    let root_ln = if is_root {
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
        let ln_total = if is_root {
            root_ln
        } else {
            child.ln_availability
        };
        let value = ucb(child, ln_total, exploration, bias_weight);
        if value > best_value {
            best_value = value;
            best = Some(i);
        }
    }
    best
}
