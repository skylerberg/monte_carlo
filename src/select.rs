use crate::node::Node;

/// The count a child's exploration term is measured against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ExplorationCount {
    ParentVisits,
    TimesAvailable,
}

impl ExplorationCount {
    /// Textbook UCB1 at the root and wherever choices never change, the ISMCTS
    /// availability rule everywhere else.
    ///
    /// The root's exception is deliberate and is not a place to "restore"
    /// consistency: nothing maintains `ln_times_available` on a root child,
    /// since that would be a libm call per child per iteration on the widest
    /// loop in the search, and `reroot_at` resets the field on promotion, so
    /// reading it there would score the root against a stale zero. The
    /// correction the root does need is in `rank`, which ranks the answer on a
    /// mean and breaks ties on visits over `times_available`; the exploration
    /// term is what apportions the budget between children the root offers on
    /// every iteration anyway.
    #[inline(always)]
    pub(crate) fn of(at_root: bool, choices_fixed: bool) -> Self {
        if at_root || choices_fixed {
            Self::ParentVisits
        } else {
            Self::TimesAvailable
        }
    }
}

/// UCB1 with an optional progressive-bias prior.
///
/// `ln_total` is the log of whichever [`ExplorationCount`] the node uses. The
/// Information Set MCTS variant, a child's own `times_available`, stops a
/// rarely-legal choice from looking under-explored.
///
/// Kept as one function so that its callers can never disagree about the
/// formula. There are two: `select`, and `duct::select_duct`, which scores a
/// simultaneous node's arms against their own availability and is the same
/// bandit applied to a different denominator. It passes a zero bias weight
/// rather than omitting the term — see the comment at that call site for why
/// the prior has no meaning at a simultaneous node.
///
/// A zero weight returns before the bias term rather than zeroing it: IEEE
/// rules forbid folding `0.0 * h / (1.0 + visits)` or the add, so the term
/// would otherwise cost a divide per child on every selection.
#[inline(always)]
pub(crate) fn ucb_raw(
    cumulative_reward: f64,
    visits: f64,
    ln_total: f64,
    exploration: f64,
    bias_weight: f64,
    heuristic_bias: f32,
) -> f64 {
    let value = cumulative_reward / visits + exploration * (ln_total / visits).sqrt();
    if bias_weight == 0.0 {
        return value;
    }
    value + bias_weight * heuristic_bias as f64 / (1.0 + visits)
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
    exploration_count: ExplorationCount,
    exploration: f64,
    bias_weight: f64,
) -> Option<usize> {
    debug_assert_eq!(avail.len(), node.children.len());

    let ln_parent_visits = match exploration_count {
        ExplorationCount::ParentVisits => (node.visits as f64).ln(),
        ExplorationCount::TimesAvailable => 0.0,
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
        let ln_total = match exploration_count {
            ExplorationCount::ParentVisits => ln_parent_visits,
            ExplorationCount::TimesAvailable => child.ln_times_available,
        };
        let value = ucb(child, ln_total, exploration, bias_weight);
        if value > best_value {
            best_value = value;
            best = Some(i);
        }
    }
    best
}
