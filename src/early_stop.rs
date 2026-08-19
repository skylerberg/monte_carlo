use crate::node::Node;
use crate::search::Config;
use crate::select::ucb_raw;

/// How often the expensive check runs, in root visits.
const CHECK_INTERVAL: u32 = 1024;
/// Ceiling on `remaining * children` for the expensive check. The simulation is
/// linear in both, so a node with thousands of children would spend more time
/// proving it can stop than it would save by stopping.
const MAX_SIMULATION_WORK: u64 = 2_000_000;

/// Whether the move with the most visits is already guaranteed, so the
/// remaining iterations cannot change the answer.
pub(crate) fn settled<C>(root: &Node<C>, target: u32, cfg: &Config) -> bool {
    let remaining = target.saturating_sub(root.visits);
    if remaining == 0 {
        return true;
    }

    let (best, second) = top_two_visits(root);
    let gap = best.saturating_sub(second);
    if gap > remaining {
        return true;
    }

    if !root.visits.is_multiple_of(CHECK_INTERVAL) || gap <= remaining / 4 {
        return false;
    }
    if remaining as u64 * root.children.len() as u64 > MAX_SIMULATION_WORK {
        return false;
    }
    !can_challenger_overtake(root, remaining, cfg)
}

fn top_two_visits<C>(root: &Node<C>) -> (u32, u32) {
    let mut best = 0;
    let mut second = 0;
    for child in root.children() {
        let visits = child.visits();
        if visits > best {
            second = best;
            best = visits;
        } else if visits > second {
            second = visits;
        }
    }
    (best, second)
}

fn leader_index<C>(root: &Node<C>) -> usize {
    let mut leader = 0;
    let mut best = 0;
    for (i, child) in root.children().iter().enumerate() {
        if child.visits() > best {
            best = child.visits();
            leader = i;
        }
    }
    leader
}

/// Replay the remaining iterations under the most favourable outcome any
/// challenger could hope for — every challenger gets `max_reward`, the leader
/// gets `min_reward` — and report whether one of them catches up on visits.
///
/// The reward bounds come from the config rather than being hardcoded to 1.0 and
/// 0.0: a game with shaped rewards outside `[0, 1]` would otherwise get an
/// unsound "cannot change" proof.
fn can_challenger_overtake<C>(root: &Node<C>, remaining: u32, cfg: &Config) -> bool {
    let leader = leader_index(root);
    let count = root.children().len();

    let mut visits: Vec<u32> = root.children().iter().map(|c| c.visits()).collect();
    let mut rewards: Vec<f64> = root.children.iter().map(|c| c.cumulative_reward).collect();
    let biases: Vec<f32> = root.children.iter().map(|c| c.heuristic_bias).collect();
    let mut total = root.visits();

    for _ in 0..remaining {
        total += 1;
        let ln_total = (total as f64).ln();

        let mut chosen = 0;
        let mut best_value = f64::NEG_INFINITY;
        for i in 0..count {
            let value = if visits[i] == 0 {
                f64::INFINITY
            } else {
                ucb_raw(
                    rewards[i],
                    visits[i] as f64,
                    ln_total,
                    cfg.exploration_constant,
                    cfg.progressive_bias_weight,
                    biases[i],
                )
            };
            if value > best_value {
                best_value = value;
                chosen = i;
            }
        }

        visits[chosen] += 1;
        rewards[chosen] += if chosen == leader {
            cfg.min_reward
        } else {
            cfg.max_reward
        };
    }

    let leader_visits = visits[leader];
    (0..count).any(|i| i != leader && visits[i] >= leader_visits)
}
