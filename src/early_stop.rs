use crate::game::Game;
use crate::node::Node;
use crate::search::Config;
use crate::select::ucb_raw;

/// How often the expensive check runs, in root visits.
const CHECK_INTERVAL: u32 = 1024;
/// Ceiling on `remaining * children` for the expensive check. The simulation is
/// linear in both, so a node with thousands of children would spend more time
/// proving it can stop than it would save by stopping.
///
/// Consulted at a sequential root only, and unchanged there. A simultaneous root
/// never reaches the expensive check at all.
const MAX_SIMULATION_WORK: u64 = 2_000_000;

/// Whether the move with the most visits is already guaranteed, so the
/// remaining iterations cannot change the answer.
///
/// `perspective` is whose move is being proved. It matters only at a
/// simultaneous root, where the candidates are that player's own arms rather
/// than the node's children.
///
/// `legal` marks which of those arms are legal in the real position, parallel to
/// the perspective player's slot, and is ignored at a sequential root. It is not
/// optional: arms accumulate over every determinization, so the arm with the
/// most visits may be one the player cannot actually play. Proving *that* arm
/// safe proves nothing about the one the search will return, and the search
/// stops early on an answer it never established.
pub(crate) fn settled<G: Game>(
    root: &Node<G::Choice>,
    perspective: u8,
    target: u32,
    cfg: &Config,
    legal: &[bool],
) -> bool {
    // Before the shared budget check, not after: under a mixing policy nothing
    // here is ever an answer, and reporting `StopReason::Proven` on the
    // iteration that happens to exhaust the budget would contradict that.
    if let Some(simul) = root.simul() {
        // The returned move is a draw from a distribution, and proving that the
        // most-visited candidate cannot be overtaken says nothing about a draw.
        // Only an argmax root policy is what this proves.
        if G::SIMULTANEOUS_POLICY.mixes() {
            return false;
        }
        debug_assert!(
            simul.arm_policy.is_empty(),
            "mcts: the early-termination proof shows that the most-visited candidate \
             cannot be overtaken, which is only the returned answer while the root \
             policy is an argmax. Under a mixing policy the answer is a draw from a \
             distribution and the proof says nothing about it."
        );
        let remaining = target.saturating_sub(root.visits);
        if remaining == 0 {
            return true;
        }
        let Some(slot) = simul.players.slot_of(perspective) else {
            return false;
        };
        // One iteration bumps exactly one arm per player, so arm visits are a
        // conserved budget just as child visits are and the cheap proof carries
        // over verbatim. The expensive one does not: its replay uses
        // `ln(total)` where arm selection uses each arm's own
        // `ln_availability`, and under determinization those diverge.
        let arms = &simul.arm_stats[simul.slot_range(slot)];
        // A mask built before the last arm was created describes a different
        // candidate set. Refusing to prove is always safe; proving against a
        // stale mask is not.
        if legal.len() != arms.len() {
            return false;
        }
        // One iteration bumps exactly one arm per player, so a challenger can
        // close at most `remaining` visits whether or not the arms it passes
        // through are legal here. Restricting both sides of the comparison to
        // the legal arms is what makes the leader the arm actually returned.
        let (best, second) = top_two_visits(
            arms.iter()
                .zip(legal)
                .filter(|(_, &legal)| legal)
                .map(|(arm, _)| arm.visits),
        );
        return best.saturating_sub(second) > remaining;
    }

    let remaining = target.saturating_sub(root.visits);
    if remaining == 0 {
        return true;
    }

    let (best, second) = top_two_visits(root.children().iter().map(|child| child.visits()));
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

/// The two highest visit counts among the candidates, in order.
///
/// Takes the counts rather than the nodes so that one implementation serves a
/// sequential root's children and a simultaneous root's arms — neither of which
/// is a slice of `u32`, and materializing one per iteration would allocate on
/// the hot path.
fn top_two_visits(candidates: impl Iterator<Item = u32>) -> (u32, u32) {
    let mut best = 0;
    let mut second = 0;
    for visits in candidates {
        if visits > best {
            second = best;
            best = visits;
        } else if visits > second {
            second = visits;
        }
    }
    (best, second)
}

/// Which candidate holds the most visits, first maximum. See
/// [`top_two_visits`] for why it takes counts.
fn leader_index(candidates: impl Iterator<Item = u32>) -> usize {
    let mut leader = 0;
    let mut best = 0;
    for (i, visits) in candidates.enumerate() {
        if visits > best {
            best = visits;
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
///
/// Sequential roots only: the replay is UCB1 over children, which is neither the
/// rule nor the candidate set at a simultaneous node.
fn can_challenger_overtake<C>(root: &Node<C>, remaining: u32, cfg: &Config) -> bool {
    let leader = leader_index(root.children().iter().map(|child| child.visits()));
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
