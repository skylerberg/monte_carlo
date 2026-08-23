use crate::game::Game;
use crate::node::Node;

/// Whether the move with the most visits is already guaranteed, so the
/// remaining iterations cannot change the answer.
///
/// The proof is the conserved-visit bound and nothing else: one iteration adds
/// exactly one visit to exactly one candidate, so a candidate `gap` visits
/// behind the leader needs at least `gap` more iterations to draw level. That
/// bound also covers the candidates that do not exist yet — a child a later
/// determinization first offers starts from zero visits — which is what a
/// simulation over the candidates present today cannot do.
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
    legal: &[bool],
) -> bool {
    // Spending the budget is not a proof. The loop's own budget test reports
    // that as `StopReason::Budget` on the next pass; answering `true` here
    // would stamp `Proven` on every search that runs to its last iteration.
    let remaining = target.saturating_sub(root.visits);
    if remaining == 0 {
        return false;
    }

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
        let Some(slot) = simul.players.slot_of(perspective) else {
            return false;
        };
        // One iteration bumps exactly one arm per player, so arm visits are a
        // conserved budget just as child visits are and the bound carries over
        // verbatim.
        let arms = &simul.arm_stats[simul.slot_range(slot)];
        // A mask built before the last arm was created describes a different
        // candidate set. Refusing to prove is always safe; proving against a
        // stale mask is not.
        if legal.len() != arms.len() {
            return false;
        }
        // A challenger can close at most `remaining` visits whether or not the
        // arms it passes through are legal here. Restricting both sides of the
        // comparison to the legal arms is what makes the leader the arm
        // actually returned.
        let (best, second) = top_two_visits(
            arms.iter()
                .zip(legal)
                .filter(|(_, &legal)| legal)
                .map(|(arm, _)| arm.visits),
        );
        return best.saturating_sub(second) > remaining;
    }

    let (best, second) = top_two_visits(root.children().iter().map(|child| child.visits()));
    best.saturating_sub(second) > remaining
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
