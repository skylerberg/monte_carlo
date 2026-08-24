use crate::game::{Game, PlayerSet};
use crate::node::Node;
use crate::rank::{leader_of, out_of_reach, out_of_reach_possible, Candidate};

/// Whether the candidate the root would answer with is already guaranteed, so
/// the remaining iterations cannot change the answer.
///
/// # What is proved, and what is no longer provable
///
/// The root's answer is the highest **mean reward** among the candidates that
/// cleared [`crate::rank`]'s evidence bar. A mean moves with the rewards, not
/// with the counts, so the conserved-visit bound this crate used to prove —
/// "a candidate `gap` selections behind needs `gap` more iterations to draw
/// level" — says nothing about it. Bounding a mean needs the rewards that reach
/// a node's accumulator to be confined to a known interval, and then a
/// challenger's mean after `r` further visits is at most
/// `(sum + r * max) / (visits + r)`, to be compared against the leader's floor.
///
/// **That interval does not exist in this crate, so that bound is not
/// implemented.** [`crate::Config::min_reward`] and
/// [`crate::Config::max_reward`] are a *declaration*, not a clamp:
/// `Node::record` adds `rewards.reward(node.player)` exactly as the game
/// returned it, and `duct::credit_marginals` adds the same raw number to an
/// arm. The only clamp in the crate is `duct::normalize_reward`, which is
/// applied to the copy regret matching accumulates and to nothing else, and the
/// only range check is a `debug_assert!` that fires at simultaneous nodes under
/// `RegretMatching` alone — absent from every release build, and never applied
/// to a sequential node's rewards at all. A game that leaves the defaults at
/// `[0, 1]` and pays `-1` for a loss is not a misuse the crate detects; it is a
/// case `Config::max_reward`'s own documentation calls out as likely. A bound
/// computed from `max_reward` would therefore stamp `Proven` on answers that
/// change, which is precisely the defect this function was last rewritten to
/// remove. A lost optimisation is cheaper than a false proof.
///
/// What survives is the part of the ranking that *is* counting:
/// [`crate::rank::out_of_reach`] proves only that a challenger cannot reach the
/// evidence bar with the iterations left, and a candidate under the bar cannot
/// outrank one over it. So this answers `true` only when the leader has cleared
/// the bar and **every** rival is locked under it for the rest of the search.
/// That is a real case — a narrow root whose alternatives UCB1 has stopped
/// returning to, in the last iterations of a budget — and it is a small
/// fraction of the searches that used to stop early. At a root where two
/// candidates are both well sampled, early termination is now a no-op: the
/// search spends its budget and reports [`crate::StopReason::Budget`].
///
/// `root_players` is the position's own answer to which kind of root this is,
/// and `perspective` is whose move is being proved. Both matter only at a
/// simultaneous root, where the candidates are that player's own arms rather
/// than the node's children — but which of the two sets is being proved has to
/// come from the same place `legal` came from, or the proof is about candidates
/// the answer is not drawn from.
///
/// `legal` marks which candidates are legal in the real position, parallel to
/// the root's children or to the perspective player's arms. It is not optional:
/// candidates accumulate over every determinization, so the leading one may be
/// something the player cannot actually play. Proving *that* candidate safe
/// proves nothing about the one the search will return, and the search stops
/// early on an answer it never established.
///
/// `complete` says every action the real position offers already has a
/// candidate in the tree. Without it there is nothing to prove: a later
/// determinization can be the first to offer a legal action, and a candidate
/// created with `remaining` iterations left can spend all of them, clear the
/// evidence bar, and be ranked on a mean nothing in hand constrains. A root
/// holding a single candidate would otherwise prove itself settled with no
/// rival to be out of anyone's reach.
pub(crate) fn settled<G: Game>(
    root: &Node<G::Choice>,
    root_players: Option<PlayerSet>,
    perspective: u8,
    target: u32,
    legal: &[bool],
    complete: bool,
) -> bool {
    // Spending the budget is not a proof. The loop's own budget test reports
    // that as `StopReason::Budget` on the next pass; answering `true` here
    // would stamp `Proven` on every search that runs to its last iteration.
    let remaining = target.saturating_sub(root.visits);
    if remaining == 0 || !complete {
        return false;
    }

    if root_players.is_some() {
        let Some(simul) = root.simul() else {
            return false;
        };
        // The returned move is a draw from a distribution, and proving that the
        // leading candidate cannot be overtaken says nothing about a draw.
        // Only an argmax root policy is what this proves.
        if G::SIMULTANEOUS_POLICY.mixes() {
            return false;
        }
        debug_assert!(
            simul.arm_policy.is_empty(),
            "mcts: the early-termination proof shows that the leading candidate \
             cannot be overtaken, which is only the returned answer while the root \
             policy is an argmax. Under a mixing policy the answer is a draw from a \
             distribution and the proof says nothing about it."
        );
        let Some(slot) = simul.players.slot_of(perspective) else {
            return false;
        };
        // One iteration bumps exactly one arm per player, so an arm's selections
        // are a conserved budget just as a child's are and the tier bound
        // carries over verbatim.
        let arms = &simul.arm_stats[simul.slot_range(slot)];
        // A mask built before the last arm was created describes a different
        // candidate set. Refusing to prove is always safe; proving against a
        // stale mask is not.
        if legal.len() != arms.len() {
            return false;
        }
        // Restricting both sides of the comparison to the legal arms is what
        // makes the leader the arm actually returned; proving something about
        // an arm the position withholds proves nothing about the answer.
        return proven(
            arms.iter()
                .zip(legal)
                .enumerate()
                .filter(|(_, (_, &legal))| legal)
                .map(|(i, (arm, _))| {
                    (
                        i,
                        Candidate::new(arm.visits, arm.availability, arm.cumulative_reward),
                    )
                }),
            remaining,
        );
    }

    let children = root.children();
    if legal.len() != children.len() {
        return false;
    }
    proven(
        children
            .iter()
            .zip(legal)
            .enumerate()
            .filter(|(_, (_, &legal))| legal)
            .map(|(i, (child, _))| {
                (
                    i,
                    Candidate::new(child.visits, child.availability, child.cumulative_reward),
                )
            }),
        remaining,
    )
}

/// Whether the candidate [`leader_of`] would answer with is out of every
/// rival's reach.
///
/// Two passes over the same iterator rather than one over a materialized slice:
/// this runs once per iteration on the hot path, and neither a root's children
/// nor a slot's arms is a slice of the statistics the ranking reads.
fn proven<I>(mut candidates: I, remaining: u32) -> bool
where
    I: Iterator<Item = (usize, Candidate)> + Clone,
{
    // While every challenger can still reach the evidence bar, the only root
    // this proves is one with no challenger on it, and both scans below are a
    // pass over every root candidate spent to answer `false`. This runs once
    // per iteration on the hot path and the proof can shorten a budget by at
    // most `MIN_EVIDENCE - 1` iterations, so the scan it saves is most of what
    // early termination costs at a wide root.
    if !out_of_reach_possible(remaining) {
        let mut sole = candidates.clone();
        return matches!(sole.next(), Some((_, only)) if only.visits() > 0)
            && sole.next().is_none();
    }
    let Some((at, leader)) = leader_of(candidates.clone()) else {
        return false;
    };
    candidates.all(|(i, challenger)| i == at || out_of_reach(&leader, &challenger, remaining))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Status;
    use crate::node::Node;
    use rand_core::{Rng, SeedableRng};
    use wyrand::WyRand;

    /// A sequential one-ply game, present only because [`settled`] is generic
    /// over one. Nothing here is searched: the roots below are built by hand so
    /// that the counts are exactly the ones under test.
    #[derive(Clone, Default)]
    struct Ply;

    impl Game for Ply {
        type Choice = u8;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();
        fn status(&self, _: &()) -> Status<[f64; 2]> {
            Status::Active { player: 0 }
        }
        fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
            out.extend([0, 1, 2]);
        }
        fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {}
        fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
            [0.0, 0.0]
        }
        fn new_buffer(&self) -> Self {
            Self
        }
        fn determinize_into<R: Rng + ?Sized>(&self, _: &mut Self, _: &(), _: u8, _: &mut R) {}
    }

    /// A root of three children carrying `(visits, availability, mean)`, and the
    /// visits they add up to.
    fn root(children: [(u32, u32, f64); 3]) -> Node<u8> {
        let mut root = Node::new_root(0);
        let mut avail = Vec::new();
        root.expand(
            &[0, 1, 2],
            0,
            usize::MAX,
            &mut avail,
            &mut WyRand::seed_from_u64(1),
        );
        for (choice, &(visits, availability, mean)) in children.iter().enumerate() {
            let child = &mut root.children[choice];
            child.visits = visits;
            child.availability = availability;
            child.cumulative_reward = visits as f64 * mean;
            root.visits += visits;
        }
        root
    }

    /// The one shape the surviving proof fires on: one candidate over the
    /// evidence bar and every rival locked under it by the iterations left.
    /// Child 2 leads; children 0 and 1 cannot reach the bar in the five
    /// iterations `TARGET` leaves.
    const CHILDREN: [(u32, u32, f64); 3] = [(10, 400, 0.9), (5, 400, 0.2), (390, 400, 0.5)];
    const TARGET: u32 = 410;

    /// The candidates the proof ranks are the ones the answer is drawn from.
    /// Child 2 is the only one over the evidence bar and it is a move this
    /// position does not offer: proving *it* safe says nothing about which of
    /// the two the player actually holds comes back, and neither of those two
    /// has cleared the bar itself.
    #[test]
    fn a_leader_the_position_withholds_proves_nothing() {
        let root = root(CHILDREN);
        assert!(!settled::<Ply>(
            &root,
            None,
            0,
            TARGET,
            &[true, true, false],
            true
        ));
        // The same counts with that child legal are a proof, which is what makes
        // the mask above the reason and not a coincidence.
        assert!(settled::<Ply>(
            &root,
            None,
            0,
            TARGET,
            &[true, true, true],
            true
        ));
    }

    /// A challenger that can still reach the evidence bar is ranked on a mean,
    /// and no count bounds a mean. One iteration either side of the bar is the
    /// whole of the surviving proof.
    #[test]
    fn a_challenger_that_can_still_reach_the_bar_is_not_out_of_reach() {
        let root = root(CHILDREN);
        // Child 0 holds the most selections of the two under the bar, so it is
        // the one that decides: it reaches the bar with `MIN_EVIDENCE - 10`
        // iterations left, and cannot with one fewer.
        let spent: u32 = CHILDREN.iter().map(|&(visits, ..)| visits).sum();
        let reachable = spent + (crate::rank::MIN_EVIDENCE - CHILDREN[0].0);
        assert!(!settled::<Ply>(
            &root,
            None,
            0,
            reachable,
            &[true, true, true],
            true
        ));
        assert!(settled::<Ply>(
            &root,
            None,
            0,
            reachable - 1,
            &[true, true, true],
            true
        ));
    }

    /// A mask of the wrong length describes some other candidate set, and a
    /// proof read against it is a proof about nothing. Refusing is always safe.
    #[test]
    fn a_mask_of_the_wrong_length_proves_nothing() {
        let root = root(CHILDREN);
        assert!(!settled::<Ply>(&root, None, 0, TARGET, &[true, true], true));
        assert!(!settled::<Ply>(
            &root,
            None,
            0,
            TARGET,
            &[true, true, true, true],
            true
        ));
    }

    /// A legal move with no candidate in the tree yet can spend every remaining
    /// iteration, clear the bar, and be ranked on a mean nothing in hand
    /// constrains.
    #[test]
    fn an_incomplete_candidate_set_proves_nothing() {
        let root = root(CHILDREN);
        assert!(!settled::<Ply>(
            &root,
            None,
            0,
            TARGET,
            &[true, true, true],
            false
        ));
    }

    /// The shortcut that keeps early termination from scanning every root
    /// candidate on every iteration must answer exactly what the scan would.
    ///
    /// With more iterations left than the evidence bar is high, no challenger
    /// is out of reach, so the only root that proves is one holding a single
    /// legal candidate — and that one still proves, however long is left.
    #[test]
    fn a_lone_legal_candidate_is_proved_at_any_remaining() {
        let sole = root(CHILDREN);
        let spent: u32 = CHILDREN.iter().map(|&(visits, ..)| visits).sum();
        let far = spent + 10 * crate::rank::MIN_EVIDENCE;
        assert!(settled::<Ply>(
            &sole,
            None,
            0,
            far,
            &[false, false, true],
            true
        ));
        // An unvisited lone candidate is no answer, so it is no proof either.
        let empty = root([(0, 400, 0.0), (5, 400, 0.2), (390, 400, 0.5)]);
        assert!(!settled::<Ply>(
            &empty,
            None,
            0,
            far,
            &[true, false, false],
            true
        ));
        // Add one rival and the same budget proves nothing at all.
        assert!(!settled::<Ply>(
            &sole,
            None,
            0,
            far,
            &[false, true, true],
            true
        ));
    }

    /// Spending the budget is not a proof; the loop's own test reports that as
    /// `StopReason::Budget`.
    #[test]
    fn a_spent_budget_is_not_a_proof() {
        let root = root(CHILDREN);
        assert!(!settled::<Ply>(
            &root,
            None,
            0,
            405,
            &[true, true, true],
            true
        ));
    }
}
