//! The one rule by which a root's answer is chosen, and the one bound by which
//! it is proved.
//!
//! Every argmax-family root reads its answer off [`leader_of`]: the sequential
//! root, the simultaneous root under [`crate::SimultaneousPolicy::Duct`], and
//! `RootParallel`'s pooled merge. They must agree, or one position has three
//! answers depending on which surface asked. So must the early-termination
//! proof, which is [`out_of_reach`] applied to the same scan — a proof about
//! some other candidate proves nothing about the answer.
//!
//! # The rule
//!
//! Among the candidates the search selected at least [`MIN_EVIDENCE`] times,
//! the highest **mean reward** wins. The **selection rate** — visits over
//! availability — only breaks a tie between two means. A candidate that has
//! cleared the bar outranks one that has not, whatever either measured, and
//! under the bar the same two statistics decide in the same order.
//!
//! # Why the answer is not the most visited candidate
//!
//! A candidate is selectable at most once per iteration in which it was legal,
//! so `visits <= availability` is a hard cap and raw visits rank by *legality
//! rate* as much as by value. A move legal in a third of the determinizations
//! and played on nearly every iteration that offered it loses the argmax to one
//! legal always and played on two thirds, however much better it is — and no
//! budget closes that, because the gap is structural. The count is a
//! measurement of how often the search *could* return to a candidate, not of
//! what it found there.
//!
//! # Why it is not the selection rate either
//!
//! Dividing the count by the opportunities that carried it removes the legality
//! bias, and for one round of this crate's history that was the rule. It is
//! still wrong, and for a reason that no discount of the rate repairs: **the
//! rate saturates at 1.0.** Selection takes an unvisited candidate on sight and
//! then keeps taking it for as long as its exploration bonus covers the gap to
//! the leader — about `(c / gap)^2 * ln(iterations)` selections, a budget that
//! depends on how good the candidate is and not at all on how often it was
//! legal. A candidate whose whole availability window is shorter than that
//! budget is therefore selected on essentially every iteration that offered it
//! *whatever it is worth*, and its rate lands against the ceiling with no room
//! left above it for a better candidate to climb into.
//!
//! Measured: a decoy offered by one determinization in a hundred reaches 97
//! visits against 97 opportunities — a rate of exactly 1.000, and enough
//! selections to clear any evidence bar worth having — while the genuinely best
//! move, which the root must split its budget with a rival to check, sits at
//! 9873 / 9999 = 0.987. The rate ranks the decoy first and the search returns a
//! move worth 0.75 in place of one worth 0.90, in every seed, stamped `Proven`.
//! The bar does not help, because the decoy is over it. Nothing that reads the
//! rate as the primary key can survive that fixture, so the rate is not the
//! primary key.
//!
//! The mean has no such ceiling. It is measured on the reward scale the caller
//! declared, it is the quantity the root is actually choosing between, and a
//! better candidate can always out-score a worse one on it however the
//! legality fell.
//!
//! # Why there is still a bar
//!
//! Ranking on the mean alone is the max-child rule that robust-child extraction
//! exists to avoid: it hands the root to whichever candidate is *furthest from
//! having been checked*, because one lucky rollout is a mean of 1.0 and no
//! amount of contrary evidence elsewhere touches it. [`MIN_EVIDENCE`]
//! selections is the bar that makes the rule robust rather than optimistic, and
//! a candidate that has cleared it outranks one that has not.
//!
//! Under the bar the mean still leads, because there is nothing better to rank
//! on down there — the rate is manufactured for exactly the candidates that
//! live under it, and a candidate offered twice and taken twice scores the same
//! 1.0 as one offered two thousand times and never refused. Every root wider
//! than its own iteration budget lives entirely under the bar and answers with
//! the best value it saw, which is the right answer there: with one visit each,
//! the visit count and the rate are constants and the mean is the only
//! statistic that says anything.
//!
//! # What it gives up
//!
//! Two things, and the second is the larger.
//!
//! A better move the search selected fewer than [`MIN_EVIDENCE`] times cannot
//! win the root over a move that cleared the bar, however well it did on those
//! few. Unlike the raw visit count, which no budget could correct, this is a
//! gap more search closes: the bar is fixed and the opportunities keep coming.
//!
//! The bar is also a count and not a confidence interval, so what it rules out
//! is the stranger with an anecdote and not the stranger with a small sample.
//! Measured on a candidate one determinization in 250 offers, against an
//! always-legal rival worth 0.50 exactly: at 10,000 iterations the candidate
//! clears the bar on about 40 selections of a payoff that is a coin flip either
//! side of its mean, and a mean of 0.45 takes the root from the rival in 54 of
//! 200 seeds. It is the price of reading a mean off the fewest selections that
//! count as evidence, and it is bounded by the gap it misreads: over the same
//! sweep, from 0.20 to 0.80, this ranking's expected value beats the visit
//! argmax's everywhere the candidate is genuinely better — by up to 0.27 of the
//! reward scale — and trails it by at most 0.014 in the band just below parity.
//! More search closes this one too, on the same opportunities.
//!
//! And most of the early-termination proof. A count is conserved — the
//! remaining iterations bound what any count can still do — but a mean moves
//! with the *rewards*, and bounding those needs a guarantee that the crate does
//! not have. See [`out_of_reach`], which refuses to prove anything the counts
//! alone do not settle.

use core::cmp::Ordering;

/// How many selections a candidate needs before its mean is read as a
/// measurement rather than as whatever the first few rollouts happened to pay.
///
/// The one tuned number in the ranking, and it is tuned to the shape of the
/// exploration bonus rather than to a fixture: `(c / gap)^2 * ln(iterations)`
/// — the selections UCB1 spends on a candidate before its mean can put it
/// down — is 32 at this crate's default exploration constant, a candidate two
/// fifths of the reward scale behind the leader, and a ten-thousand-iteration
/// budget. A candidate the search has returned to that many times has been
/// given its chance to be refuted; one it has not is a stranger with an
/// anecdote.
pub(crate) const MIN_EVIDENCE: u32 = 32;

/// One root candidate's statistics: a sequential root's child, a simultaneous
/// root's marginal arm, or a `RootParallel` merge's pooled entry.
///
/// Private fields: [`Candidate::new`] is the only way in, and it is where the
/// one relationship the ranking depends on is checked.
#[derive(Clone, Copy)]
pub(crate) struct Candidate {
    visits: u32,
    /// Iterations in which this candidate was legal, over the same window that
    /// produced `visits`. Numerator and denominator must describe the same
    /// window or the rate is meaningless, which is why re-rooting carries both
    /// forward or neither.
    availability: u32,
    cumulative_reward: f64,
}

impl Candidate {
    pub(crate) fn new(visits: u32, availability: u32, cumulative_reward: f64) -> Self {
        debug_assert!(
            visits <= availability,
            "mcts: a root candidate holds {visits} selections against {availability} \
             opportunities to make them. A selection is only possible on an iteration \
             that offered the candidate, so the two counts have come from different \
             windows and the rate this ranking breaks its ties with is meaningless."
        );
        Self {
            visits,
            availability,
            cumulative_reward,
        }
    }

    /// Opportunities, never fewer than the selections they carried. The
    /// assertion in [`Candidate::new`] is the real guard; this is what a release
    /// build does if that guard is ever wrong, and it degrades the tiebreak to
    /// the visit argmax rather than to a rate above 1 that would win every tie
    /// against an honest candidate.
    fn opportunities(&self) -> u64 {
        self.availability.max(self.visits).max(1) as u64
    }

    /// Selections this candidate has taken.
    pub(crate) fn visits(&self) -> u32 {
        self.visits
    }

    /// Whether the search sampled this candidate often enough for its mean to
    /// mean anything. See [`MIN_EVIDENCE`].
    fn established(&self) -> bool {
        self.visits >= MIN_EVIDENCE
    }

    /// The two selection rates compared as a cross product, in integers: no
    /// division, no rounding, and the same order on every target. Only ever
    /// reached on an exact tie between two means, where the candidate the
    /// search kept coming back to given the chance is the better answer.
    fn rate_cmp(&self, other: &Self) -> Ordering {
        (self.visits as u128 * other.opportunities() as u128)
            .cmp(&(other.visits as u128 * self.opportunities() as u128))
    }

    fn mean(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.cumulative_reward / self.visits as f64
        }
    }

    fn mean_cmp(&self, other: &Self) -> Ordering {
        // `Equal` on a NaN mean, which leaves the rate to order the pair rather
        // than letting one poisoned reward win or lose the root.
        self.mean()
            .partial_cmp(&other.mean())
            .unwrap_or(Ordering::Equal)
    }

    /// Whether this candidate outranks `other`. Strict, so a scan that keeps
    /// the incumbent on a tie returns the first maximum and nothing depends on
    /// the order candidates happen to be stored in.
    fn beats(&self, other: &Self) -> bool {
        if self.established() != other.established() {
            return self.established();
        }
        self.mean_cmp(other).then(self.rate_cmp(other)) == Ordering::Greater
    }
}

/// The leading candidate and the index it was offered under: the first maximum
/// under the ranking, among the candidates the search actually visited. `None`
/// if it visited none of them.
///
/// The scan lives here rather than at each root because the answer and the
/// proof that the answer is settled have to be the same argmax. Five
/// hand-written copies agreeing is a coincidence, and the first change to the
/// ranking that one of them did not anticipate is a search stopping early on a
/// move it never chose.
pub(crate) fn leader_of(
    candidates: impl Iterator<Item = (usize, Candidate)>,
) -> Option<(usize, Candidate)> {
    let mut leader: Option<(usize, Candidate)> = None;
    for (index, candidate) in candidates {
        // An unvisited candidate has no estimate at all, and a root that
        // visited nothing has no answer to give.
        if candidate.visits == 0 {
            continue;
        }
        match leader {
            Some((_, best)) if !candidate.beats(&best) => {}
            _ => leader = Some((index, candidate)),
        }
    }
    leader
}

/// Whether `challenger` cannot outrank `leader` however `remaining` more
/// iterations fall.
///
/// **This is the evidence bar and nothing else**, and that is the honest limit
/// of what can be proved about a mean-ranked leader.
///
/// The bound the crate used to state was the conserved-visit one: a candidate
/// `gap` selections behind needs at least `gap` more iterations to draw level,
/// because one iteration buys one candidate one visit. Counts are conserved and
/// a rate is built out of counts, so ranking on a rate kept the whole proof in
/// integer arithmetic. A **mean is not built out of counts** — it moves with
/// the rewards — so that proof does not survive the ranking above, and the only
/// sound replacement would need the rewards reaching a node's accumulator to be
/// confined to a known interval. They are not; see [`crate::early_stop`], which
/// records the check.
///
/// What is left is a claim about tiers, which is pure counting and holds
/// whatever any reward turns out to be: a challenger that cannot reach
/// [`MIN_EVIDENCE`] selections even by taking every remaining iteration stays
/// under the bar for the rest of the search, and a candidate under the bar
/// never outranks one over it. The leader's own tier cannot move — visits only
/// grow — so a leader over the bar stays over it.
///
/// Everything else is refused. An unestablished leader could be overtaken on
/// the mean by anything, and so could an established one by an established
/// rival, so those return `false` and the search spends its budget.
pub(crate) fn out_of_reach(leader: &Candidate, challenger: &Candidate, remaining: u32) -> bool {
    leader.established()
        && u64::from(challenger.visits) + u64::from(remaining) < u64::from(MIN_EVIDENCE)
}

/// Whether [`out_of_reach`] can hold for any challenger at all with `remaining`
/// iterations left.
///
/// The bound is a count against [`MIN_EVIDENCE`], so a challenger holding no
/// selections at all still reaches the bar while more iterations are left than
/// the bar is high — and one that reaches it is one no count settles. The
/// caller may skip the scan entirely there, which is what keeps a proof that
/// can shorten any budget by at most `MIN_EVIDENCE - 1` iterations from costing
/// a pass over every root candidate on every iteration of it.
pub(crate) fn out_of_reach_possible(remaining: u32) -> bool {
    remaining < MIN_EVIDENCE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leader(candidates: &[(u32, u32, f64)]) -> Option<usize> {
        leader_of(
            candidates
                .iter()
                .enumerate()
                .map(|(i, &(visits, availability, reward))| {
                    (i, Candidate::new(visits, availability, reward))
                }),
        )
        .map(|(i, _)| i)
    }

    /// The defect this ranking exists for: the always-legal candidate holds
    /// twice the visits and loses, because the search measured it at 0.5 and
    /// its rival at 0.9.
    #[test]
    fn the_rarely_offered_candidate_wins_on_its_mean() {
        assert_eq!(
            leader(&[(3995, 6000, 1997.5), (2005, 2005, 1804.5)]),
            Some(1)
        );
    }

    /// And the failure that rules out ranking on the rate: the decoy's rate is
    /// a perfect 1.000 off enough selections to clear any bar, and it is worth
    /// 0.75 against the best move's 0.90.
    #[test]
    fn a_saturated_rate_does_not_beat_a_better_mean() {
        assert_eq!(leader(&[(9873, 9999, 8885.7), (97, 97, 72.75)]), Some(0));
        // The same shape one determinization in 250, worth 0.60.
        assert_eq!(leader(&[(9960, 9999, 8964.0), (39, 39, 23.4)]), Some(0));
    }

    /// The Duct case, where the loss was sharper because decoupled selection had
    /// already found the dominant arm.
    #[test]
    fn the_dominant_arm_wins_the_simultaneous_root() {
        assert_eq!(
            leader(&[(45180, 60000, 18072.0), (14820, 14821, 13338.0)]),
            Some(1)
        );
    }

    /// A mean off a handful of rollouts is what selection hands any candidate it
    /// has not visited, so it does not outrank a mean measured over thousands.
    /// This is the whole of the bar's job.
    #[test]
    fn a_lucky_stranger_does_not_win() {
        assert_eq!(leader(&[(45180, 60000, 18072.0), (2, 2, 2.0)]), Some(0));
        // Not even against one that is merely well sampled and much worse.
        assert_eq!(leader(&[(600, 60000, 60.0), (2, 2, 2.0)]), Some(0));
        assert_eq!(
            leader(&[(MIN_EVIDENCE, 6000, 0.0), (MIN_EVIDENCE - 1, 31, 31.0)]),
            Some(0)
        );
        // One selection either side of the bar is the whole difference.
        assert_eq!(
            leader(&[(MIN_EVIDENCE - 1, 6000, 0.0), (MIN_EVIDENCE - 1, 31, 31.0)]),
            Some(1)
        );
    }

    /// Under the bar the mean leads too. Every root wider than its budget is
    /// this case, with one visit on every candidate.
    #[test]
    fn under_the_bar_the_best_measured_candidate_wins() {
        assert_eq!(leader(&[(1, 1, 0.0), (1, 1, 0.5), (1, 1, 1.0)]), Some(2));
        // Ties fall to the rate, and then to the first maximum.
        assert_eq!(leader(&[(1, 4, 1.0), (1, 1, 1.0), (1, 1, 1.0)]), Some(1));
    }

    /// The rate breaks a tie above the bar as well: two candidates measured at
    /// the same value, and the search kept returning to one of them.
    #[test]
    fn the_rate_breaks_a_tie_between_equal_means() {
        assert_eq!(leader(&[(100, 400, 50.0), (100, 200, 50.0)]), Some(1));
    }

    /// The bar is a count, so the remaining iterations bound it: a challenger
    /// that cannot reach [`MIN_EVIDENCE`] stays under it for good.
    #[test]
    fn the_tier_bound_is_what_is_left_of_the_proof() {
        let leader = Candidate::new(4000, 4000, 2000.0);
        let stranger = Candidate::new(1, 1, 1.0);
        assert!(out_of_reach(&leader, &stranger, MIN_EVIDENCE - 2));
        assert!(!out_of_reach(&leader, &stranger, MIN_EVIDENCE - 1));
    }

    /// Nothing else is proved. A challenger already over the bar is ranked on a
    /// mean no count bounds, however far behind it is; and a leader under the
    /// bar proves nothing about anyone.
    #[test]
    fn a_mean_is_never_proved() {
        let leader = Candidate::new(9000, 9000, 8100.0);
        let established = Candidate::new(MIN_EVIDENCE, 9000, 0.0);
        assert!(!out_of_reach(&leader, &established, 1));

        let thin = Candidate::new(MIN_EVIDENCE - 1, MIN_EVIDENCE - 1, 31.0);
        assert!(!out_of_reach(&thin, &Candidate::new(1, 4000, 0.0), 1));
    }
}
