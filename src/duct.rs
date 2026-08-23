//! Selection and credit at simultaneous nodes.
//!
//! Nothing in this module is reachable from a game that never reports
//! [`crate::Status::Simultaneous`]: every entry point takes a `Simul<C>`, and a
//! node only grows one after reporting it. `select` and the sequential half of
//! backup are untouched, which is what keeps sequential codegen — and the
//! `tiny/100k` bench — exactly what it was.
//!
//! The two policies differ in one place only, and it is not the arm statistics:
//! [`SimultaneousPolicy::Duct`] scores arms and plays the argmax, so its
//! sampling probability is 1 and backup needs no correction, while
//! [`SimultaneousPolicy::RegretMatching`] samples from a distribution and
//! therefore has to carry that distribution's probability into backup as an
//! importance weight. Reconstructing the weight at backup instead is not
//! possible: it depends on a floor that has already decayed by then.
//!
//! They also read different [`crate::Config`] fields —
//! [`crate::SimultaneousConfig::duct_exploration`] and
//! [`crate::SimultaneousConfig::regret_matching_exploration`] — because a UCB1 constant on
//! the reward scale and a sampling probability are not the same number. The
//! measurements behind that split are on those two doc comments.

use rand_core::Rng;

use crate::game::{Rewards, SimultaneousPolicy};
use crate::node::{JointKey, Simul};
use crate::rank::{leader_of, Candidate};
use crate::select::ucb_raw;
use crate::util::{below, uniform_01};

/// Fraction of the reward span within which two arm values count as tied.
///
/// A tolerance rather than exact equality, because two arms that are equal in
/// the game are almost never equal after a division and a square root. A strict
/// test would see a tie about never and quietly fall back to first-maximum —
/// which is index order, the exact rule randomized tie-breaking exists to
/// avoid.
///
/// The span is [`crate::Config::max_reward`] minus
/// [`crate::Config::min_reward`], so this is the one place the declared range
/// reaches a `Duct` search — there is no clamp on that path — and a range
/// declared wider than the payoffs really are widens the pool in proportion.
/// The reference cannot instead be the spread of the values actually observed
/// at the node: that makes one arm with a terrible estimate widen the band the
/// good arms are compared inside, which is backwards. It is the caller's
/// declared scale or nothing, and [`crate::Config::max_reward`] says so.
const TIE_TOLERANCE: f64 = 0.01;

/// Slack on the reward-range assertion, as specified in DESIGN.md §6.
///
/// The assertion is there to catch a range that does not describe the payoffs,
/// not to referee the last bit of a float: a game whose own arithmetic reaches
/// a declared bound by a different route than the caller's literal did lands a
/// few ulps outside it and has done nothing wrong.
const RANGE_EPSILON: f64 = 1e-9;

/// The largest exploration floor regret matching will mix in.
///
/// Above one half the floor is most of the distribution, and the strategy being
/// averaged is mostly noise.
const MAX_FLOOR: f64 = 0.5;

/// The smallest exploration floor regret matching will mix in.
///
/// A guard rather than a schedule: with the shipped starting value the decay
/// only reaches it after `(0.4 / 0.005)^4`, about forty million visits to a
/// single node. It is what makes
/// the floor positive for *every* configuration, including a caller who passes
/// zero or a negative number, and backup divides by a sampling probability this
/// bounds below.
const MIN_FLOOR: f64 = 0.005;

/// A raw payoff on the `[0, 1]` scale a regret lives on.
///
/// `span` is `Config::max_reward - Config::min_reward`, which
/// `Config::validate` has already refused to let a search run with at
/// zero or below. The `0.5` this returns for an empty one is therefore a
/// release-build backstop rather than a supported configuration, and it is
/// `0.5` rather than a `0 / 0`: a NaN regret would make its arm's `sigma` NaN,
/// `f64::max` would quietly report the arm's positive part as `0.0` rather than
/// propagating, and the arm would be unselectable for the rest of the search
/// with no symptom other than a worse move.
///
/// The clamp is the part with teeth. Regret matching's `sigma` is invariant
/// under any positive rescaling of every regret, so `span` does not change the
/// strategy at all except by clamping — which means a game paying in `[-1, 1]`
/// that leaves [`crate::Config::min_reward`] and
/// [`crate::Config::max_reward`] at their defaults maps every loss *and* every
/// draw to `0.0`, and regret matching cannot tell a loss from a draw at any
/// budget. Debug builds assert the range in [`credit_marginals`].
#[inline]
pub(crate) fn normalize_reward(reward: f64, min_reward: f64, span: f64) -> f64 {
    if span > 0.0 {
        ((reward - min_reward) / span).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Regret matching's uniform-exploration floor at a node that has been visited
/// `epoch` times: `gamma_0 * t^(-1/4)`, clamped to `[MIN_FLOOR, MAX_FLOOR]`.
///
/// The exponent is the whole design, and `1/2` — which this crate shipped
/// first, and which §4.3 of the design document specified — is exactly the
/// wrong value. Backup divides the observed payoff by the probability the arm
/// was sampled with, so the floor is what bounds the importance weight, at
/// `arms / gamma_t`. For a floor decaying as `t^-a` the regret-matching bound
/// carries `sum_t gamma_t^-2` under a square root, and the average regret it
/// bounds goes as `T^(a - 1/2) / gamma_0`. At `a = 1/2` that exponent is
/// **zero**: the bound stops depending on the budget, and so does the search.
///
/// Measured on biased rock-paper-scissors driven straight through
/// [`select_marginal`] and [`credit_marginals`], 64 seeds times both slots,
/// mean (and worst) exploitability of the extracted strategy at 10k / 50k /
/// 200k / 800k iterations:
///
/// | floor | 10k | 50k | 200k | 800k |
/// |---|---|---|---|---|
/// | `0.1 * t^-1/2` (shipped first) | 0.070 (0.444) | 0.050 (0.305) | 0.038 (0.225) | 0.028 (0.112) |
/// | constant 0.05 | 0.021 (0.113) | 0.010 (0.035) | 0.0082 (0.024) | 0.0077 (0.017) |
/// | `0.4 * t^-1/4` (here) | 0.019 (0.085) | 0.013 (0.071) | 0.0071 (0.026) | 0.0045 (0.019) |
///
/// The uniform strategy is exploitable for 0.083 on that game, so the first row
/// is what a search that has learned very little looks like — and it is what
/// the crate default produced.
///
/// Decaying is still right, but not for the reason the design document gave. It
/// argued that a constant floor cannot get closer than a `gamma * depth`-Nash
/// equilibrium and that accumulating the *pre-floor* `sigma` into `strategy_sum`
/// answers that objection exactly. The first half is true and the second half
/// is not: removing the floor from the average is arithmetic, and it does not
/// touch the fact that each player is learning a best reply to the opponent's
/// *played* mixture, floor included. The fixed point is an equilibrium of the
/// perturbed game. That is why the constant-0.05 row above flattens out around
/// 0.008 while this one keeps falling — and why the answer is a floor that
/// decays slowly enough for the importance weights to stay finite, rather than
/// either a constant or `1 / sqrt(t)`.
///
/// The starting value is not delicate: over 64 seeds every `gamma_0` from 0.35
/// to 0.6 lands inside the seed noise of every other, at every one of those four
/// budgets. 0.4 is picked from that band for sitting clear of `MAX_FLOOR`, so it
/// is genuinely the probability an arm is explored with on a node's first visit
/// rather than a number the cap has already clipped, and for being the top of
/// the range published SM-MCTS studies tune over. The exponent is what matters.
///
/// `epoch` is a per-node visit count, not a global one, so a node the search
/// has only just reached explores hard while the root explores gently. That is
/// right for the bandit at that node and is not a claim about the tree: nothing
/// is proved past one simultaneous ply (`SimultaneousPolicy` says so).
#[inline]
fn exploration_floor(exploration: f64, epoch: u32) -> f64 {
    // `sqrt().sqrt()` is `t^(1/4)` in two instructions, against a `powf` call
    // per participant per iteration.
    let t = (epoch.max(1) as f64).sqrt().sqrt();
    (exploration / t).clamp(MIN_FLOOR, MAX_FLOOR)
}

/// One participant's action for this iteration, as a **slot-relative** arm
/// index paired with the probability the policy gave it.
///
/// `None` when no arm of `slot` is legal this iteration: the joint action does
/// not exist, and the caller must score the descent zero and break, exactly as
/// the degenerate empty-choice node already does.
///
/// `epoch` is the node's visit count *before* this iteration's record. It does
/// double duty: `arm_stats[a].avail_epoch == epoch` is the legality test, so
/// there is no availability buffer, and it is regret matching's `t`.
///
/// `exploration` is whichever [`crate::Config`] field this policy reads —
/// `duct_exploration` under [`SimultaneousPolicy::Duct`],
/// `regret_matching_exploration` under
/// [`SimultaneousPolicy::RegretMatching`] — resolved by the caller, which is
/// the one place that knows `G::SIMULTANEOUS_POLICY` as a constant.
///
/// The probability is returned rather than recomputed during backup because it
/// cannot be recovered from the tree — it depends on the whole distribution at
/// the moment of selection and on an exploration floor that has decayed by the
/// time backup runs. It is `1.0` under [`SimultaneousPolicy::Duct`], which does
/// no importance weighting.
pub(crate) fn select_marginal<C, R: Rng + ?Sized>(
    simul: &mut Simul<C>,
    slot: usize,
    epoch: u32,
    policy: SimultaneousPolicy,
    exploration: f64,
    span: f64,
    rng: &mut R,
) -> Option<(u32, f64)> {
    match policy {
        SimultaneousPolicy::Duct => select_duct(simul, slot, epoch, exploration, span, rng),
        SimultaneousPolicy::RegretMatching => {
            select_regret_matching(simul, slot, epoch, exploration, rng)
        }
    }
}

/// Decoupled UCB1 over one participant's arms, on the availability count.
///
/// Both random choices here are reservoir draws — the `k`-th candidate replaces
/// the incumbent with probability `1/k`, which leaves every candidate equally
/// likely without a buffer of indices.
///
/// Both are also mandatory. Two decoupled bandits that break ties by index pair
/// player 0's arm `i` with player 1's arm `i`, so the search opens the diagonal
/// of the joint matrix first and starves the off-diagonal cells that decide a
/// mixed equilibrium; measured exploitability on biased rock-paper-scissors is
/// 0.5-0.8 that way and 0.01-0.05 with these draws. The sequential `select`
/// keeps first-strict-maximum, whose fixed point does not depend on the tie
/// rule.
///
/// The tie pool costs a **second pass**, and that is not an oversight. A single
/// pass can only compare each arm against the maximum *so far*, and a maximum
/// that only ever rises admits an arm within `tol` of an early leader into a
/// pool it is nowhere near by the end: twenty arms spaced a hair under `tol`
/// apart all land in one pool and the draw over them is uniform, so the true
/// argmax is picked one time in twenty and the worst arm just as often. The
/// error that allows is `(arms - 1) * TIE_TOLERANCE * span`, unbounded in the
/// arm count. The maximum has to be known before the pool is built.
fn select_duct<C, R: Rng + ?Sized>(
    simul: &Simul<C>,
    slot: usize,
    epoch: u32,
    exploration: f64,
    span: f64,
    rng: &mut R,
) -> Option<(u32, f64)> {
    let range = simul.slot_range(slot);
    let start = range.start;
    let tol = TIE_TOLERANCE * span;

    // `ucb_raw`, not a second copy of the formula. `select.rs` keeps it as one
    // function precisely so two callers can never drift apart on it, and an arm
    // is the same bandit as a child scored against its own availability. The
    // progressive-bias term is passed off rather than omitted: `heuristic_bias`
    // describes a state, so at a simultaneous node it would describe a joint
    // successor while selection here scores one player's arms. A zero weight
    // makes the term identically zero and the constant folds away.
    let score = |a: usize| -> f64 {
        let arm = &simul.arm_stats[a];
        ucb_raw(
            arm.cumulative_reward,
            arm.visits as f64,
            arm.ln_availability,
            exploration,
            0.0,
            0.0,
        )
    };

    let mut unvisited = 0u64;
    let mut unvisited_pick = None;
    let mut best_value = f64::NEG_INFINITY;

    for a in range.clone() {
        let arm = &simul.arm_stats[a];
        if arm.avail_epoch != epoch {
            continue;
        }
        if arm.visits == 0 {
            unvisited += 1;
            if below(rng, unvisited) == 0 {
                unvisited_pick = Some(a);
            }
            continue;
        }
        let value = score(a);
        if value > best_value {
            best_value = value;
        }
    }

    // An arm nothing is known about outranks every arm something is known
    // about, so the tie pool is never built when one exists.
    if let Some(pick) = unvisited_pick {
        return Some(((pick - start) as u32, 1.0));
    }

    let mut best = None;
    let mut ties = 0u64;
    for a in range {
        let arm = &simul.arm_stats[a];
        if arm.avail_epoch != epoch || arm.visits == 0 {
            continue;
        }
        if score(a) >= best_value - tol {
            ties += 1;
            if below(rng, ties) == 0 {
                best = Some(a);
            }
        }
    }

    let pick = best?;
    Some(((pick - start) as u32, 1.0))
}

/// Regret matching over one participant's arms, with a decaying exploration
/// floor, sampled inline.
///
/// Two passes and a scalar accumulator: the first sums the positive-regret mass
/// over the legal arms, the second walks the inverse CDF against a single draw.
/// No probability vector is materialized, which is why the search's scratch
/// buffers include no strategy buffer.
///
/// `strategy_sum` accumulates `sigma`, the distribution *before* the floor is
/// mixed in, and never `mu`. Reconstructing it afterwards by subtracting
/// `gamma * t / n` is only correct for a constant floor over a constant legal
/// set, and here the floor decays and determinization varies the legal set — so
/// the published subtraction is wrong in precisely the setting this crate runs
/// in. It is worth being clear about what this does and does not buy: it takes
/// the floor out of the *average*, which is real, and it does nothing about the
/// floor being in the opponent's *played* mixture, which is what leaves a
/// constant floor with a residual exploitability. That is
/// [`exploration_floor`]'s problem, and why the floor decays.
fn select_regret_matching<C, R: Rng + ?Sized>(
    simul: &mut Simul<C>,
    slot: usize,
    epoch: u32,
    exploration: f64,
    rng: &mut R,
) -> Option<(u32, f64)> {
    debug_assert_eq!(
        simul.arm_policy.len(),
        simul.arm_stats.len(),
        "mcts: regret matching needs policy state for every arm; the arms were built \
         for a different SimultaneousPolicy than the one selecting over them, and the \
         strategy this node has been accumulating belongs to neither"
    );

    let range = simul.slot_range(slot);
    let start = range.start;

    let mut n = 0u32;
    let mut sum_r = 0.0;
    for a in range.clone() {
        if simul.arm_stats[a].avail_epoch != epoch {
            continue;
        }
        n += 1;
        sum_r += simul.arm_policy[a].regret.max(0.0);
    }
    if n == 0 {
        return None;
    }

    let gamma = exploration_floor(exploration, epoch);
    let uniform = 1.0 / n as f64;
    let target = uniform_01(rng);

    let mut acc = 0.0;
    let mut picked = None;
    let mut last = None;
    for a in range {
        if simul.arm_stats[a].avail_epoch != epoch {
            continue;
        }
        let sigma = if sum_r > 0.0 {
            simul.arm_policy[a].regret.max(0.0) / sum_r
        } else {
            uniform
        };
        simul.arm_policy[a].strategy_sum += sigma;
        let mu = (1.0 - gamma) * sigma + gamma * uniform;
        acc += mu;
        if mu > 0.0 {
            if picked.is_none() && acc >= target {
                picked = Some((a, mu));
            }
            last = Some((a, mu));
        }
    }

    // The accumulator can land a ULP short of a `target` near 1, and an arm with
    // no mass must never come back as the pick: backup divides by this
    // probability, so a zero would put an infinity into a regret and every
    // comparison after it would be false.
    let (pick, prob) = picked.or(last)?;
    Some(((pick - start) as u32, prob))
}

/// Credit one simultaneous level of the descent to its participants' arms.
///
/// `key` is the joint edge that was taken, `probs` holds one sampling
/// probability per slot in ascending player order, and `epoch` is the node's
/// visit count as the descent stamped it.
///
/// This must run **before** `Node::record` bumps that visit count. The legality
/// test is `avail_epoch == node.visits`, so after the record every arm reads as
/// illegal, the regret subtraction below silently covers nothing, and the
/// sleeping-bandit correction that keeps a rarely-legal action playable is gone
/// with no symptom other than a worse move.
///
/// Each player's own reward goes into their own arm — the max^n rule
/// specialized — which is exactly what makes the `k` bandits independent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn credit_marginals<C, W: Rewards>(
    simul: &mut Simul<C>,
    key: JointKey,
    epoch: u32,
    probs: &[f32],
    rewards: &W,
    policy: SimultaneousPolicy,
    min_reward: f64,
    max_reward: f64,
) {
    let span = max_reward - min_reward;
    let arity = simul.arity as usize;
    let key_bits = simul.key_bits;
    debug_assert_eq!(
        probs.len(),
        arity,
        "mcts: one sampling probability per participant is recorded per simultaneous \
         level; a mismatch means the descent's frames and this node's arity disagree \
         and one player's weight is being applied to another's regret"
    );

    for (slot, &prob) in probs.iter().enumerate().take(arity) {
        let player = simul
            .players
            .player_at(slot)
            .expect("mcts: every slot of a simultaneous node has a participant");
        let at = simul.starts[slot] as usize + key.arm(slot, key_bits) as usize;
        let reward = rewards.reward(player);

        // Both policies, and no exemption for an empty span. The range is what
        // regret matching clamps to and what scales `Duct`'s tie tolerance, so
        // a payoff outside it is a live defect under either — and the empty
        // range this used to wave through is the configuration that makes
        // regret matching worse than random, which `Config::validate` now
        // refuses outright.
        //
        // Against `max_reward` itself, with slack, rather than against a
        // `min_reward + span` reconstruction of it: that sum is
        // `-3.9000000000000004` for a declared `[-10, -3.9]`, so a game paying
        // exactly its own declared maximum would be accused of the defect this
        // assertion exists to report.
        debug_assert!(
            reward >= min_reward - RANGE_EPSILON && reward <= max_reward + RANGE_EPSILON,
            "mcts: player {player} was paid {reward} at a simultaneous node, outside \
             the declared [{min_reward}, {max_reward}] reward range. Regret matching \
             clamps payoffs into [0, 1] before touching a regret, so an out-of-range \
             reward is indistinguishable from one at the boundary — a game paying in \
             [-1, 1] against the default [0, 1] range cannot tell a loss from a draw \
             — and Duct measures its tie tolerance against the width of the range, so \
             one declared wider than the payoffs draws uniformly between arms it \
             should be ranking."
        );

        if let SimultaneousPolicy::RegretMatching = policy {
            let u_hat = normalize_reward(reward, min_reward, span);
            let mu = prob as f64;
            debug_assert!(
                mu > 0.0,
                "mcts: an arm was played with probability {mu}; the importance weight \
                 below divides by it"
            );

            // Only arms that were legal this iteration accumulate regret. An
            // illegal arm could not have been played, so there is nothing to
            // regret; leaving the subtraction unrestricted drives a rarely-legal
            // arm's regret toward negative infinity at rate
            // `(1 - f) * E[u] * t` in its legality rate `f`, and it is never
            // selected again.
            //
            // The filter is necessary and it is not sufficient: an arm the
            // strategy has already driven negative is reachable only through
            // the exploration floor, so a floor that decays too fast starves it
            // anyway. Measured on an arm legal half the time and strictly best
            // whenever it is, at 3000 visits over 40 seeds: 2 seeds leave it
            // with zero visits under the `0.1 / sqrt(t)` floor this crate
            // shipped first and 10 do at `0.05 / sqrt(t)`, against 0 under
            // `exploration_floor`, which also lifts its share of the iterations
            // that offered it from 0.67 to 0.94.
            for b in simul.slot_range(slot) {
                if simul.arm_stats[b].avail_epoch == epoch {
                    simul.arm_policy[b].regret -= u_hat;
                }
            }
            simul.arm_policy[at].regret += u_hat / mu;
        }

        let arm = &mut simul.arm_stats[at];
        arm.cumulative_reward += reward;
        arm.visits += 1;
    }
}

/// Write one participant's mixed strategy over `slot`'s arms into `out`,
/// restricted to the arms legal in the real position and normalized over them.
///
/// `legal` is parallel to the slot's arms. `out` is cleared first and comes
/// back the same length, with a zero at every illegal arm; it sums to 1 unless
/// nothing is legal, in which case it is all zeros and the caller has no move
/// to extract here.
///
/// Under [`SimultaneousPolicy::RegretMatching`] an arm's weight is its
/// `strategy_sum` divided by its `availability`. That division is not cosmetic:
/// `strategy_sum` only accumulates on iterations where the arm was legal, so
/// two equally good arms legal in 90% and 20% of determinizations end up with
/// mass in roughly a 4.5:1 ratio, and dividing turns each accumulator into a
/// mean probability *conditional on being legal*. It is sound only because
/// re-rooting keeps arm availability alongside `strategy_sum`: any future change
/// that resets one without the other silently corrupts the extracted
/// equilibrium.
///
/// It is a correction and not a cure, and the residual is worth stating. Two
/// arms conditioned on *different* legality events are still not two entries of
/// one distribution: an arm legal every iteration absorbs mass on the
/// iterations where its rival is absent, and no per-arm accumulator can tell
/// that mass from mass it earned. In the sharpest case — arm A legal always,
/// arm B legal half the time and strictly better whenever it is legal — the
/// exact answer at a position where both are legal is all of the mass on B, and
/// this extraction reports 0.33 on A (measured over 16 seeds at 200k
/// iterations, and unchanged by the floor schedule). Getting
/// that exactly right needs a statistic per legal *set* rather than per arm.
/// The design document is candid that the sleeping-bandit corrections are
/// principled engineering rather than theorems; this is where that bites.
///
/// Under [`SimultaneousPolicy::Duct`] the result is one-hot at [`leading_arm`],
/// deliberately: decoupled UCB1 converges to a pure policy, and its visit
/// distribution is the object an opponent exploits, so it is not handed back
/// dressed as a mixed strategy. `Duct` ranks by the mean reward there rather
/// than by the raw count, for the same reason this branch divides
/// `strategy_sum` by availability: a count is a measurement of how often the
/// arm was legal as much as of how good it is.
pub(crate) fn root_strategy_into<C>(
    simul: &Simul<C>,
    slot: usize,
    legal: &[bool],
    policy: SimultaneousPolicy,
    out: &mut Vec<f64>,
) {
    out.clear();
    let arms = simul.slot_len(slot);
    debug_assert_eq!(
        legal.len(),
        arms,
        "mcts: the legality mask and the participant's arms are out of step"
    );
    if arms == 0 {
        return;
    }

    let one_hot = duct_target(simul, slot, legal, policy);
    let mut total = 0.0;
    for (arm, &is_legal) in legal.iter().enumerate().take(arms) {
        let weight = if is_legal {
            root_weight(simul, slot, arm, policy, one_hot)
        } else {
            0.0
        };
        out.push(weight);
        total += weight;
    }

    if total > 0.0 {
        for weight in out.iter_mut() {
            *weight /= total;
        }
        return;
    }

    let count = legal
        .iter()
        .take(arms)
        .filter(|&&is_legal| is_legal)
        .count();
    if count == 0 {
        return;
    }
    let share = 1.0 / count as f64;
    for (weight, &is_legal) in out.iter_mut().zip(legal) {
        *weight = if is_legal { share } else { 0.0 };
    }
}

/// Draw one of `slot`'s legal arms from the same strategy
/// [`root_strategy_into`] reports, as a **slot-relative** index.
///
/// The same two-pass inverse CDF selection uses, for the same reason: the root
/// of a wide simultaneous node has no reason to allocate a probability vector
/// to make one draw. `None` when no arm is legal.
pub(crate) fn sample_root_arm<C, R: Rng + ?Sized>(
    simul: &Simul<C>,
    slot: usize,
    legal: &[bool],
    policy: SimultaneousPolicy,
    rng: &mut R,
) -> Option<usize> {
    let arms = simul.slot_len(slot);
    debug_assert_eq!(
        legal.len(),
        arms,
        "mcts: the legality mask and the participant's arms are out of step"
    );

    let one_hot = duct_target(simul, slot, legal, policy);
    let legal_arms = || {
        legal
            .iter()
            .enumerate()
            .take(arms)
            .filter(|(_, &is_legal)| is_legal)
            .map(|(arm, _)| arm)
    };

    let mut total = 0.0;
    let mut count = 0u64;
    for arm in legal_arms() {
        count += 1;
        total += root_weight(simul, slot, arm, policy, one_hot);
    }
    if count == 0 {
        return None;
    }
    if total <= 0.0 {
        return legal_arms().nth(below(rng, count) as usize);
    }

    let target = uniform_01(rng) * total;
    let mut acc = 0.0;
    let mut last = None;
    for arm in legal_arms() {
        let weight = root_weight(simul, slot, arm, policy, one_hot);
        if weight <= 0.0 {
            continue;
        }
        acc += weight;
        if acc >= target {
            return Some(arm);
        }
        last = Some(arm);
    }
    last
}

/// The legal arm of `slot` this policy puts the most weight on, as a
/// **slot-relative** index, first maximum. `None` if the search has nothing to
/// say about any legal arm yet.
///
/// This is what [`crate::RootPolicy::MostVisited`] returns, and it is
/// deliberately *not* the most-visited arm under either policy. Under
/// [`SimultaneousPolicy::RegretMatching`] the weight is the conditional
/// strategy mass [`root_strategy_into`] extracts, whose corrections do not stop
/// applying because the caller asked for a deterministic answer. Under
/// [`SimultaneousPolicy::Duct`] the weight is one-hot at [`leading_arm`], which
/// ranks by mean reward for the same reason: an argmax over raw visits hands
/// back an arm that is strictly dominated wherever it is legal, purely because
/// it was legal more often than its rival — and under `Duct` that is the
/// sharper failure of the two, since decoupled UCB1's *selection* is
/// availability-corrected and had already found the dominant arm.
///
/// Never an argmax over joint successors: this player's action appears in one
/// joint child per opponent action, so the best *pair* is not the best action —
/// the optimistic bug that makes a decoupled agent assume its opponent plays
/// along.
///
/// Where no legal arm carries any weight the answer is [`leading_arm`], for the
/// same reason [`root_strategy_into`] and [`sample_root_arm`] fall back to
/// uniform over the legal arms there: the caller asked which action to play and
/// a distribution with no mass on it is not an answer. Zero mass is a state the
/// search reaches and stays in rather than a startup transient — an arm
/// dominated wherever it is legal has `sigma = 0` on every iteration, so its
/// `strategy_sum` never leaves zero while the exploration floor keeps handing it
/// visits — and falling through to the caller's own uniform draw there returned
/// a move whose reported visits and mean reward were the hardcoded zeroes of
/// that fallback rather than the statistics the tree holds for it. The
/// fallback is a ranking rather than a draw because
/// [`crate::RootPolicy::MostVisited`] is documented deterministic; the uniform
/// distribution its siblings report leaves every legal arm tied, and this
/// breaks that tie by the crate's root ranking instead of by index order.
pub(crate) fn best_arm<C>(
    simul: &Simul<C>,
    slot: usize,
    legal: &[bool],
    policy: SimultaneousPolicy,
) -> Option<usize> {
    let arms = simul.slot_len(slot);
    let one_hot = duct_target(simul, slot, legal, policy);
    let mut best = None;
    let mut best_weight = 0.0;
    for (arm, &is_legal) in legal.iter().enumerate().take(arms) {
        if !is_legal {
            continue;
        }
        let weight = root_weight(simul, slot, arm, policy, one_hot);
        if weight > best_weight {
            best_weight = weight;
            best = Some(arm);
        }
    }
    // Under `Duct` the weight is already one-hot at `leading_arm`, so this only
    // ever answers where nothing legal was selected at all — and then it is
    // `None` either way.
    best.or_else(|| leading_arm(simul, slot, legal))
}

/// The highest-ranked legal arm of `slot`, first maximum, as a **slot-relative**
/// index, under the crate's one root ranking ([`crate::rank`]).
///
/// The right answer only where the equilibrium is pure, which is why
/// [`best_arm`] and not this is what a root move is read off. `None` if nothing
/// legal has been selected yet.
fn leading_arm<C>(simul: &Simul<C>, slot: usize, legal: &[bool]) -> Option<usize> {
    let start = simul.starts[slot] as usize;
    leader_of(
        legal
            .iter()
            .enumerate()
            .filter(|(_, &is_legal)| is_legal)
            .map(|(arm, _)| {
                let stats = &simul.arm_stats[start + arm];
                (
                    arm,
                    Candidate::new(stats.visits, stats.availability, stats.cumulative_reward),
                )
            }),
    )
    .map(|(arm, _)| arm)
}

/// The arm `Duct` puts all of its root mass on, or `None` under a policy that
/// mixes.
#[inline]
fn duct_target<C>(
    simul: &Simul<C>,
    slot: usize,
    legal: &[bool],
    policy: SimultaneousPolicy,
) -> Option<usize> {
    match policy {
        SimultaneousPolicy::Duct => leading_arm(simul, slot, legal),
        SimultaneousPolicy::RegretMatching => None,
    }
}

/// One arm's unnormalized root weight, so that the vector form, the sampled
/// form and the deterministic form cannot drift apart.
#[inline]
fn root_weight<C>(
    simul: &Simul<C>,
    slot: usize,
    arm: usize,
    policy: SimultaneousPolicy,
    one_hot: Option<usize>,
) -> f64 {
    match policy {
        SimultaneousPolicy::RegretMatching => {
            let at = simul.starts[slot] as usize + arm;
            simul.arm_policy[at].strategy_sum.max(0.0)
                / simul.arm_stats[at].availability.max(1) as f64
        }
        SimultaneousPolicy::Duct => {
            if one_hot == Some(arm) {
                1.0
            } else {
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PlayerSet;
    use crate::node::Node;
    use crate::search::Config;
    use rand_core::SeedableRng;
    use wyrand::WyRand;

    /// Biased rock-paper-scissors (Shafiei/Sturtevant/Schaeffer) rescaled to
    /// `[0, 1]`, indexed `[player 0][player 1]`. Zero-sum, so player 1 is paid
    /// `1 - p0`, and the unique Nash equilibrium is `(1/16, 10/16, 5/16)` for
    /// both players with a value of 0.5.
    const BIASED_RPS: [[f64; 3]; 3] = [[0.50, 0.25, 1.00], [0.75, 0.50, 0.45], [0.00, 0.55, 0.50]];
    const EQUILIBRIUM: [f64; 3] = [0.0625, 0.625, 0.3125];
    const RM: SimultaneousPolicy = SimultaneousPolicy::RegretMatching;
    const DUCT: SimultaneousPolicy = SimultaneousPolicy::Duct;

    /// The shipped defaults, so that no test here can pass at a value a user
    /// would not get. This is the whole point: the floor schedule these tests
    /// exercise is the one `Config::default()` hands out.
    fn floor() -> f64 {
        Config::default().simultaneous.regret_matching_exploration
    }

    fn ucb() -> f64 {
        Config::default().simultaneous.duct_exploration
    }
    /// Upper tail of chi-square at p = 0.001 with three degrees of freedom.
    const CHI2_3DF: f64 = 16.266;

    fn simul_node(players: PlayerSet) -> Node<u8> {
        let mut node = Node::new_root(0);
        node.ensure_simul(players);
        node
    }

    fn chi_square(observed: &[u64], expected: &[f64]) -> f64 {
        observed
            .iter()
            .zip(expected)
            .map(|(&count, &want)| {
                let diff = count as f64 - want;
                diff * diff / want
            })
            .sum()
    }

    /// One simultaneous node playing biased RPS against itself, with no tree,
    /// no game and no rollout: `select_marginal` picks both players' actions,
    /// the payoff matrix answers, and `credit_marginals` closes the loop.
    fn play_biased_rps(iterations: u32, exploration: f64, seed: u64) -> Node<u8> {
        let mut node = simul_node(PlayerSet::first_n(2));
        let mut rng = WyRand::seed_from_u64(seed);
        let actions = [0u8, 1, 2];
        for _ in 0..iterations {
            let epoch = node.visits();
            node.expand_marginals(0, &actions, true, usize::MAX);
            node.expand_marginals(1, &actions, true, usize::MAX);
            let simul = node.simul_mut().expect("just created");
            let (a0, mu0) = select_marginal(simul, 0, epoch, RM, exploration, 1.0, &mut rng)
                .expect("every action is legal");
            let (a1, mu1) = select_marginal(simul, 1, epoch, RM, exploration, 1.0, &mut rng)
                .expect("every action is legal");
            let payoff = BIASED_RPS[a0 as usize][a1 as usize];
            let key = JointKey::pack(&[a0, a1], simul.key_bits);
            credit_marginals(
                simul,
                key,
                epoch,
                &[mu0 as f32, mu1 as f32],
                &[payoff, 1.0 - payoff],
                RM,
                0.0,
                1.0,
            );
            node.record(0.0);
        }
        node
    }

    fn l_infinity(strategy: &[f64], want: &[f64]) -> f64 {
        strategy
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// How much a best-responding opponent beats the game's value of 0.5 by,
    /// against `strategy`. The game is symmetric and zero-sum, so one row of
    /// arithmetic covers either player.
    fn exploitability(strategy: &[f64]) -> f64 {
        let against = |reply: usize| -> f64 {
            (0..3)
                .map(|action| strategy[action] * (1.0 - BIASED_RPS[action][reply]))
                .sum()
        };
        (0..3).map(against).fold(f64::NEG_INFINITY, f64::max) - 0.5
    }

    /// Both players' time-averaged strategies approach the known equilibrium of
    /// biased rock-paper-scissors, with `select_marginal` and
    /// `credit_marginals` driven directly against the payoff matrix — no tree,
    /// no game, no rollout.
    ///
    /// The thresholds are roughly twice the worst case measured over sixteen
    /// seeds, not the error of this one: bandit feedback makes the regret
    /// estimator heavy-tailed (the importance weight is `1 / mu`, and `mu` is
    /// bounded below by the exploration floor), so a single run's L-infinity
    /// error is noisy even where the strategy is already hard to exploit. At
    /// this budget and the shipped floor the worst of sixteen seeds times both
    /// slots is an exploitability of 0.020 and an L-infinity of 0.060; this
    /// seed's two slots come in at 0.004 / 0.008 and 0.006 / 0.041.
    /// Exploitability is the quantity that actually converges, and it is what a
    /// regression here would move first.
    #[test]
    fn regret_matching_learns_the_biased_rps_equilibrium() {
        let node = play_biased_rps(500_000, floor(), 0xB1A5);
        let simul = node.simul().expect("simultaneous");
        let legal = [true; 3];
        let mut strategy = Vec::new();
        for slot in 0..2 {
            root_strategy_into(simul, slot, &legal, RM, &mut strategy);
            let exploited = exploitability(&strategy);
            assert!(
                exploited < 0.05,
                "slot {slot} plays {strategy:?}, exploitable for {exploited}"
            );
            let error = l_infinity(&strategy, &EQUILIBRIUM);
            assert!(
                error < 0.15,
                "slot {slot} plays {strategy:?}, {error} away from {EQUILIBRIUM:?}"
            );
            assert!(
                strategy[1] > strategy[2] && strategy[2] > strategy[0],
                "slot {slot} plays {strategy:?}, which does not rank the actions the \
                 equilibrium {EQUILIBRIUM:?} does"
            );
        }
    }

    /// The property the first cut of this feature silently did not have, and the
    /// reason `exploration_floor` is not `gamma_0 / sqrt(t)`: spending ten times
    /// the budget buys a materially better strategy.
    ///
    /// A `1 / sqrt(t)` floor makes the average-regret bound independent of the
    /// budget, and the search behaves accordingly — measured over these same six
    /// seeds, `0.1 / sqrt(t)` goes from a mean exploitability of 0.061 at 20k to
    /// 0.038 at 200k, a ratio of 0.61, against 0.083 for playing uniformly at
    /// random. The shipped schedule goes from 0.015 to 0.004, a ratio of 0.27.
    /// The ratio alone is not enough, and assuming it was is how the first cut of
    /// this test failed to test its own subject. That reasoning was measured at
    /// `gamma_0 = 0.1`, and it does not survive the move to the shipped 0.4: at
    /// that constant the reverted schedule goes from 0.0402 to 0.0142, a ratio of
    /// 0.35, which clears a 0.5 bar comfortably while being 3.5x worse in
    /// absolute terms. So the level is asserted too. Both are needed — the ratio
    /// catches a rule that is uniformly mediocre, the level catches a rule that
    /// improves at a respectable rate from a bad starting point.
    #[test]
    fn the_default_floor_keeps_improving_with_budget() {
        const SEEDS: u64 = 6;
        let mut early = 0.0;
        let mut late = 0.0;
        for i in 0..SEEDS {
            let seed = 0xF100u64.wrapping_add(i.wrapping_mul(0x9E3779B97F4A7C15));
            let (short, long) = play_and_snapshot(20_000, 200_000, floor(), seed);
            early += short / SEEDS as f64;
            late += long / SEEDS as f64;
        }
        assert!(
            late < early * 0.5,
            "mean exploitability went from {early} at 20k iterations to {late} at 200k; \
             a floor that decays too fast makes the regret bound budget-independent and \
             this ratio is what catches it"
        );
        // The shipped schedule reaches 0.0040 here and `gamma_0 / sqrt(t)` reaches
        // 0.0142, so this separates them by better than 2x in both directions.
        assert!(
            late < 0.008,
            "mean exploitability at 200k iterations is {late}; the shipped floor reaches \
             0.0040 and a schedule that decays too fast reaches 0.0142, so a rate of \
             improvement alone does not show the strategy actually got hard to exploit"
        );
    }

    /// Mean exploitability over both slots at two budgets from one run, so that
    /// the second measurement is the same search carried further rather than a
    /// different one.
    fn play_and_snapshot(short: u32, long: u32, exploration: f64, seed: u64) -> (f64, f64) {
        let mut node = simul_node(PlayerSet::first_n(2));
        let mut rng = WyRand::seed_from_u64(seed);
        let actions = [0u8, 1, 2];
        let mut strategy = Vec::new();
        let mut early = 0.0;
        for i in 0..long {
            let epoch = node.visits();
            node.expand_marginals(0, &actions, true, usize::MAX);
            node.expand_marginals(1, &actions, true, usize::MAX);
            let simul = node.simul_mut().expect("just created");
            let (a0, mu0) =
                select_marginal(simul, 0, epoch, RM, exploration, 1.0, &mut rng).unwrap();
            let (a1, mu1) =
                select_marginal(simul, 1, epoch, RM, exploration, 1.0, &mut rng).unwrap();
            let payoff = BIASED_RPS[a0 as usize][a1 as usize];
            let key = JointKey::pack(&[a0, a1], simul.key_bits);
            credit_marginals(
                simul,
                key,
                epoch,
                &[mu0 as f32, mu1 as f32],
                &[payoff, 1.0 - payoff],
                RM,
                0.0,
                1.0,
            );
            node.record(0.0);
            if i + 1 == short {
                early = mean_exploitability(&node, &mut strategy);
            }
        }
        (early, mean_exploitability(&node, &mut strategy))
    }

    fn mean_exploitability(node: &Node<u8>, strategy: &mut Vec<f64>) -> f64 {
        let simul = node.simul().expect("simultaneous");
        (0..2)
            .map(|slot| {
                root_strategy_into(simul, slot, &[true; 3], RM, strategy);
                exploitability(strategy)
            })
            .sum::<f64>()
            / 2.0
    }

    /// The exploration floor is positive for every configuration, including the
    /// ones a caller should not have written. Backup divides by a sampling
    /// probability this bounds below, so a zero here is an infinity in a regret.
    #[test]
    fn the_exploration_floor_is_always_a_usable_probability() {
        for exploration in [-1.0, 0.0, 1e-300, 0.4, 5.0, f64::INFINITY] {
            for epoch in [0u32, 1, 1_000, u32::MAX] {
                let gamma = exploration_floor(exploration, epoch);
                assert!(
                    (MIN_FLOOR..=MAX_FLOOR).contains(&gamma),
                    "exploration {exploration} at epoch {epoch} gives a floor of {gamma}"
                );
            }
        }
        // The schedule itself, at the shipped starting value.
        assert!((exploration_floor(0.4, 1) - 0.4).abs() < 1e-12);
        assert!((exploration_floor(0.4, 16) - 0.2).abs() < 1e-12);
        assert!((exploration_floor(0.4, 10_000) - 0.04).abs() < 1e-12);
    }

    /// A tie pool built against a running maximum is not a tie pool.
    ///
    /// Twenty arms whose values ascend in steps just under `TIE_TOLERANCE`: the
    /// first arm ties the second, the second the third, and a single pass that
    /// only ever raises its incumbent maximum admits all twenty, so the draw
    /// over them is uniform and the true argmax — a full `19 * tol` above the
    /// worst arm — is picked one time in twenty.
    #[test]
    fn duct_ties_are_measured_against_the_final_maximum() {
        const ARMS: usize = 20;
        let choices: Vec<u8> = (0..ARMS as u8).collect();
        let mut node = simul_node(PlayerSet::first_n(1));
        node.expand_marginals(0, &choices, false, usize::MAX);
        let simul = node.simul_mut().expect("simultaneous");
        for arm in 0..ARMS {
            let stats = &mut simul.arm_stats[arm];
            stats.visits = 100;
            stats.availability = 100;
            stats.ln_availability = 0.0;
            // Mean rewards 0.500, 0.5045, 0.509, ... — steps of 0.9 * tol, so
            // every adjacent pair ties and the ends are nowhere near.
            stats.cumulative_reward = (0.5 + arm as f64 * 0.9 * TIE_TOLERANCE) * 100.0;
        }

        let draws = 200_000;
        let mut counts = [0u64; ARMS];
        let mut rng = WyRand::seed_from_u64(0x71E5);
        for _ in 0..draws {
            let (arm, _) = select_marginal(simul, 0, 0, DUCT, ucb(), 1.0, &mut rng).unwrap();
            counts[arm as usize] += 1;
        }

        // Only the top two arms are within `tol` of the maximum, so each should
        // take about half the draws and nothing else should take any.
        assert_eq!(
            counts[..ARMS - 2].iter().sum::<u64>(),
            0,
            "an arm more than a tolerance below the best was drawn: {counts:?}"
        );
        for arm in ARMS - 2..ARMS {
            let share = counts[arm] as f64 / draws as f64;
            assert!(
                (share - 0.5).abs() < 0.01,
                "arm {arm} took {share} of the draws, not about half: {counts:?}"
            );
        }
    }

    /// `RootPolicy::MostVisited` must not be an argmax over raw visits, under
    /// either policy.
    ///
    /// Arm 0 was legal on every iteration and played on a fifth of them; arm 1
    /// was legal on a fifth of the iterations and played on nearly all of those,
    /// and paid 0.9 against arm 0's 0.4 wherever it was dealt. Their visit
    /// counts are almost equal, so raw visits answer arm 0 — and arm 1 is the
    /// move on both readings: regret matching's conditional strategy mass, and
    /// the mean reward `Duct` ranks by.
    ///
    /// Arm 2 is the other half of the reading. `select_duct` takes an unvisited
    /// legal arm on sight, so an arm dealt three times holds a perfect mean
    /// having proved nothing: under `Duct` it must not outrank either arm the
    /// search actually measured. Regret matching ranks the same arm by the
    /// strategy mass it accumulated over those three iterations and answers it,
    /// which is the per-arm conditional's known limit (DESIGN.md §4.8) and not
    /// something this ranking reaches.
    #[test]
    fn the_deterministic_root_move_is_not_the_visit_argmax() {
        let mut node = simul_node(PlayerSet::first_n(1));
        node.expand_marginals(0, &[0u8, 1, 2], true, usize::MAX);
        let simul = node.simul_mut().expect("simultaneous");
        simul.arm_policy[0].strategy_sum = 200.0;
        simul.arm_stats[0].availability = 1_000;
        simul.arm_stats[0].visits = 200;
        simul.arm_stats[0].cumulative_reward = 80.0;
        simul.arm_policy[1].strategy_sum = 190.0;
        simul.arm_stats[1].availability = 200;
        simul.arm_stats[1].visits = 190;
        simul.arm_stats[1].cumulative_reward = 171.0;
        simul.arm_policy[2].strategy_sum = 3.0;
        simul.arm_stats[2].availability = 3;
        simul.arm_stats[2].visits = 3;
        simul.arm_stats[2].cumulative_reward = 3.0;

        let all = [true, true, true];
        let measured = [true, true, false];
        assert_eq!(raw_visit_argmax(simul, 0, &all), Some(0));
        assert_eq!(leading_arm(simul, 0, &all), Some(1));
        assert_eq!(best_arm(simul, 0, &all, DUCT), Some(1));
        assert_eq!(best_arm(simul, 0, &measured, RM), Some(1));
        // Restricted to the arm that is actually legal here, either answers it.
        assert_eq!(best_arm(simul, 0, &[true, false, false], RM), Some(0));
        assert_eq!(best_arm(simul, 0, &[true, false, false], DUCT), Some(0));
        assert_eq!(best_arm(simul, 0, &[false, false, false], RM), None);
    }

    /// `RootPolicy::MostVisited` has to answer with an arm even where the
    /// policy puts no weight on any of them.
    ///
    /// Regret matching's `sigma` is exactly zero for an arm dominated wherever
    /// it is legal, so its `strategy_sum` never leaves zero while the
    /// exploration floor keeps handing it visits: a legal set carrying no
    /// strategy mass at all is a state the search reaches and stays in, not a
    /// startup transient. Answering `None` there sent the caller to its own
    /// uniform draw over the position's choice list, which reports a hardcoded
    /// zero visit count and zero mean for a marginal the tree holds real
    /// statistics for.
    #[test]
    fn the_deterministic_root_move_answers_a_zero_mass_legal_set() {
        let mut node = simul_node(PlayerSet::first_n(1));
        node.expand_marginals(0, &[0u8, 1, 2], true, usize::MAX);
        let simul = node.simul_mut().expect("simultaneous");
        for (arm, (visits, reward)) in [(100u32, 40.0), (100, 70.0), (1, 1.0)]
            .into_iter()
            .enumerate()
        {
            simul.arm_stats[arm].visits = visits;
            simul.arm_stats[arm].availability = visits;
            simul.arm_stats[arm].cumulative_reward = reward;
            assert_eq!(simul.arm_policy[arm].strategy_sum, 0.0);
        }

        let all = [true; 3];
        let mut strategy = Vec::new();
        root_strategy_into(simul, 0, &all, RM, &mut strategy);
        assert_eq!(
            strategy,
            vec![1.0 / 3.0; 3],
            "the strategy this answer has to be an argmax of is the uniform fallback"
        );
        // Every legal arm is tied under that fallback, and the tie is broken by
        // the crate's root ranking rather than by index order: arm 1 measured
        // 0.7 against arm 0's 0.4, and arm 2's perfect mean is one selection of
        // evidence.
        assert_eq!(best_arm(simul, 0, &all, RM), Some(1));
        assert_eq!(best_arm(simul, 0, &[true, false, false], RM), Some(0));
        // An arm nothing was ever selected at is still nothing to report.
        assert_eq!(best_arm(simul, 0, &[false; 3], RM), None);
    }

    /// Rescaling a game's payoffs, the declared reward range and the
    /// exploration constant together leaves `Duct` selection identical — arm
    /// for arm, draw for draw, and including which arms land in the tie pool.
    ///
    /// That is the sense in which the range is a *scale*. It is also why
    /// declaring one wider than the payoffs really are is not free under
    /// `Duct`: the tie tolerance is a fraction of the declared span and nothing
    /// else moves with it, so over-declaring widens the pool until arms the
    /// search can tell apart are drawn between uniformly.
    #[test]
    fn duct_selection_is_invariant_under_rescaling_the_reward_scale() {
        // A power of two, so every scaled product below is exact and the two
        // runs are comparable draw for draw rather than to within a tolerance.
        const SCALE: f64 = 4.0;
        const DRAWS: usize = 2_000;

        let draws = |scale: f64| -> Vec<u32> {
            let mut node = simul_node(PlayerSet::first_n(1));
            node.expand_marginals(0, &[0u8, 1, 2, 3], false, usize::MAX);
            let simul = node.simul_mut().expect("simultaneous");
            // Arms 0 and 1 are exactly tied and lead; arm 3 sits 0.03 of the
            // span below them, which is outside the tolerance and inside three
            // times it.
            for (arm, (visits, reward)) in [(20u32, 9.0), (20, 9.0), (12, 3.0), (40, 21.0)]
                .into_iter()
                .enumerate()
            {
                let stats = &mut simul.arm_stats[arm];
                stats.visits = visits;
                stats.availability = 100;
                stats.ln_availability = 100f64.ln();
                stats.cumulative_reward = reward * scale;
            }
            let mut rng = WyRand::seed_from_u64(0x5CA1);
            (0..DRAWS)
                .map(|_| {
                    select_marginal(simul, 0, 0, DUCT, ucb() * scale, scale, &mut rng)
                        .expect("every arm is legal")
                        .0
                })
                .collect()
        };

        let plain = draws(1.0);
        assert!(
            plain.contains(&0) && plain.contains(&1),
            "the tie pool is what this test is about, and it did not form"
        );
        assert!(
            !plain.contains(&2) && !plain.contains(&3),
            "an arm more than a tolerance below the leaders was drawn"
        );
        assert_eq!(plain, draws(SCALE));
    }

    /// The rule this crate deliberately does not use, kept so that the test
    /// above can say what the corrected answer is corrected *away from*.
    fn raw_visit_argmax<C>(simul: &Simul<C>, slot: usize, legal: &[bool]) -> Option<usize> {
        let start = simul.starts[slot] as usize;
        let mut best = None;
        let mut best_visits = 0;
        for (arm, &is_legal) in legal.iter().enumerate() {
            if is_legal && simul.arm_stats[start + arm].visits > best_visits {
                best_visits = simul.arm_stats[start + arm].visits;
                best = Some(arm);
            }
        }
        best
    }

    /// The reservoir draws in `select_duct` are only uniform if `below` is: the
    /// first candidate must always be taken, and the `k`-th must replace the
    /// incumbent exactly one time in `k`.
    #[test]
    fn the_reservoir_counter_accepts_the_kth_candidate_one_time_in_k() {
        let mut rng = WyRand::seed_from_u64(0x4E57);
        for _ in 0..10_000 {
            assert_eq!(
                below(&mut rng, 1),
                0,
                "the first candidate is never skipped"
            );
        }
        let draws = 400_000u64;
        for k in 2..=5u64 {
            let accepted = (0..draws).filter(|_| below(&mut rng, k) == 0).count() as f64;
            let want = draws as f64 / k as f64;
            let sigma = (want * (1.0 - 1.0 / k as f64)).sqrt();
            assert!(
                (accepted - want).abs() < 5.0 * sigma,
                "the {k}th candidate was accepted {accepted} times in {draws}, not about {want}"
            );
        }
    }

    #[test]
    fn sampling_follows_the_mixed_distribution_it_reports() {
        let mut node = simul_node(PlayerSet::first_n(1));
        node.expand_marginals(0, &[0u8, 1, 2, 3], true, usize::MAX);
        let simul = node.simul_mut().expect("simultaneous");
        for (arm, regret) in [3.0, 1.0, 0.0, -2.0].into_iter().enumerate() {
            simul.arm_policy[arm].regret = regret;
        }

        // sigma = (0.75, 0.25, 0, 0); at epoch 0 the floor is the whole
        // exploration constant, so mu = 0.96 * sigma + 0.04 / 4.
        let want = [0.73, 0.25, 0.01, 0.01];
        let draws = 200_000;
        let mut counts = [0u64; 4];
        let mut rng = WyRand::seed_from_u64(0x5A3D);
        for _ in 0..draws {
            let (arm, prob) =
                select_marginal(simul, 0, 0, RM, 0.04, 1.0, &mut rng).expect("every arm is legal");
            assert!(
                (prob - want[arm as usize]).abs() < 1e-12,
                "arm {arm} reported probability {prob}, not {}",
                want[arm as usize]
            );
            counts[arm as usize] += 1;
        }

        let expected: Vec<f64> = want.iter().map(|p| p * draws as f64).collect();
        let chi2 = chi_square(&counts, &expected);
        assert!(
            chi2 < CHI2_3DF,
            "draws {counts:?} against {expected:?}, chi2 {chi2}"
        );
    }

    #[test]
    fn an_arm_legal_half_the_time_keeps_a_usable_regret() {
        let mut node = simul_node(PlayerSet::first_n(2));
        let mut rng = WyRand::seed_from_u64(0x5EEE);
        let iterations = 20_000;
        let mut legal_for_arm_two = 0u32;
        for iteration in 0..iterations {
            let epoch = node.visits();
            // Every payoff is identical, so no arm deserves any regret at all
            // and the only thing that can move arm 2's is the subtraction.
            let sometimes = iteration % 2 == 0;
            if sometimes {
                legal_for_arm_two += 1;
            }
            let actions: &[u8] = if sometimes { &[0, 1, 2] } else { &[0, 1] };
            node.expand_marginals(0, actions, true, usize::MAX);
            node.expand_marginals(1, &[0u8, 1], true, usize::MAX);
            let simul = node.simul_mut().expect("simultaneous");
            let (a0, mu0) = select_marginal(simul, 0, epoch, RM, floor(), 1.0, &mut rng).unwrap();
            let (a1, mu1) = select_marginal(simul, 1, epoch, RM, floor(), 1.0, &mut rng).unwrap();
            let key = JointKey::pack(&[a0, a1], simul.key_bits);
            credit_marginals(
                simul,
                key,
                epoch,
                &[mu0 as f32, mu1 as f32],
                &[0.5, 0.5],
                RM,
                0.0,
                1.0,
            );
            node.record(0.0);
        }

        let simul = node.simul().expect("simultaneous");
        let arm = &simul.arm_stats[2];
        assert_eq!(
            arm.availability, legal_for_arm_two,
            "arm 2 should have been available on exactly the iterations that offered it"
        );
        assert!(arm.visits > 0, "arm 2 was never selected");
        // Without the legality filter the subtraction would run on every
        // iteration and drive this to about -0.5 * 10 000 = -5000. With it, the
        // estimator leaves a random walk: measured, +92 after 2459 selections
        // out of 10 000 availabilities.
        let regret = simul.arm_policy[2].regret;
        assert!(
            regret > -1000.0,
            "arm 2's regret ran to {regret}; the sleeping-bandit filter is not applying"
        );
    }

    #[test]
    fn duct_breaks_a_tie_uniformly() {
        let mut node = simul_node(PlayerSet::first_n(1));
        node.expand_marginals(0, &[0u8, 1, 2, 3], false, usize::MAX);
        let simul = node.simul_mut().expect("simultaneous");
        for arm in 0..4 {
            let stats = &mut simul.arm_stats[arm];
            stats.visits = 10;
            stats.cumulative_reward = 5.0;
            stats.availability = 10;
            stats.ln_availability = 10f64.ln();
        }

        let draws = 100_000;
        let mut counts = [0u64; 4];
        let mut rng = WyRand::seed_from_u64(0xD0C7);
        for _ in 0..draws {
            let (arm, prob) = select_marginal(simul, 0, 0, DUCT, ucb(), 1.0, &mut rng)
                .expect("every arm is legal");
            assert_eq!(prob, 1.0, "Duct does no importance weighting");
            counts[arm as usize] += 1;
        }

        let expected = [draws as f64 / 4.0; 4];
        let chi2 = chi_square(&counts, &expected);
        assert!(chi2 < CHI2_3DF, "tie draws {counts:?}, chi2 {chi2}");
    }

    #[test]
    fn duct_opens_unvisited_arms_uniformly() {
        let mut node = simul_node(PlayerSet::first_n(1));
        node.expand_marginals(0, &[0u8, 1, 2, 3], false, usize::MAX);
        let simul = node.simul_mut().expect("simultaneous");
        simul.arm_stats[1].visits = 7;
        simul.arm_stats[1].cumulative_reward = 7.0;

        let draws = 100_000;
        let mut counts = [0u64; 4];
        let mut rng = WyRand::seed_from_u64(0x11FE);
        for _ in 0..draws {
            let (arm, _) = select_marginal(simul, 0, 0, DUCT, ucb(), 1.0, &mut rng).unwrap();
            counts[arm as usize] += 1;
        }

        assert_eq!(
            counts[1], 0,
            "a visited arm cannot win against an unvisited one"
        );
        let expected = [draws as f64 / 3.0; 3];
        let observed = [counts[0], counts[2], counts[3]];
        let chi2 = chi_square(&observed, &expected);
        assert!(chi2 < 13.816, "unvisited draws {counts:?}, chi2 {chi2}");
    }

    #[test]
    fn nothing_is_selected_when_no_arm_is_legal() {
        let mut node = simul_node(PlayerSet::first_n(1));
        node.expand_marginals(0, &[0u8, 1], true, usize::MAX);
        node.record(0.0);
        let simul = node.simul_mut().expect("simultaneous");
        let mut rng = WyRand::seed_from_u64(1);
        // Epoch 1 stamps nothing: the arms were last legal at epoch 0.
        assert!(select_marginal(simul, 0, 1, RM, floor(), 1.0, &mut rng).is_none());
        assert!(select_marginal(simul, 0, 1, DUCT, ucb(), 1.0, &mut rng).is_none());
    }

    #[test]
    fn an_empty_reward_range_normalizes_to_the_midpoint() {
        assert_eq!(normalize_reward(3.0, 5.0, 0.0), 0.5);
        assert_eq!(normalize_reward(-4.0, -4.0, 8.0), 0.0);
        assert_eq!(normalize_reward(4.0, -4.0, 8.0), 1.0);
        assert_eq!(normalize_reward(0.0, -4.0, 8.0), 0.5);
        assert_eq!(normalize_reward(99.0, -4.0, 8.0), 1.0);
    }

    #[test]
    fn root_extraction_divides_out_availability() {
        let mut node = simul_node(PlayerSet::first_n(1));
        node.expand_marginals(0, &[0u8, 1], true, usize::MAX);
        let simul = node.simul_mut().expect("simultaneous");
        // Equal strategy mass, but one arm was legal five times as often — so
        // the arm that was rarely legal was played on nearly every iteration
        // that offered it. `strategy_sum` can never exceed `availability`,
        // which is what makes this a state the search can actually reach.
        for arm in 0..2 {
            simul.arm_policy[arm].strategy_sum = 20.0;
        }
        simul.arm_stats[0].availability = 100;
        simul.arm_stats[1].availability = 20;

        let mut strategy = Vec::new();
        root_strategy_into(simul, 0, &[true, true], RM, &mut strategy);
        assert!((strategy[0] - 1.0 / 6.0).abs() < 1e-12, "{strategy:?}");
        assert!((strategy[1] - 5.0 / 6.0).abs() < 1e-12, "{strategy:?}");
    }

    #[test]
    fn root_extraction_renormalizes_over_the_legal_arms() {
        let node = play_biased_rps(20_000, floor(), 0xC0FE);
        let simul = node.simul().expect("simultaneous");
        let mut strategy = Vec::new();
        root_strategy_into(simul, 0, &[true, false, true], RM, &mut strategy);
        assert_eq!(strategy[1], 0.0);
        assert!(
            (strategy.iter().sum::<f64>() - 1.0).abs() < 1e-12,
            "{strategy:?}"
        );

        // Duct puts everything on the highest-ranked arm it is allowed to have.
        root_strategy_into(simul, 0, &[true, true, true], DUCT, &mut strategy);
        let visits: Vec<u32> = (0..3).map(|a| simul.arm_stats[a].visits).collect();
        let best = leading_arm(simul, 0, &[true, true, true]).unwrap();
        assert_eq!(
            strategy.iter().filter(|&&p| p > 0.0).count(),
            1,
            "{strategy:?}"
        );
        assert_eq!(
            strategy[best], 1.0,
            "{strategy:?} against visits {visits:?}"
        );
    }

    #[test]
    fn root_extraction_falls_back_to_uniform_over_the_legal_arms() {
        let mut node = simul_node(PlayerSet::first_n(1));
        node.expand_marginals(0, &[0u8, 1, 2], true, usize::MAX);
        let simul = node.simul().expect("simultaneous");

        let mut strategy = Vec::new();
        root_strategy_into(simul, 0, &[true, false, true], RM, &mut strategy);
        assert_eq!(strategy, vec![0.5, 0.0, 0.5]);

        root_strategy_into(simul, 0, &[false, false, false], RM, &mut strategy);
        assert_eq!(strategy, vec![0.0, 0.0, 0.0]);

        let mut rng = WyRand::seed_from_u64(2);
        assert_eq!(sample_root_arm(simul, 0, &[false; 3], RM, &mut rng), None);
    }

    #[test]
    fn sampled_and_reported_root_strategies_agree() {
        let node = play_biased_rps(50_000, floor(), 0xA11E);
        let simul = node.simul().expect("simultaneous");
        let legal = [true, false, true];
        let mut strategy = Vec::new();
        root_strategy_into(simul, 0, &legal, RM, &mut strategy);

        let draws = 100_000;
        let mut counts = [0u64; 3];
        let mut rng = WyRand::seed_from_u64(0x7A11);
        for _ in 0..draws {
            counts[sample_root_arm(simul, 0, &legal, RM, &mut rng).unwrap()] += 1;
        }
        assert_eq!(counts[1], 0, "an illegal arm was drawn");
        let observed = [counts[0], counts[2]];
        let expected = [strategy[0] * draws as f64, strategy[2] * draws as f64];
        let chi2 = chi_square(&observed, &expected);
        assert!(
            chi2 < 10.828,
            "{observed:?} against {expected:?}, chi2 {chi2}"
        );
    }
}
