//! Correctness and determinism of the simultaneous-move machinery
//! (design sections 7.2 and 7.5).
//!
//! Every test here runs at [`Config::default`]'s exploration settings. That is
//! deliberate and it is a regression gate: the first cut of this feature
//! shipped a `gamma_0 / sqrt(t)` exploration floor that does not converge at
//! any budget, and these tolerances were originally met only by setting
//! `simultaneous_exploration` to 6.4 — a value no caller would guess, hiding a
//! default that learned essentially nothing. If a tolerance below stops being
//! met, the schedule in `duct.rs`'s `exploration_floor` is what to look at, not
//! the number in the assertion.

// The other integration binaries use the sequential fixtures; this one does
// not, and each test binary compiles `tests/common` on its own.
#[allow(dead_code)]
mod common;

use common::*;
use mcts::rand_core::{Rng, SeedableRng};
use mcts::{
    Config, Edge, Game, JointChoices, NodeKind, PlayerSet, RootPolicy, SearchResult, Searcher,
    SimultaneousConfig, SimultaneousPolicy, Status, StopReason,
};
use wyrand::WyRand;

fn rng(seed: u64) -> WyRand {
    WyRand::seed_from_u64(seed)
}

/// The shipped defaults with a budget and without early termination.
///
/// Early termination is off so that every test spends its whole budget: at a
/// mixed simultaneous root the proof is gated off anyway, but these games are
/// also searched sequentially and under `Duct`, where it fires.
///
/// Nothing else is overridden. `Config::duct_exploration` and
/// `Config::regret_matching_exploration` are separate fields precisely so that
/// one config can be right for both policies, which is why `Duct` and regret
/// matching tests here share this function.
fn config(iterations: u32) -> Config {
    Config {
        iterations,
        early_termination: false,
        ..Config::default()
    }
}

/// Search `game` on behalf of `player` and return the result alongside the
/// player's extracted mixed strategy, ordered by choice.
fn search<G: Game<Choice = u8, Context = ()>>(
    game: &G,
    player: u8,
    cfg: &Config,
    seed: u64,
) -> (SearchResult<u8>, Vec<f64>) {
    let mut searcher = Searcher::new(game);
    let result = searcher.search(game, &(), player, cfg, None, &mut rng(seed));
    let mut pairs = Vec::new();
    assert!(
        searcher.root_policy_into(game, &(), player, &mut pairs),
        "the root is simultaneous and {player} acts at it"
    );
    pairs.sort_by_key(|(choice, _)| *choice);
    (result, pairs.into_iter().map(|(_, p)| p).collect())
}

/// Just the mixed strategy, for the tests that do not read the result.
fn mixture<G: Game<Choice = u8, Context = ()>>(
    game: &G,
    player: u8,
    cfg: &Config,
    seed: u64,
) -> Vec<f64> {
    search(game, player, cfg, seed).1
}

fn linf(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// How much a best-responding opponent gains over the value of [`BiasedRps`]
/// by knowing `mix` is what `player` plays.
///
/// The quantity the design's exploitability harness gates on, and a fairer
/// judge of a strategy than L-infinity: biased rock-paper-scissors is flat near
/// its equilibrium, so a mixture can sit 0.1 away in probability and still give
/// almost nothing up. Uniform play is exploitable by 0.0833.
fn biased_rps_exploitability(mix: &[f64], player: u8) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for (reply, _) in BIASED_RPS_PAYOFFS.iter().enumerate() {
        let mut value = 0.0;
        for (action, weight) in mix.iter().enumerate() {
            let to_player = match player {
                0 => BIASED_RPS_PAYOFFS[action][reply],
                _ => BIASED_RPS_PAYOFFS[reply][action],
            };
            value += weight * (1.0 - to_player);
        }
        best = best.max(value);
    }
    best - 0.5
}

/// `G` played under [`SimultaneousPolicy::Duct`] instead of its own policy.
///
/// The fixtures all take the crate default, and the point of the two `Duct`
/// tests is to run the *same game* under the other rule: a fixture pair that
/// differed in anything else would not be evidence about the policy.
#[derive(Clone, Default)]
struct Ducted<G>(G);

impl<G: Game> Game for Ducted<G> {
    type Choice = G::Choice;
    type Rewards = G::Rewards;
    type Context = G::Context;
    type Side = G::Side;

    const ROOT_CHOICES_INVARIANT: bool = G::ROOT_CHOICES_INVARIANT;
    const CHILD_INDEX_THRESHOLD: usize = G::CHILD_INDEX_THRESHOLD;
    const SIMULTANEOUS_POLICY: SimultaneousPolicy = SimultaneousPolicy::Duct;

    fn status(&self, ctx: &Self::Context) -> Status<Self::Rewards> {
        self.0.status(ctx)
    }

    fn choices_into(&self, ctx: &Self::Context, out: &mut Vec<Self::Choice>) {
        self.0.choices_into(ctx, out)
    }

    fn choices_for_into(&self, ctx: &Self::Context, player: u8, out: &mut Vec<Self::Choice>) {
        self.0.choices_for_into(ctx, player, out)
    }

    fn apply_choice<R: Rng + ?Sized>(
        &mut self,
        ctx: &Self::Context,
        choice: &Self::Choice,
        rng: &mut R,
    ) {
        self.0.apply_choice(ctx, choice, rng)
    }

    fn apply_joint<R: Rng + ?Sized>(
        &mut self,
        ctx: &Self::Context,
        joint: JointChoices<'_, Self::Choice>,
        rng: &mut R,
    ) {
        self.0.apply_joint(ctx, joint, rng)
    }

    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &Self::Context, rng: &mut R) -> Self::Rewards {
        self.0.rollout(ctx, rng)
    }

    fn new_buffer(&self) -> Self {
        Ducted(self.0.new_buffer())
    }

    fn determinize_into<R: Rng + ?Sized>(
        &self,
        dest: &mut Self,
        ctx: &Self::Context,
        perspective: u8,
        rng: &mut R,
    ) {
        self.0.determinize_into(&mut dest.0, ctx, perspective, rng)
    }

    fn advance<R: Rng + ?Sized>(
        &mut self,
        ctx: &Self::Context,
        side: &mut Self::Side,
        perspective: u8,
        rng: &mut R,
    ) {
        self.0.advance(ctx, side, perspective, rng)
    }

    fn init_side(&self, ctx: &Self::Context, side: &mut Self::Side) {
        self.0.init_side(ctx, side)
    }

    fn begin_iteration(side: &mut Self::Side) {
        G::begin_iteration(side)
    }

    fn credit_iteration(side: &mut Self::Side, rewards: &Self::Rewards) {
        G::credit_iteration(side, rewards)
    }
}

/// A simultaneous node at which player 1 has no legal action at all, so no
/// joint action exists.
///
/// Section 7.1 lists no fixture for this and section 7.2 requires one: a game
/// can reach a degenerate position through its own rules, and the library's
/// contract is that release builds score the iteration zero instead of reading
/// past the end of an empty slot.
#[derive(Clone, Default)]
struct NoLegalReply {
    resolved: bool,
}

impl Game for NoLegalReply {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.resolved {
            true => Status::Terminal([0.5, 0.5]),
            false => Status::Simultaneous {
                players: PlayerSet::first_n(2),
            },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend([0, 1]);
    }

    fn choices_for_into(&self, _: &(), player: u8, out: &mut Vec<u8>) {
        if player == 0 {
            out.extend([0, 1]);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("NoLegalReply has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), _: JointChoices<'_, u8>, _: &mut R) {
        unreachable!("NoLegalReply has no joint action to apply")
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        [0.5, 0.5]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

// ---------------------------------------------------------------------------
// 7.2 Correctness of the simultaneous machinery
// ---------------------------------------------------------------------------

/// Rock-paper-scissors has no best move, and a search that reports one is
/// exploitable by exactly as much as it commits.
///
/// Asserted for both players, because the two slots are separate arrays and a
/// slot-indexing bug shows up in one of them only.
#[test]
fn rps_does_not_converge_to_a_pure_strategy() {
    for player in 0..2 {
        let mix = mixture(&Rps::default(), player, &config(50_000), 7);
        assert_eq!(mix.len(), 3);
        for (action, weight) in mix.iter().enumerate() {
            assert!(
                (0.2..=0.5).contains(weight),
                "player {player} plays action {action} with probability {weight}, \
                 outside [0.2, 0.5]; the mixture is {mix:?}"
            );
        }
        let total: f64 = mix.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "the mixture sums to {total}");
    }
}

/// Payoff to player 0 at [`DominantPair`], indexed
/// `[player 0's choice][player 1's choice]`.
///
/// Zero-sum on `[0, 1]`. Row 1 strictly dominates row 0 for player 0 (0.7 > 0.2
/// and 0.8 > 0.3) and column 0 strictly dominates column 1 for player 1, who is
/// paid `1 - payoff` and therefore minimizes this table (0.2 < 0.3 and
/// 0.7 < 0.8). So the unique equilibrium is pure and *asymmetric* — player 0
/// plays 1, player 1 plays 0 — which an index-ordered tie-break would fake and
/// a symmetric one would not distinguish.
const DOMINANT_PAIR_PAYOFFS: [[f64; 2]; 2] = [[0.2, 0.3], [0.7, 0.8]];

/// The equilibrium action of each player at [`DominantPair`], by slot.
const DOMINANT_PAIR_EQUILIBRIUM: [u8; 2] = [1, 0];

/// One simultaneous ply whose equilibrium is **pure**, which the rest of these
/// fixtures deliberately are not.
///
/// It is the case [`SimultaneousPolicy::Duct`] is sold for — "the right choice
/// when you know your simultaneous nodes have pure equilibria" — and without it
/// every `Duct` test here would be measuring the policy on games where it is
/// known to be the wrong tool.
#[derive(Clone, Default)]
struct DominantPair {
    payoff: Option<f64>,
}

impl Game for DominantPair {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.payoff {
            Some(payoff) => Status::Terminal([payoff, 1.0 - payoff]),
            None => Status::Simultaneous {
                players: PlayerSet::first_n(2),
            },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend([0, 1]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("DominantPair has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(DOMINANT_PAIR_PAYOFFS[*joint.get(0) as usize][*joint.get(1) as usize]);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let payoff = *self.payoff.get_or_insert_with(|| {
            let row = (rng.next_u64() >> 63) as usize;
            let col = (rng.next_u64() >> 63) as usize;
            DOMINANT_PAIR_PAYOFFS[row][col]
        });
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// Where the equilibrium is pure, both policies find it and commit to it.
///
/// This is the claim `Duct` exists for, and it is also the one place a `Duct`
/// test can assert *visit concentration*. It cannot be asserted on plain
/// rock-paper-scissors: there every action is worth exactly 0.5 against
/// uniform, so a UCB1 constant large enough to be worth using keeps the visits
/// spread, and the only thing that concentrates them is a constant small enough
/// to make the bandit near-greedy — which is the regime
/// `Config::duct_exploration`'s doc comment measures as costing half the value
/// of the game. Concentration on rock-paper-scissors was a symptom of a bad
/// default, not evidence of a working policy.
#[test]
fn both_policies_commit_to_a_pure_equilibrium() {
    for player in 0..2u8 {
        let want = DOMINANT_PAIR_EQUILIBRIUM[player as usize];

        let mix = mixture(&DominantPair::default(), player, &config(20_000), 5);
        assert!(
            mix[want as usize] > 0.9,
            "regret matching plays {mix:?} for player {player}, which does not commit \
             to the dominant action {want}"
        );

        let game = Ducted(DominantPair::default());
        let mut searcher = Searcher::new(&game);
        searcher.search(&game, &(), player, &config(20_000), None, &mut rng(5));
        let root = searcher.tree().expect("the search retained its tree");
        let marginals = root.marginals(player).expect("the player acts at the root");
        let visits: u32 = (0..marginals.len()).map(|arm| marginals.visits(arm)).sum();
        let top = marginals.most_visited().expect("some arm was selected");
        assert_eq!(
            *marginals.choice(top),
            want,
            "Duct's most-visited arm for player {player} is not the dominant action {want}"
        );
        assert!(
            f64::from(marginals.visits(top)) / f64::from(visits) > 0.9,
            "Duct gave the dominant action {} of {visits} arm visits, which is not the \
             concentration a pure equilibrium should produce",
            marginals.visits(top)
        );
    }
}

/// The same game under `Duct`, which is why `Duct` is not the default.
///
/// Decoupled UCB1 hands back a pure strategy where the equilibrium is mixed,
/// and this pins that: `policy_into` is one-hot however the visits fall. It
/// fails loudly if `Duct` ever silently starts mixing — at which point the
/// contract in `Marginals::policy_into`'s doc, that a `Duct` strategy is never
/// handed back dressed as a mixed one, would be broken.
///
/// The arm it commits to is `Marginals::leader`, not `Marginals::most_visited`.
/// Those coincide only where the visit argmax and the ranking agree, and on a
/// mixed equilibrium they need not: three arms that are worth the same to within
/// noise split the budget nearly evenly, so the best measured arm and the most
/// selected one come apart on a coin toss.
#[test]
fn duct_extracts_a_pure_strategy_on_rps() {
    let game = Ducted(Rps::default());
    for player in 0..2 {
        let mut searcher = Searcher::new(&game);
        searcher.search(&game, &(), player, &config(50_000), None, &mut rng(7));
        let root = searcher.tree().expect("the search retained its tree");

        let mut strategy = Vec::new();
        root.marginals(player)
            .expect("the player acts at the root")
            .policy_into(&mut strategy);
        assert_eq!(strategy.len(), 3);
        assert_eq!(
            strategy.iter().filter(|weight| **weight > 0.0).count(),
            1,
            "Duct's extracted strategy is {strategy:?}, which is not one-hot"
        );
        let picked = strategy.iter().position(|weight| *weight > 0.0).unwrap();
        let marginals = root.marginals(player).expect("the player acts at the root");
        assert_eq!(
            Some(picked),
            marginals.leader(),
            "Duct's extracted strategy {strategy:?} does not agree with its ranking"
        );
    }
}

/// The load-bearing correctness test for the whole feature: on a game whose
/// unique equilibrium is mixed, known exactly and far from uniform, the
/// extracted strategy is that equilibrium.
///
/// Averaged over four seeded runs rather than read off one, because the per-run
/// distribution is heavy-tailed: at this budget the worst single run of the 64
/// behind these numbers sat 0.082 from the equilibrium while the worst
/// four-run *average* sat 0.038. Averaging is what an exploitability harness
/// does anyway, and the average of four seeded runs is still exactly
/// reproducible. Measured over sixteen independent groups of four seeds at the
/// shipped exploration default: worst group L-infinity 0.038, worst group
/// exploitability 0.0068. Seeds 1..=4 are one of the worse groups, at 0.038 and
/// 0.0027.
///
/// An average is also the one shape in which three good runs can carry a
/// degenerate fourth, so the tight assertions on the average are paired with a
/// loose **per-run floor** that no single run may fall through. The floors are
/// not tuned to the runs: each is set to half of what the same measure scores
/// for uniform play — the strategy a search that learned nothing reports — and
/// the test derives both from [`BIASED_RPS_EQUILIBRIUM`] and
/// [`biased_rps_exploitability`] rather than restating them, so a floor that
/// stopped meaning "better than learning nothing" fails on the spot. Measured
/// per run at seeds 1..=4: L-infinity 0.049 / 0.052 / 0.032 / 0.023 against the
/// 0.12 floor, exploitability 0.0036 / 0.0060 / 0.0032 / 0.0017 against the 0.04
/// floor. Over a wider survey of 24 seeds the worst single run sat at 0.082 and
/// 0.0158, still inside both.
#[test]
fn biased_rps_learns_the_known_equilibrium() {
    const RUNS: usize = 4;
    // Loose enough that no run of the 24 surveyed comes near it, tight enough
    // that a run which learned nothing cannot pass — see the assertions below,
    // which check that second half rather than assume it.
    const RUN_FLOOR_LINF: f64 = 0.12;
    const RUN_FLOOR_EXPLOITABILITY: f64 = 0.04;

    let game = BiasedRps::default();
    let uninformed = [1.0 / 3.0; 3];
    assert!(
        RUN_FLOOR_LINF < linf(&uninformed, &BIASED_RPS_EQUILIBRIUM) / 2.0
            && RUN_FLOOR_EXPLOITABILITY < biased_rps_exploitability(&uninformed, 0) / 2.0,
        "the per-run floors {RUN_FLOOR_LINF} and {RUN_FLOOR_EXPLOITABILITY} are not \
         inside half of what uniform play scores ({} and {}), so a run that learned \
         nothing would clear them and the floor would be decoration",
        linf(&uninformed, &BIASED_RPS_EQUILIBRIUM),
        biased_rps_exploitability(&uninformed, 0)
    );

    let mut average = [0.0; 3];
    for seed in 1..=RUNS as u64 {
        let mix = mixture(&game, 0, &config(800_000), seed);
        assert_eq!(mix.len(), 3);

        // Only regret matching can hand back a mixture with every action alive;
        // `Duct`'s extraction is one-hot by construction. A silent route to the
        // other policy has to fail here rather than be averaged away.
        assert!(
            mix.iter().all(|weight| *weight > 0.0),
            "seed {seed} extracted {mix:?}, which abandons an action outright — that \
             is a pure strategy, and this game's equilibrium is mixed"
        );

        let run_distance = linf(&mix, &BIASED_RPS_EQUILIBRIUM);
        let run_exploitability = biased_rps_exploitability(&mix, 0);
        assert!(
            run_distance < RUN_FLOOR_LINF && run_exploitability < RUN_FLOOR_EXPLOITABILITY,
            "seed {seed} alone extracted {mix:?}: {run_distance} from the equilibrium and \
             exploitable by {run_exploitability}, through the per-run floor of \
             {RUN_FLOOR_LINF} / {RUN_FLOOR_EXPLOITABILITY}. The average over the four \
             seeds may still look fine; one of them learned next to nothing"
        );

        for (action, weight) in mix.iter().enumerate() {
            average[action] += weight / RUNS as f64;
        }
    }

    let distance = linf(&average, &BIASED_RPS_EQUILIBRIUM);
    assert!(
        distance < 0.06,
        "the extracted strategy {average:?} is {distance} from the equilibrium \
         {BIASED_RPS_EQUILIBRIUM:?}"
    );

    // The same claim in the units that decide a game. Uniform play — the
    // strategy a search that learned nothing would report — gives up 0.0833.
    let exploitability = biased_rps_exploitability(&average, 0);
    assert!(
        exploitability < 0.02,
        "the extracted strategy {average:?} is exploitable by {exploitability}"
    );
}

/// The same budget under `Duct` misses the equilibrium, and cannot help but
/// miss it.
///
/// A one-hot strategy is at least 0.375 from `(1/16, 10/16, 5/16)` whichever
/// action it picks, so this is a bound rather than a measurement. It documents
/// the proved failure, and it fails if someone makes `Duct` mix.
#[test]
fn biased_rps_duct_misses_the_equilibrium() {
    let game = Ducted(BiasedRps::default());
    let mix = mixture(&game, 0, &config(50_000), 1);

    // The bound above is a statement about a *one-hot* strategy, so this is the
    // premise it rests on rather than a second measurement: a distance over 0.15
    // is also what a badly mixed strategy would show, and the two would be
    // indistinguishable here without it.
    assert_eq!(
        mix.iter().filter(|weight| **weight > 0.0).count(),
        1,
        "Duct extracted {mix:?}, which is not one-hot, so the distance below is not \
         the bound this test claims it is"
    );

    let distance = linf(&mix, &BIASED_RPS_EQUILIBRIUM);
    assert!(
        distance > 0.15,
        "Duct's strategy {mix:?} landed {distance} from the equilibrium, which means \
         it is mixing"
    );
}

/// **The test that justifies the whole feature.** The same payoff matrix,
/// searched twice: once as one simultaneous ply, once serialized into two.
///
/// Simultaneously, matching pennies is worth its true 0.5 to either player and
/// the equilibrium is uniform. Serialized, the second mover sees the committed
/// coin and mismatches it every time, so the first mover's root value collapses
/// to zero. That gap is not approximation error — serialization is a *bound* on
/// the value of a simultaneous node, and the bound is the whole distance from
/// 0.5 to 0.0 on a two-action game. The two halves live in one test so the
/// comparison cannot be read apart.
#[test]
fn matching_pennies_is_worth_half_simultaneously_and_sequentialized_it_is_not() {
    let (simultaneous, mix) = search(&Pennies::default(), 0, &config(50_000), 1);
    assert!(
        (simultaneous.best_mean_reward - PENNIES_VALUE).abs() < 0.05,
        "the simultaneous root scored {} for the first player, not {PENNIES_VALUE}",
        simultaneous.best_mean_reward
    );
    assert!(
        linf(&mix, &PENNIES_EQUILIBRIUM) < 0.06,
        "the simultaneous root plays {mix:?}, not {PENNIES_EQUILIBRIUM:?}"
    );

    let game = SequentialPennies::default();
    let mut searcher = Searcher::new(&game);
    let sequential = searcher.search(&game, &(), 0, &config(50_000), None, &mut rng(1));
    let root_value = searcher
        .tree()
        .expect("the search retained its tree")
        .mean_reward();
    assert!(
        root_value < 0.15,
        "serializing the same matrix left the first mover {root_value}, which is not \
         the collapse to {SEQUENTIAL_PENNIES_VALUE} that giving the opponent the last \
         word causes"
    );
    assert!(
        sequential.best_mean_reward < 0.15,
        "the best sequential first move scored {}",
        sequential.best_mean_reward
    );

    assert!(
        simultaneous.best_mean_reward - root_value > 0.3,
        "simultaneous {} vs serialized {root_value}: the two agree, which means the \
         simultaneous node is being resolved one player at a time",
        simultaneous.best_mean_reward
    );
}

/// Three players choosing at once, and none of them ends up regretting it.
///
/// **Renamed from section 7.2's `three_players_each_learn_a_uniform_mixture`,
/// and the uniformity assertion is gone**, because no no-regret algorithm
/// delivers it: bandit regret matching converges to the coarse correlated
/// equilibrium set, and in this three-cycle that set does not pin the
/// marginals. Measured at the shipped exploration default, the three marginals
/// settle into a rotating pattern near `(0.25, 0.5, 0.7)` in some order — which
/// player holds which is a function of the seed and of the budget, not of the
/// game — at every budget from 100k to 800k, and independent reimplementations
/// of section 4.3 under a constant floor and under a decaying one reproduce
/// exactly that. The unique Nash equilibrium really is uniform; independent
/// learners are simply not obliged to find it.
///
/// What *is* guaranteed, and is what this asserts: every player does at least
/// as well as the 0.5 they could have had by ignoring the search and flipping a
/// coin, all three of them keep mixing, and the three of them cannot all be
/// happy at once — an odd cycle cannot be two-coloured, so the payoffs sum to
/// strictly less than 2. Any of those breaking means a slot's arms are being
/// credited with another slot's reward.
#[test]
fn three_players_each_do_at_least_as_well_as_a_coin_flip() {
    let game = ThreeWayParity::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(100_000), None, &mut rng(1));
    let root = searcher.tree().expect("the search retained its tree");

    assert_eq!(root.simultaneous_players(), Some(PlayerSet::first_n(3)));
    assert_eq!(root.simultaneous_players().unwrap().len(), 3);

    let mut total = 0.0;
    for player in 0..3 {
        let marginals = root.marginals(player).expect("every player acts here");
        assert_eq!(marginals.len(), 2);

        let visits: f64 = (0..2).map(|arm| f64::from(marginals.visits(arm))).sum();
        let payoff: f64 = (0..2)
            .map(|arm| marginals.mean_reward(arm) * f64::from(marginals.visits(arm)))
            .sum::<f64>()
            / visits;
        assert!(
            payoff > 0.5 - 0.02,
            "player {player} averaged {payoff}, below the 0.5 a coin flip guarantees \
             against any opponents at all"
        );
        total += payoff;

        let mut strategy = Vec::new();
        marginals.policy_into(&mut strategy);
        for (action, weight) in strategy.iter().enumerate() {
            assert!(
                (0.10..=0.90).contains(weight),
                "player {player} plays action {action} with probability {weight}: they \
                 have stopped mixing, which in this game means they are being paid \
                 someone else's reward"
            );
        }
    }
    assert!(
        total < 2.0,
        "the three players shared {total} between them, which is more than the two \
         edges of an odd cycle that can mismatch at once"
    );
}

/// One tree, both kinds of node, and the sequential root reads the simultaneous
/// branch at its true value.
///
/// Branch A is a matching-pennies ply worth 0.5; branch B pays 0.25 outright.
/// A search that serialized the simultaneous node would value A at 0.0 and take
/// B, so the choice itself is the assertion. `node_count` is here because arms
/// are statistics, not nodes: two players times two actions is four arms at the
/// simultaneous node, and if any of them were stored as a `Node` the count
/// would be four too high.
#[test]
fn a_mixed_tree_searches_both_kinds_of_node() {
    let game = SeqThenSimul::default();
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &config(20_000), None, &mut rng(1));
    assert_eq!(
        result.choice, BRANCH_A,
        "the root took the branch worth {BRANCH_B_VALUE} over the one worth \
         {PENNIES_VALUE}"
    );

    let root = searcher.tree().expect("the search retained its tree");
    assert_eq!(root.kind(), NodeKind::Root);
    assert!(root.simultaneous_players().is_none());
    assert_eq!(root.children().len(), 2);

    // A plain `u8` throughout: no enum wrapper anywhere, checked by the
    // compiler rather than by an assertion.
    let _: &u8 = root.children()[0]
        .edge()
        .choice()
        .expect("a root child has a choice");

    let branch_a = root
        .children()
        .iter()
        .find(|child| *child.edge().choice().unwrap() == BRANCH_A)
        .expect("branch A was expanded");
    let branch_b = root
        .children()
        .iter()
        .find(|child| *child.edge().choice().unwrap() == BRANCH_B)
        .expect("branch B was expanded");
    assert_eq!(
        branch_a.simultaneous_players(),
        Some(PlayerSet::first_n(2)),
        "branch A leads to the simultaneous node"
    );
    assert!(branch_b.simultaneous_players().is_none());
    assert_eq!(branch_b.children().len(), 0, "branch B is terminal");
    assert!(
        (branch_a.mean_reward() - PENNIES_VALUE).abs() < 0.05,
        "branch A is valued at {}, not {PENNIES_VALUE}",
        branch_a.mean_reward()
    );

    let joint = branch_a.children().len();
    assert!(joint > 1, "only {joint} joint successors were materialized");
    assert_eq!(
        root.node_count(),
        3 + joint,
        "node_count is counting the four marginal arms as nodes"
    );
}

/// Two players with different action lists, and neither is handed the other's.
///
/// The failure this catches is a slot reading the wrong player's marginals,
/// which a symmetric fixture cannot see: here the action values themselves say
/// which player owns them.
#[test]
fn asymmetric_participants_get_their_own_action_lists() {
    let game = AsymmetricBid::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(200_000), None, &mut rng(1));
    let root = searcher.tree().expect("the search retained its tree");

    let bids = root.marginals(0).expect("player 0 acts here");
    let replies = root.marginals(1).expect("player 1 acts here");
    assert_eq!(bids.len(), 3);
    assert_eq!(replies.len(), 2);
    for arm in 0..bids.len() {
        assert!(
            BID_ACTIONS.contains(bids.choice(arm)),
            "player 0's arm {arm} plays {}, which is one of player 1's replies",
            bids.choice(arm)
        );
    }
    for arm in 0..replies.len() {
        assert!(
            RESPONSE_ACTIONS.contains(replies.choice(arm)),
            "player 1's arm {arm} plays {}, which is one of player 0's bids",
            replies.choice(arm)
        );
    }
    assert!(root.marginals(2).is_none(), "player 2 does not act here");

    // The unequal slot lengths reach the extraction too, which is where one
    // slot reading from the other's offset would show up as a player being
    // handed the wrong number of probabilities — or the wrong actions'.
    let mut pairs = Vec::new();
    assert!(searcher.root_policy_into(&game, &(), 0, &mut pairs));
    let bid_mix: Vec<f64> = {
        pairs.sort_by_key(|(choice, _)| *choice);
        pairs.iter().map(|(_, weight)| *weight).collect()
    };
    let reply_mix = mixture(&game, 1, &config(200_000), 1);
    assert_eq!(bid_mix.len(), 3);
    assert_eq!(reply_mix.len(), 2);
    assert!(
        linf(&bid_mix, &ASYMMETRIC_BID_P0_EQUILIBRIUM) < 0.10,
        "player 0 plays {bid_mix:?}, not {ASYMMETRIC_BID_P0_EQUILIBRIUM:?}"
    );
    assert!(
        linf(&reply_mix, &ASYMMETRIC_BID_P1_EQUILIBRIUM) < 0.10,
        "player 1 plays {reply_mix:?}, not {ASYMMETRIC_BID_P1_EQUILIBRIUM:?}"
    );
}

/// An action that is legal in only half the determinizations is still played.
///
/// This is the direct test of the availability correction. Without it, an arm's
/// regret is docked on every iteration it was not even allowed to play in, runs
/// to negative infinity at a rate proportional to how often it sleeps, and the
/// arm is never selected again — so the assertion that matters is that its
/// visits are healthy and its extracted weight has not collapsed.
///
/// The comparison against an always-legal run is **one-sided**, unlike section
/// 7.2's row. `policy_into` reports a probability *conditional on the action
/// being legal*, and `VariableRps` is a genuinely different game from
/// `BiasedRps` — player 1 is weaker in half the worlds, which moves player 0's
/// strategy and makes scissors worth more when it is available. Measured over
/// eight seeds, the conditional probability is 0.41-0.48 against the
/// always-legal run's 0.25-0.33: consistently *higher*, never starved. Starving
/// is a one-sided failure and this is the one-sided assertion for it.
#[test]
fn a_rarely_legal_action_is_not_starved() {
    const SCISSORS: u8 = 2;
    let game = VariableRps::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(100_000), None, &mut rng(1));
    let root = searcher.tree().expect("the search retained its tree");

    let marginals = root.marginals(1).expect("player 1 acts here");
    let scissors = (0..marginals.len())
        .find(|arm| *marginals.choice(*arm) == SCISSORS)
        .expect("player 1's scissors has an arm");

    // The scenario before the correction: this test is only about the
    // availability correction if the determinizations really did disagree about
    // what player 1 may play. Bracketed strictly, because both ends are a way for
    // the scenario to evaporate — an arm available on every iteration means
    // legality never varied and there was nothing to correct, and an arm
    // available on none of them means it was never in the game at all. Neither
    // shows up in the assertions below, which would both be satisfied by an
    // ordinary always-legal arm.
    assert!(
        marginals.availability(scissors) > 0 && marginals.availability(scissors) < root.visits(),
        "scissors was available on {} of the root's {} iterations; legality never \
         varied, so this measures an always-legal arm and proves nothing about the \
         availability correction",
        marginals.availability(scissors),
        root.visits()
    );

    // And the control for that bracket: every other arm in the tree *is*
    // always-legal, so a counter that had drifted for some unrelated reason —
    // counting selections, say, or missing iterations — would show it here
    // instead of only on the one arm the fixture withholds.
    for player in 0..2 {
        let all = root.marginals(player).expect("both players act here");
        for arm in 0..all.len() {
            if player == 1 && arm == scissors {
                continue;
            }
            assert_eq!(
                all.availability(arm),
                root.visits(),
                "player {player}'s arm {arm} plays {}, which every determinization \
                 offers, yet it was counted available on {} of {} iterations",
                all.choice(arm),
                all.availability(arm),
                root.visits()
            );
        }
    }

    let availability = f64::from(marginals.availability(scissors)) / f64::from(root.visits());
    assert!(
        (availability - VARIABLE_RPS_SCISSORS_RATE).abs() < 0.02,
        "scissors was available on {availability} of the iterations, not \
         {VARIABLE_RPS_SCISSORS_RATE}; the availability counter is not tracking \
         determinization"
    );

    let arm_visits: u32 = (0..marginals.len()).map(|arm| marginals.visits(arm)).sum();
    let share = f64::from(marginals.visits(scissors)) / f64::from(arm_visits);
    assert!(
        marginals.visits(scissors) > 0,
        "scissors was never selected: it has starved"
    );
    assert!(
        share > 0.15,
        "scissors took {share} of player 1's selections, far below the 0.26-0.31 \
         the availability correction leaves it over eight seeds"
    );

    let restricted = mixture(&game, 1, &config(100_000), 1);
    let always_legal = mixture(&BiasedRps::default(), 1, &config(100_000), 1);
    // Strictly above, not merely near. Dividing `strategy_sum` by availability
    // turns it into a mean probability *conditional on being legal*, so an arm
    // legal half the time and worth playing when it is legal must come back
    // weighted higher than the same arm in a run where it was always available
    // — not just within a tolerance of it. A one-sided `> always_legal - 0.10`
    // let the whole availability division be deleted with this test still
    // green: the correct code extracts 0.4624 here and dropping the division
    // gives 0.3007, both of which clear a bar of 0.1969.
    assert!(
        restricted[SCISSORS as usize] > always_legal[SCISSORS as usize] + 0.10,
        "scissors is played with probability {} where an always-legal run gives it \
         {}; the availability division should push the restricted run's weight \
         above the always-legal one, and scoring it on visits alone is what \
         collapses the gap",
        restricted[SCISSORS as usize],
        always_legal[SCISSORS as usize]
    );
}

/// Forty actions per player, which is past `CHILD_INDEX_THRESHOLD` in both
/// directions: the arm lookup and the joint-child lookup both leave their
/// linear scan for a hash index here, and nothing else in the suite reaches
/// that code.
#[test]
fn wide_simultaneous_nodes_track_every_arm_and_joint_child() {
    let game = ManyArm::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(20_000), None, &mut rng(1));
    let root = searcher.tree().expect("the search retained its tree");

    // The fixture is wide relative to a threshold it does not set: it takes
    // `CHILD_INDEX_THRESHOLD`'s default, so raising that default would leave
    // this test scanning linearly and asserting nothing about the index it
    // exists to cover, with no other symptom.
    assert!(
        usize::from(MANY_ARM_ACTIONS) > <ManyArm as Game>::CHILD_INDEX_THRESHOLD,
        "{MANY_ARM_ACTIONS} actions per player is not past the {} at which the arm \
         lookup builds an index, so the arm scan is what ran here",
        <ManyArm as Game>::CHILD_INDEX_THRESHOLD
    );

    for player in 0..2 {
        let marginals = root.marginals(player).expect("both players act here");
        assert_eq!(marginals.len(), usize::from(MANY_ARM_ACTIONS));
        let mut seen = vec![false; usize::from(MANY_ARM_ACTIONS)];
        for arm in 0..marginals.len() {
            let choice = usize::from(*marginals.choice(arm));
            assert!(!seen[choice], "action {choice} has two arms");
            seen[choice] = true;
        }
    }

    // Every action round-trips through the arm index, which is what
    // `root_policy_into` looks each of them up in.
    let mut pairs = Vec::new();
    assert!(searcher.root_policy_into(&game, &(), 0, &mut pairs));
    assert_eq!(pairs.len(), usize::from(MANY_ARM_ACTIONS));

    let children = root.children().len();
    assert!(
        children > usize::from(MANY_ARM_ACTIONS)
            && children > <ManyArm as Game>::CHILD_INDEX_THRESHOLD,
        "only {children} joint successors from 1600 possible pairs, which is not past \
         the {} at which the joint-child lookup builds an index either",
        <ManyArm as Game>::CHILD_INDEX_THRESHOLD
    );
    let mut seen = std::collections::HashSet::new();
    for child in 0..children {
        let first = root
            .joint_arm(child, 0)
            .expect("a joint child names an arm");
        let second = root
            .joint_arm(child, 1)
            .expect("a joint child names an arm");
        assert!(
            seen.insert((first, second)),
            "two joint children share the action pair ({first}, {second}), so the \
             joint index handed back the wrong node"
        );
        // The round trip section 7.2 asks for: an arm index is only meaningful
        // through the marginals it indexes.
        let played = *root.marginals(0).unwrap().choice(first);
        assert!(played < MANY_ARM_ACTIONS);
    }
    // A tree with one joint successor makes the uniqueness loop above vacuous:
    // there is no second pair for it to collide with, and the joint key could be
    // a constant.
    assert!(
        seen.len() > 1,
        "the root has {} distinct joint successors, so nothing above distinguished \
         one joint key from another",
        seen.len()
    );
    assert!(root.joint_arm(0, 2).is_none(), "player 2 does not act here");
    assert!(
        root.joint_arm(children, 0).is_none(),
        "child is out of range"
    );
}

/// A joint successor is not reached by anybody's single choice, and says so.
///
/// Code that walks `children()` and reads `choice().is_none()` as "this is the
/// root" would silently treat every joint successor as one; `kind` is what
/// keeps the two apart.
#[test]
fn a_joint_child_reports_no_single_choice() {
    const PERSPECTIVE: u8 = 1;
    let game = BiasedRps::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), PERSPECTIVE, &config(20_000), None, &mut rng(1));
    let root = searcher.tree().expect("the search retained its tree");

    // More than one, so that the loop below is a statement about joint
    // successors in general rather than about the single one a degenerate tree
    // would hold.
    assert!(
        root.children().len() > 1,
        "the root materialized {} joint successors",
        root.children().len()
    );
    for child in root.children() {
        assert_eq!(child.kind(), NodeKind::Joint);
        // `Edge::Joint`, not merely "carries no choice": a root carries no
        // choice either, and conflating the two is what `Edge` exists to stop.
        assert_eq!(child.edge(), Edge::Joint);
        assert_eq!(
            child.reward_player(),
            PERSPECTIVE,
            "a joint successor has no single mover, so it accumulates for the player \
             the search is running for"
        );
    }
}

/// A simultaneous node where one player has no legal action has no joint action
/// either. Release builds score the iteration zero and carry on.
#[cfg(not(debug_assertions))]
#[test]
fn a_degenerate_simultaneous_node_scores_zero() {
    let game = NoLegalReply::default();
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &config(1_000), None, &mut rng(1));

    assert!(
        result.choice == 0 || result.choice == 1,
        "the search returned {}, which is not one of player 0's actions",
        result.choice
    );
    assert_eq!(result.root_visits, 1_000, "every iteration still completed");
    assert_eq!(result.best_visits, 0, "no arm was ever selected");
    let root = searcher.tree().expect("the search retained its tree");
    assert_eq!(root.mean_reward(), 0.0, "the degenerate node scores zero");
}

/// The same position in a debug build, where it is a bug worth naming rather
/// than a zero worth returning: a node scored zero on every iteration drags its
/// parent's mean down indistinguishably from a bad evaluation.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "has no legal action at a simultaneous node")]
fn a_degenerate_simultaneous_node_is_a_debug_assertion() {
    let game = NoLegalReply::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(1_000), None, &mut rng(1));
}

// ---------------------------------------------------------------------------
// 7.5 Determinism
// ---------------------------------------------------------------------------

/// Sampling at a simultaneous node is seeded, so two identical runs are
/// identical — including the draw that picks the returned move, which happens
/// after the search loop and is the only part of a simultaneous search a caller
/// might mistake for non-determinism.
#[test]
fn same_seed_gives_the_same_simultaneous_search() {
    let game = BiasedRps::default();
    let run = || {
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &config(20_000), None, &mut rng(11));
        let root = searcher.tree().expect("the search retained its tree");
        let stats: Vec<(u8, u32, u32, f64)> = (0..2)
            .flat_map(|player| {
                let marginals = root.marginals(player).expect("both players act here");
                (0..marginals.len())
                    .map(|arm| {
                        (
                            *marginals.choice(arm),
                            marginals.visits(arm),
                            marginals.availability(arm),
                            marginals.mean_reward(arm),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        (result.choice, result.best_visits, result.root_visits, stats)
    };

    // Two equal tuples are only evidence if the tuples have something in them:
    // an empty `stats` and a zero visit count compare equal every time. Six arms
    // is two players times three actions, which is the whole state a repeat run
    // could differ in.
    let (_, best_visits, root_visits, stats) = run();
    assert_eq!(stats.len(), 6, "the two players hold six arms between them");
    assert_eq!(root_visits, 20_000, "the search spent its whole budget");
    assert!(best_visits > 0, "the returned action was never selected");

    assert_eq!(run(), run());
}

/// `RootPolicy::MostVisited` consumes no rng draw, so at a budget where the
/// most-selected action has settled it is the same action whatever the seed.
///
/// This is the property that makes `MostVisited` worth having: pinned output
/// for a caller that needs reproducibility across sessions, at the price of
/// being exploitable wherever the equilibrium is mixed.
#[test]
fn most_visited_root_policy_is_deterministic() {
    let game = BiasedRps::default();
    let cfg = Config {
        simultaneous: SimultaneousConfig {
            root_policy: RootPolicy::MostVisited,
            ..Default::default()
        },
        ..config(50_000)
    };
    let mut choices = Vec::new();
    for seed in 1..=10 {
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));
        assert_eq!(result.stop_reason, StopReason::Budget);
        choices.push(result.choice);
    }
    assert!(
        choices.iter().all(|choice| *choice == choices[0]),
        "ten seeds returned {choices:?}; the most-visited arm is not settled or the \
         policy is drawing from the rng"
    );
    // Paper is the equilibrium's heaviest action at 10/16, so this is also a
    // check that the visits went where the strategy did.
    assert_eq!(choices[0], 1);
}

/// Four independent trees, merged by `Choice` rather than by arm index.
///
/// The merge cannot pool joint successors — worker 1's arm 2 is not worker 2's
/// arm 2 — so a simultaneous root merges the perspective player's marginals
/// instead, and this is the test that the merged answer means the same thing a
/// single tree's does.
#[cfg(feature = "parallel")]
#[test]
fn root_parallel_merges_marginals() {
    use mcts::RootParallel;

    const THREADS: usize = 4;
    const PER_WORKER: u32 = 50_000;

    let game = BiasedRps::default();
    let cfg = Config {
        simultaneous: SimultaneousConfig {
            root_policy: RootPolicy::MostVisited,
            ..Default::default()
        },
        ..config(PER_WORKER)
    };

    let mut parallel = RootParallel::new(THREADS, &game, |worker| rng(worker as u64 + 1));
    let merged = parallel.search(&game, &(), 0, &cfg, None);

    assert!(
        (0..3).contains(&merged.choice),
        "the merged answer {} is not one of player 0's actions",
        merged.choice
    );
    assert_eq!(
        merged.root_visits,
        PER_WORKER * THREADS as u32,
        "the budget is per worker, so the merged visits are the sum"
    );
    // A merged answer is an aggregate, and an aggregate can be carried: three
    // workers that learned the equilibrium outvote a fourth that learned nothing,
    // and the comparison below would still pass. So each worker is held to a
    // floor of its own — half of what uniform play gives up, the same rule
    // `biased_rps_learns_the_known_equilibrium` uses — and to the mixing that
    // only regret matching produces. Measured at this budget, the four workers
    // sit at 0.0073 / 0.0149 / 0.0154 / 0.0034.
    const WORKER_FLOOR: f64 = 0.04;
    assert!(
        WORKER_FLOOR < biased_rps_exploitability(&[1.0 / 3.0; 3], 0) / 2.0,
        "the per-worker floor {WORKER_FLOOR} is not inside half of uniform play's {}",
        biased_rps_exploitability(&[1.0 / 3.0; 3], 0)
    );

    assert_eq!(parallel.trees().count(), THREADS);
    for (worker, tree) in parallel.trees().enumerate() {
        assert_eq!(tree.simultaneous_players(), Some(PlayerSet::first_n(2)));

        let mut strategy = Vec::new();
        tree.marginals(0)
            .expect("player 0 acts at every worker's root")
            .policy_into(&mut strategy);
        assert_eq!(strategy.len(), 3);
        assert!(
            strategy.iter().all(|weight| *weight > 0.0),
            "worker {worker} extracted {strategy:?}, abandoning an action outright on a \
             game whose equilibrium is mixed"
        );
        let exploitability = biased_rps_exploitability(&strategy, 0);
        assert!(
            exploitability < WORKER_FLOOR,
            "worker {worker} alone plays {strategy:?}, exploitable by {exploitability} \
             against the floor of {WORKER_FLOOR}. The merged answer may still be right; \
             one of the trees it was merged from is not"
        );
    }

    let single = Config {
        iterations: PER_WORKER * THREADS as u32,
        ..cfg
    };
    let mut searcher = Searcher::new(&game);
    let alone = searcher.search(&game, &(), 0, &single, None, &mut rng(1));
    assert_eq!(
        merged.choice, alone.choice,
        "four workers and one tree at the same total budget disagree about the \
         most-selected action"
    );
}

/// The arms of a simultaneous root are the union of every determinization's
/// action list, so the answer read off them has to be masked against the real
/// position — and the merge of four workers' arms is no exception.
///
/// [`ForbiddenFavourite`] makes an unmasked merge fail rather than get lucky:
/// the action the real position withholds is player 0's equilibrium action, so
/// it leads the pooled visits by a wide margin and an unmasked merge returns a
/// move player 0 does not have.
#[cfg(feature = "parallel")]
#[test]
fn a_parallel_simultaneous_root_returns_an_action_the_player_has() {
    use mcts::RootParallel;

    const THREADS: usize = 4;
    const PER_WORKER: u32 = 2_000;

    let game = ForbiddenFavourite::banned();
    let cfg = Config {
        simultaneous: SimultaneousConfig {
            root_policy: RootPolicy::MostVisited,
            ..Default::default()
        },
        ..config(PER_WORKER)
    };

    let mut parallel = RootParallel::new(THREADS, &game, |worker| rng(worker as u64 + 1));
    let merged = parallel.search(&game, &(), 0, &cfg, None);
    assert_eq!(
        parallel.trees().count(),
        THREADS,
        "the merge had fewer trees to filter than workers were asked for"
    );

    let mut pooled = [0u32; 3];
    for tree in parallel.trees() {
        let marginals = tree.marginals(0).expect("player 0 acts at this root");
        for arm in 0..marginals.len() {
            pooled[*marginals.choice(arm) as usize] += marginals.visits(arm);
        }
    }
    assert_eq!(
        pooled
            .iter()
            .enumerate()
            .max_by_key(|&(_, visits)| visits)
            .map(|(action, _)| action as u8),
        Some(FORBIDDEN_ACTION),
        "the workers pooled {pooled:?}, so the action the real position withholds is \
         not the one an unmasked merge would return and this test proves nothing"
    );

    let mut legal = Vec::new();
    game.choices_for_into(&(), 0, &mut legal);
    assert!(
        legal.contains(&merged.choice),
        "the merged answer {} is not one of player 0's actions {legal:?}; it was \
         learned in the determinizations, which hand back the banned action",
        merged.choice
    );
}

/// Early termination must not change the answer — the contract `tests/search.rs`
/// states for a sequential root, held at a simultaneous one.
///
/// Under `Duct` with [`RootPolicy::MostVisited`] the answer is the leading
/// *legal* arm, so a proof about the leading arm of the whole tree is a proof
/// about the answer only if the two are ranked over the same candidates. At
/// [`AbandonedRivals`] they are not: the arm that runs away with the visits is
/// the one the real position withholds, and the two arms the answer is really
/// drawn from are a matching-pennies sub-game the search abandons after a
/// handful of selections, with its leader still swinging when the budget ends.
///
/// That is exactly the shape the surviving proof fires on, which is what makes
/// the mask observable from outside. Ranked over all three arms, the withheld
/// favourite has cleared the evidence bar and both of its rivals are stuck far
/// under it, so the proof reports `Proven` — measured at 1974 of 2000 — about a
/// question the search was never asked. Ranked over the two legal arms alone,
/// neither has cleared the bar, so there is nothing to prove and the search
/// spends its budget. `StopReason::Budget` here *is* the mask.
///
/// Sixteen seeds because the stale answer only differs from the full-budget one
/// when the sub-game swings after the stop, which is about every other seed.
#[test]
fn early_termination_does_not_change_a_simultaneous_answer() {
    const BUDGET: u32 = 2_000;
    // `rank::MIN_EVIDENCE`, which is not public. A rival this far under it
    // cannot reach it with the iterations this search has left.
    const EVIDENCE_BAR: u32 = 32;

    let game = Ducted(AbandonedRivals::banned());
    let patient = Config {
        simultaneous: SimultaneousConfig {
            root_policy: RootPolicy::MostVisited,
            ..Default::default()
        },
        ..config(BUDGET)
    };
    let hasty = Config {
        early_termination: true,
        ..patient
    };

    let mut legal = Vec::new();
    game.choices_for_into(&(), 0, &mut legal);

    for seed in 0..16 {
        let mut searcher = Searcher::new(&game);
        let full = searcher.search(&game, &(), 0, &patient, None, &mut rng(seed));

        let arms = root_arms(&searcher);
        let visits = |want: u8| {
            arms.iter()
                .find(|&&(action, _)| action == want)
                .map_or(0, |&(_, visits)| visits)
        };
        assert!(
            visits(FORBIDDEN_ACTION) >= EVIDENCE_BAR
                && legal
                    .iter()
                    .all(|&action| visits(action) + EVIDENCE_BAR < visits(FORBIDDEN_ACTION)),
            "seed {seed}: an unmasked proof has to be available for this seed to say \
             anything — the withheld action needs the evidence and the legal ones must \
             not have it. Withheld {}, legal {:?}",
            visits(FORBIDDEN_ACTION),
            legal.iter().map(|&a| visits(a)).collect::<Vec<_>>()
        );

        let mut searcher = Searcher::new(&game);
        let stopped = searcher.search(&game, &(), 0, &hasty, None, &mut rng(seed));

        assert_eq!(
            stopped.stop_reason,
            StopReason::Budget,
            "seed {seed}: stopped after {} of {BUDGET} iterations. The answer's own \
             rivals are both under the evidence bar, so nothing about it is provable \
             and the only arm that could have been proved is one the answer is never \
             drawn from",
            stopped.root_visits
        );
        assert_eq!(stopped.root_visits, BUDGET);

        assert!(
            legal.contains(&stopped.choice),
            "seed {seed} returned {}, which is not one of player 0's actions {legal:?}",
            stopped.choice
        );
        assert_eq!(
            stopped.choice, full.choice,
            "seed {seed}: early termination answered {} after {} visits where the full \
             budget answers {} after {}",
            stopped.choice, stopped.root_visits, full.choice, full.root_visits
        );
    }
}

/// The same contract as
/// [`early_termination_does_not_change_a_simultaneous_answer`], held by a
/// `Searcher` that has already searched a different position.
///
/// That is the usage `Searcher`'s own documentation prescribes — one per player
/// per game, `search` called repeatedly — and it is the one the sibling test
/// above cannot see, because it builds a fresh `Searcher` for every seed. The
/// legality mask the proof ranks against is a `Searcher` field that outlives
/// both the search and the tree, and it is cached on arm count alone. Inside
/// one search that is sound: arms are only ever appended. Across searches it
/// guarantees nothing, because a new position can offer the same number of arms
/// as the last one and a different set of them.
///
/// So the warming search here is the *unbanned* [`AbandonedRivals`], whose three
/// arms are all legal, and the search under test is the banned one, whose three
/// arms include the withheld favourite. A mask carried over from the warming
/// search calls all three legal, the proof then ranks the one arm the answer is
/// never drawn from, finds both of its rivals under the evidence bar, and reports
/// `Proven` about a question the search was not asked. A rebuilt mask leaves the
/// proof with two arms that have cleared nothing, and it declines.
#[test]
fn a_reused_searcher_does_not_carry_a_legality_mask_between_searches() {
    const BUDGET: u32 = 2_000;
    // See the sibling test.
    const EVIDENCE_BAR: u32 = 32;

    let warmup = Ducted(AbandonedRivals::default());
    let game = Ducted(AbandonedRivals::banned());
    let patient = Config {
        simultaneous: SimultaneousConfig {
            root_policy: RootPolicy::MostVisited,
            ..Default::default()
        },
        ..config(BUDGET)
    };
    let hasty = Config {
        early_termination: true,
        ..patient
    };

    let mut legal = Vec::new();
    game.choices_for_into(&(), 0, &mut legal);
    assert!(
        !legal.contains(&FORBIDDEN_ACTION),
        "the banned fixture still offers the withheld action"
    );

    for seed in 0..16 {
        let mut cold = Searcher::new(&game);
        let full = cold.search(&game, &(), 0, &patient, None, &mut rng(seed));

        let arms = root_arms(&cold);
        let visits = |want: u8| {
            arms.iter()
                .find(|&&(action, _)| action == want)
                .map_or(0, |&(_, visits)| visits)
        };
        assert!(
            visits(FORBIDDEN_ACTION) >= EVIDENCE_BAR
                && legal
                    .iter()
                    .all(|&action| visits(action) + EVIDENCE_BAR < visits(FORBIDDEN_ACTION)),
            "seed {seed}: a stale all-legal mask has to be able to prove something for \
             this seed to say anything. Withheld {}, legal {:?}",
            visits(FORBIDDEN_ACTION),
            legal.iter().map(|&a| visits(a)).collect::<Vec<_>>()
        );

        let mut warm = Searcher::new(&game);
        let warmed = warm.search(&warmup, &(), 0, &hasty, None, &mut rng(seed));
        assert_eq!(
            root_arms(&warm).len(),
            arms.len(),
            "seed {seed}: the warming search left a different number of arms than the \
             search under test builds, so a stale mask would fail the length check and \
             this test proves nothing"
        );
        assert_eq!(
            warmed.stop_reason,
            StopReason::Proven,
            "seed {seed}: the warming search must leave behind the all-legal mask a \
             proof consulted, or there is no stale mask to carry"
        );

        let stopped = warm.search(&game, &(), 0, &hasty, None, &mut rng(seed));

        assert_eq!(
            stopped.stop_reason,
            StopReason::Budget,
            "seed {seed}: the warmed search stopped after {} of {BUDGET} iterations. \
             Nothing about the answer is provable here, so the mask it ranked against \
             was the warming position's",
            stopped.root_visits
        );
        assert_eq!(stopped.root_visits, BUDGET);

        assert!(
            legal.contains(&stopped.choice),
            "seed {seed} returned {}, which is not one of player 0's actions {legal:?}",
            stopped.choice
        );
        assert_eq!(
            stopped.choice, full.choice,
            "seed {seed}: a warmed searcher answered {} after {} visits where the full \
             budget answers {} after {}",
            stopped.choice, stopped.root_visits, full.choice, full.root_visits
        );
    }
}

/// Player 0's arms at the root of `searcher`'s tree, as `(action, visits)` in
/// arm order.
fn root_arms<G: Game<Choice = u8>>(searcher: &Searcher<G>) -> Vec<(u8, u32)> {
    let marginals = searcher
        .tree()
        .expect("a search leaves a tree")
        .marginals(0)
        .expect("player 0 acts at this root");
    (0..marginals.len())
        .map(|arm| (*marginals.choice(arm), marginals.visits(arm)))
        .collect()
}

/// Spending the budget is not a proof at a `Duct` root either.
///
/// `settled` is asked after the iteration that reaches `target`, so answering
/// "settled" there stamped `Proven` on a search that proved nothing. Only
/// `Duct` was exposed: the mixing branch declines for its own reason, which is
/// why [`early_termination_does_not_change_a_simultaneous_answer`] never
/// reached this.
#[test]
fn a_spent_budget_at_a_duct_root_reports_budget() {
    const BUDGET: u32 = 50;
    let game = Ducted(Pennies::default());
    let cfg = Config {
        early_termination: true,
        ..config(BUDGET)
    };
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(3));

    assert_eq!(
        result.root_visits, BUDGET,
        "the search must reach its budget"
    );
    assert_eq!(result.stop_reason, StopReason::Budget);
}
