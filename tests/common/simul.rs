//! Simultaneous-move fixture games shared by the integration tests.
//!
//! Every fixture here is small enough that its equilibrium can be derived by
//! hand, and each derivation is written down beside the payoffs it comes from.
//! That matters because these fixtures are the yardstick: when a test asserts
//! that the search converges to `(1/16, 10/16, 5/16)`, a reader has to be able
//! to check that the number is a property of the game rather than of whatever
//! the search happened to print on the day the test was written. The equilibria
//! are `pub const`s for the same reason — the tests assert against the
//! arithmetic, not against a remembered number.

// Each integration test binary compiles `tests/common` on its own, so a fixture
// that only one of them uses is dead code in the others.
#![allow(dead_code)]

use mcts::rand_core::Rng;
use mcts::{Game, JointChoices, PlayerSet, Status};

/// Uniform draw from `0..n`, by the same multiply-shift the crate uses, so a
/// rollout here costs one `next_u64` per decision and no modulo bias.
fn below<R: Rng + ?Sized>(rng: &mut R, n: u64) -> u64 {
    ((rng.next_u64() as u128 * n as u128) >> 64) as u64
}

/// Payoff to player 0 at rock-paper-scissors, indexed
/// `[player 0's choice][player 1's choice]`, with rock 0, paper 1, scissors 2.
///
/// Zero-sum on `[0, 1]`: player 1 scores `1.0 - payoff`, so a draw pays both
/// 0.5 and the value of the game is 0.5.
pub const RPS_PAYOFFS: [[f64; 3]; 3] = [[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]];

/// The unique Nash equilibrium of [`Rps`], the same for both players.
///
/// Against a uniform opponent every action is worth exactly 0.5, so uniform is
/// an equilibrium; and any strategy that overweights one action is beaten
/// outright by the action that beats it, so no other profile is one.
pub const RPS_EQUILIBRIUM: [f64; 3] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

/// One simultaneous ply of rock-paper-scissors, and nothing else.
///
/// The smallest game whose equilibrium is mixed, which makes it the smallest
/// game a search that reports a pure strategy gets exploitably wrong.
#[derive(Clone, Default)]
pub struct Rps {
    payoff: Option<f64>,
}

impl Game for Rps {
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
        out.extend([0, 1, 2]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("Rps has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(RPS_PAYOFFS[*joint.get(0) as usize][*joint.get(1) as usize]);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let payoff = *self
            .payoff
            .get_or_insert_with(|| RPS_PAYOFFS[below(rng, 3) as usize][below(rng, 3) as usize]);
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// Payoff to player 0 at the biased rock-paper-scissors of Shafiei, Sturtevant
/// and Schaeffer, rescaled from `[-1, 1]` to `[0, 1]`. Indexed
/// `[player 0's choice][player 1's choice]`, rock 0, paper 1, scissors 2.
///
/// Zero-sum: `payoff[i][j] + payoff[j][i] == 1` for every pair, so player 1
/// scores `1.0 - payoff` and the value is 0.5. The bias is that scissors beats
/// paper by only 0.05 while rock beats scissors by the full 0.5, which pulls
/// the equilibrium a long way off uniform — the whole point of the fixture.
pub const BIASED_RPS_PAYOFFS: [[f64; 3]; 3] =
    [[0.50, 0.25, 1.00], [0.75, 0.50, 0.45], [0.00, 0.55, 0.50]];

/// The unique Nash equilibrium of [`BiasedRps`], the same for both players.
///
/// Derived, not copied. An equilibrium strategy `x` must hold the value 0.5
/// against every column of [`BIASED_RPS_PAYOFFS`]:
///
/// * rock: `0.50 x0 + 0.75 x1 >= 0.5`, i.e. `2 x0 + 3 x1 >= 2`
/// * paper: `0.25 x0 + 0.50 x1 + 0.55 x2 >= 0.5`, i.e. `6 x0 + x1 <= 1`
/// * scissors: `1.00 x0 + 0.45 x1 + 0.50 x2 >= 0.5`, i.e. `x1 <= 10 x0`
///
/// Paper's bound gives `2 x0 + 3 x1 <= 3 - 16 x0`, so rock's needs `x0 <= 1/16`;
/// scissors' gives `2 x0 + 3 x1 <= 32 x0`, so rock's needs `x0 >= 1/16`. Both
/// bounds are therefore tight: `x0 = 1/16`, `x1 = 10/16`, `x2 = 5/16`, and no
/// other strategy is optimal. The game is skew-symmetric about 0.5, so player 1
/// plays the same mixture.
pub const BIASED_RPS_EQUILIBRIUM: [f64; 3] = [1.0 / 16.0, 10.0 / 16.0, 5.0 / 16.0];

/// One simultaneous ply of biased rock-paper-scissors.
///
/// The load-bearing convergence fixture: its equilibrium is mixed, far from
/// uniform, and known exactly, so a policy that settles anywhere else is
/// measurably exploitable rather than merely different.
#[derive(Clone, Default)]
pub struct BiasedRps {
    payoff: Option<f64>,
}

impl Game for BiasedRps {
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
        out.extend([0, 1, 2]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("BiasedRps has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(BIASED_RPS_PAYOFFS[*joint.get(0) as usize][*joint.get(1) as usize]);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let payoff = *self.payoff.get_or_insert_with(|| {
            BIASED_RPS_PAYOFFS[below(rng, 3) as usize][below(rng, 3) as usize]
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

/// Payoff to player 0 at matching pennies, indexed
/// `[player 0's choice][player 1's choice]`. Player 0 wants the coins to match;
/// player 1 scores `1.0 - payoff`.
pub const PENNIES_PAYOFFS: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];

/// The unique Nash equilibrium of [`Pennies`], for both players. Against a
/// uniform opponent both coins are worth 0.5; against anything else the
/// opponent's better reply is pure, so nothing but uniform survives.
pub const PENNIES_EQUILIBRIUM: [f64; 2] = [0.5, 0.5];

/// What [`Pennies`] is worth to either player at equilibrium.
pub const PENNIES_VALUE: f64 = 0.5;

/// Matching pennies as a single simultaneous ply.
///
/// Paired with [`SequentialPennies`], which is the same payoff matrix played
/// one player at a time. The gap between the two values is the entire argument
/// for simultaneous nodes, so the two fixtures must stay payoff-identical.
#[derive(Clone, Default)]
pub struct Pennies {
    payoff: Option<f64>,
}

impl Game for Pennies {
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
        unreachable!("Pennies has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(PENNIES_PAYOFFS[*joint.get(0) as usize][*joint.get(1) as usize]);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let payoff = *self
            .payoff
            .get_or_insert_with(|| PENNIES_PAYOFFS[below(rng, 2) as usize][below(rng, 2) as usize]);
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// What [`SequentialPennies`] is worth to the first mover under best play.
///
/// Zero, not 0.5: the second mover sees the committed coin and mismatches it
/// every time. Serializing a simultaneous node does not approximate its value,
/// it bounds it, and this constant is the size of the error.
pub const SEQUENTIAL_PENNIES_VALUE: f64 = 0.0;

/// The [`Pennies`] payoff matrix modelled as two sequential plies: player 0
/// commits, then player 1 chooses knowing what player 0 played.
///
/// Exists only to be searched alongside [`Pennies`] and get the answer wrong.
#[derive(Clone, Default)]
pub struct SequentialPennies {
    first: Option<u8>,
    second: Option<u8>,
}

impl SequentialPennies {
    fn rewards(&self) -> [f64; 2] {
        let payoff = match (self.first, self.second) {
            (Some(first), Some(second)) => PENNIES_PAYOFFS[first as usize][second as usize],
            _ => unreachable!("both coins are committed before the game is terminal"),
        };
        [payoff, 1.0 - payoff]
    }
}

impl Game for SequentialPennies {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match (self.first, self.second) {
            (None, _) => Status::Active { player: 0 },
            (Some(_), None) => Status::Active { player: 1 },
            (Some(_), Some(_)) => Status::Terminal(self.rewards()),
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend([0, 1]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &u8, _: &mut R) {
        if self.first.is_none() {
            self.first = Some(*choice);
        } else {
            self.second = Some(*choice);
        }
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        while self.second.is_none() {
            let coin = below(rng, 2) as u8;
            self.apply_choice(&(), &coin, rng);
        }
        self.rewards()
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// The unique Nash equilibrium of [`ThreeWayParity`], for every player.
///
/// Player `i`'s payoff depends only on player `i + 1`'s bit, so player `i` is
/// willing to mix only when player `i + 1` is uniform. Suppose some player is
/// not uniform: then their predecessor's best reply is strictly pure, so the
/// predecessor is not uniform either, and walking the cycle backwards makes all
/// three pure. But a pure profile needs `b[i] != b[i + 1]` for all three `i` at
/// once, which two-colours an odd cycle and is impossible — so every player is
/// uniform, and that profile is an equilibrium because it leaves everyone
/// indifferent.
pub const THREE_WAY_PARITY_EQUILIBRIUM: [f64; 2] = [0.5, 0.5];

/// Three players each pick a bit at the same time; player `i` scores 1.0 when
/// their bit differs from player `i + 1`'s, cyclically, and 0.0 otherwise.
///
/// A three-player node with a unique fully mixed equilibrium, which is what the
/// arity-3 tests need: nothing but genuine mixing at all three slots reaches
/// it. See [`THREE_WAY_PARITY_EQUILIBRIUM`] for the derivation.
///
/// The payoff is a cycle of pairwise parities rather than each player's bit
/// against the parity of the other two, because the latter is the same number
/// for all three players — everyone scores 1.0 exactly when the three bits have
/// odd parity. That makes it a pure coordination game with four strict pure
/// equilibria, each paying every player 1.0 against the uniform profile's 0.5,
/// and three independent learners find one of them and stay there. The cycle
/// keeps the game symmetric and keeps the uniform profile, but makes it the
/// only equilibrium there is.
#[derive(Clone, Default)]
pub struct ThreeWayParity {
    bits: Option<[u8; 3]>,
}

impl ThreeWayParity {
    /// Every player's payoff for a bit triple.
    pub fn payoffs(bits: [u8; 3]) -> [f64; 3] {
        let mut rewards = [0.0; 3];
        for player in 0..3 {
            rewards[player] = f64::from(u8::from(bits[player] != bits[(player + 1) % 3]));
        }
        rewards
    }
}

impl Game for ThreeWayParity {
    type Choice = u8;
    type Rewards = [f64; 3];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 3]> {
        match self.bits {
            Some(bits) => Status::Terminal(Self::payoffs(bits)),
            None => Status::Simultaneous {
                players: PlayerSet::first_n(3),
            },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend([0, 1]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("ThreeWayParity has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.bits = Some([*joint.get(0), *joint.get(1), *joint.get(2)]);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 3] {
        let bits = *self.bits.get_or_insert_with(|| {
            [
                below(rng, 2) as u8,
                below(rng, 2) as u8,
                below(rng, 2) as u8,
            ]
        });
        Self::payoffs(bits)
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// The branch of [`SeqThenSimul`] that leads to the simultaneous node.
pub const BRANCH_A: u8 = 0;

/// The branch of [`SeqThenSimul`] that pays out immediately.
pub const BRANCH_B: u8 = 1;

/// What [`BRANCH_B`] pays player 0. Below [`PENNIES_VALUE`], so a search that
/// values the simultaneous branch correctly prefers [`BRANCH_A`], and a search
/// that serializes it — valuing it at [`SEQUENTIAL_PENNIES_VALUE`] — prefers
/// [`BRANCH_B`].
pub const BRANCH_B_VALUE: f64 = 0.25;

/// A sequential root above a simultaneous node: player 0 picks [`BRANCH_A`],
/// which leads to a [`Pennies`] ply worth [`PENNIES_VALUE`], or [`BRANCH_B`],
/// which pays [`BRANCH_B_VALUE`] outright.
///
/// The mixed-tree fixture. Descent, the simultaneous frame stack, the backup
/// cursor and tree reuse all have to handle both node kinds in one tree, and a
/// bug in any of them shows up here as the search preferring the branch worth
/// half as much.
#[derive(Clone, Default)]
pub struct SeqThenSimul {
    branch: Option<u8>,
    payoff: Option<f64>,
}

impl SeqThenSimul {
    /// The position one ply in, after player 0 has taken [`BRANCH_A`]: a bare
    /// simultaneous node, which is what the tree-reuse tests re-root onto.
    pub fn after_branch_a() -> Self {
        Self {
            branch: Some(BRANCH_A),
            payoff: None,
        }
    }
}

impl Game for SeqThenSimul {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match (self.branch, self.payoff) {
            (None, _) => Status::Active { player: 0 },
            (Some(_), Some(payoff)) => Status::Terminal([payoff, 1.0 - payoff]),
            (Some(_), None) => Status::Simultaneous {
                players: PlayerSet::first_n(2),
            },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        match self.branch {
            None => out.extend([BRANCH_A, BRANCH_B]),
            Some(_) => out.extend([0, 1]),
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &u8, _: &mut R) {
        self.branch = Some(*choice);
        if *choice == BRANCH_B {
            self.payoff = Some(BRANCH_B_VALUE);
        }
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(PENNIES_PAYOFFS[*joint.get(0) as usize][*joint.get(1) as usize]);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        if self.branch.is_none() {
            let branch = below(rng, 2) as u8;
            self.apply_choice(&(), &branch, rng);
        }
        let payoff = *self
            .payoff
            .get_or_insert_with(|| PENNIES_PAYOFFS[below(rng, 2) as usize][below(rng, 2) as usize]);
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// Player 0's three bids at [`AsymmetricBid`].
pub const BID_ACTIONS: [u8; 3] = [0, 1, 2];

/// Player 1's two replies at [`AsymmetricBid`]: 3 passes, 4 calls.
///
/// Deliberately disjoint from [`BID_ACTIONS`], so a test can tell by the value
/// alone which player an arm belongs to — the failure this fixture guards
/// against is a slot being handed the other player's action list.
pub const RESPONSE_ACTIONS: [u8; 2] = [3, 4];

/// Payoff to player 0 at [`AsymmetricBid`], indexed `[bid][reply - 3]`.
/// Zero-sum on `[0, 1]`: player 1 scores `1.0 - payoff`.
pub const ASYMMETRIC_BID_PAYOFFS: [[f64; 2]; 3] = [[0.4, 0.5], [0.6, 0.4], [0.8, 0.1]];

/// Player 0's unique equilibrium mixture at [`AsymmetricBid`], over
/// [`BID_ACTIONS`]. The high bid is never played: it wins big against a pass
/// but loses too much to a call, and player 1 has only two replies, so at most
/// two of player 0's bids can be held indifferent.
pub const ASYMMETRIC_BID_P0_EQUILIBRIUM: [f64; 3] = [2.0 / 3.0, 1.0 / 3.0, 0.0];

/// Player 1's unique equilibrium mixture at [`AsymmetricBid`], over
/// [`RESPONSE_ACTIONS`].
pub const ASYMMETRIC_BID_P1_EQUILIBRIUM: [f64; 2] = [1.0 / 3.0, 2.0 / 3.0];

/// What [`AsymmetricBid`] is worth to player 0 at equilibrium.
///
/// Both mixtures check out by hand: against player 1's `(1/3, 2/3)` the bids
/// are worth `0.4/3 + 1.0/3 = 7/15`, `0.6/3 + 0.8/3 = 7/15` and
/// `0.8/3 + 0.2/3 = 1/3`, so player 0 is indifferent between the two bids they
/// play and strictly prefers them to the third; against player 0's
/// `(2/3, 1/3, 0)` both replies concede `7/15`, so player 1 is indifferent too.
pub const ASYMMETRIC_BID_VALUE: f64 = 7.0 / 15.0;

/// A simultaneous node where the two players own different action lists:
/// player 0 picks one of three bids, player 1 either passes or calls.
///
/// The only fixture that overrides [`Game::choices_for_into`], and so the only
/// one that exercises unequal slot lengths in the marginal arrays, the joint
/// key packing and the root extraction.
#[derive(Clone, Default)]
pub struct AsymmetricBid {
    payoff: Option<f64>,
}

impl AsymmetricBid {
    /// Payoff to player 0 for one bid and one reply.
    pub fn payoff(bid: u8, reply: u8) -> f64 {
        ASYMMETRIC_BID_PAYOFFS[bid as usize][(reply - RESPONSE_ACTIONS[0]) as usize]
    }
}

impl Game for AsymmetricBid {
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

    fn choices_into(&self, _: &(), _: &mut Vec<u8>) {
        unreachable!("AsymmetricBid names each player's actions through choices_for_into")
    }

    fn choices_for_into(&self, _: &(), player: u8, out: &mut Vec<u8>) {
        match player {
            0 => out.extend(BID_ACTIONS),
            _ => out.extend(RESPONSE_ACTIONS),
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("AsymmetricBid has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(Self::payoff(*joint.get(0), *joint.get(1)));
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let payoff = *self.payoff.get_or_insert_with(|| {
            let bid = BID_ACTIONS[below(rng, BID_ACTIONS.len() as u64) as usize];
            let reply = RESPONSE_ACTIONS[below(rng, RESPONSE_ACTIONS.len() as u64) as usize];
            Self::payoff(bid, reply)
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

/// How often a determinization of [`VariableRps`] leaves scissors legal for
/// player 1.
pub const VARIABLE_RPS_SCISSORS_RATE: f64 = 0.5;

/// [`BiasedRps`] in which every determinization takes scissors away from
/// player 1 with probability one half.
///
/// The only fixture whose `determinize_into` is not a `clone_from`, and the
/// only one that makes an arm's availability differ from its node's visits.
/// Scored on visits alone, player 1's scissors is punished for the iterations
/// it was not allowed to play in, its regret runs away, and it starves; the
/// availability correction is what keeps it alive, so this fixture is the
/// direct test of that correction.
#[derive(Clone, Default)]
pub struct VariableRps {
    payoff: Option<f64>,
    scissors_banned: bool,
}

impl Game for VariableRps {
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
        out.extend([0, 1, 2]);
    }

    fn choices_for_into(&self, _: &(), player: u8, out: &mut Vec<u8>) {
        out.extend([0, 1]);
        if player == 0 || !self.scissors_banned {
            out.push(2);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("VariableRps has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(BIASED_RPS_PAYOFFS[*joint.get(0) as usize][*joint.get(1) as usize]);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let banned = self.scissors_banned;
        let payoff = *self.payoff.get_or_insert_with(|| {
            let mine = below(rng, 3) as usize;
            let theirs = below(rng, if banned { 2 } else { 3 }) as usize;
            BIASED_RPS_PAYOFFS[mine][theirs]
        });
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.scissors_banned = rng.next_u64() & 1 == 1;
    }
}

/// How many actions each player has at [`ManyArm`].
///
/// Above the default `Game::CHILD_INDEX_THRESHOLD` of 16, so both the arm index
/// and the joint-child index are built rather than scanned — which is the point
/// of the fixture, since the scan and the index are separate code paths and only
/// the scan is exercised by every other fixture here.
pub const MANY_ARM_ACTIONS: u8 = 40;

/// A simultaneous node with [`MANY_ARM_ACTIONS`] actions per player: a cyclic
/// generalization of rock-paper-scissors, zero-sum on `[0, 1]`, whose unique
/// equilibrium is uniform for both players by symmetry.
#[derive(Clone, Default)]
pub struct ManyArm {
    payoff: Option<f64>,
}

impl ManyArm {
    /// Payoff to player 0 when the two players pick `a` and `b`.
    ///
    /// An action beats the 19 actions above it modulo 40 and loses to the 19
    /// below, drawing with itself and with the action directly opposite — which
    /// has to be a draw, since the two are the same distance apart in both
    /// directions and anything else would break the zero sum.
    pub fn payoff(a: u8, b: u8) -> f64 {
        let n = MANY_ARM_ACTIONS;
        let distance = (a + n - b) % n;
        if distance == 0 || distance == n / 2 {
            0.5
        } else if distance < n / 2 {
            1.0
        } else {
            0.0
        }
    }
}

impl Game for ManyArm {
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
        out.extend(0..MANY_ARM_ACTIONS);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("ManyArm has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(Self::payoff(*joint.get(0), *joint.get(1)));
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let arms = u64::from(MANY_ARM_ACTIONS);
        let payoff = *self
            .payoff
            .get_or_insert_with(|| Self::payoff(below(rng, arms) as u8, below(rng, arms) as u8));
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// Player 0's action that [`ForbiddenFavourite`] takes away in the real
/// position, and hands back in every determinization of it.
pub const FORBIDDEN_ACTION: u8 = 2;

/// Payoff to player 0 at [`AbandonedRivals`], indexed
/// `[player 0's action][player 1's action]`. Player 1 scores `1.0 - payoff`.
///
/// [`FORBIDDEN_ACTION`] pays a flat 1.0 and the two actions player 0 keeps are
/// matching pennies worth 0.10, so decoupled UCB1 stops returning to the kept
/// rows after a handful of selections — `(c / gap)^2 * ln(t)` of them at
/// `c = 0.75` and `gap = 0.9`, which is single digits at any budget this test
/// suite uses. That is the shape the surviving early-termination proof is about:
/// the withheld arm is the only one with enough evidence behind it to be
/// trusted, and the two the answer is actually drawn from are locked under the
/// bar with the answer still swinging between them.
pub const ABANDONED_RIVALS_PAYOFFS: [[f64; 2]; 3] = [[0.05, 0.15], [0.15, 0.05], [1.0, 1.0]];

/// [`ForbiddenFavourite`] with the gap widened until the kept actions are
/// abandoned.
///
/// The sibling fixture keeps its two legal rows in play deliberately, which is
/// what makes it useless for the early-termination proof that survives the root
/// ranking change: that proof fires only when every rival of the answer is too
/// thinly sampled to be trusted and cannot be sampled enough with the iterations
/// left. Here the withheld action is worth 1.0 against the kept rows' 0.10, so
/// the kept rows *are* that thinly sampled — and a consumer that forgets to mask
/// the arms against the real position ranks the withheld action first, finds
/// both of its rivals stuck under the evidence bar, and reports
/// `StopReason::Proven` about a question the search was never asked. A masked
/// consumer sees only the two kept arms, neither of which has cleared the bar,
/// and refuses.
#[derive(Clone, Default)]
pub struct AbandonedRivals {
    payoff: Option<f64>,
    banned: bool,
}

impl AbandonedRivals {
    /// The position player 0 is actually in: [`FORBIDDEN_ACTION`] is not
    /// available, though every determinization says otherwise.
    pub fn banned() -> Self {
        Self {
            payoff: None,
            banned: true,
        }
    }
}

impl Game for AbandonedRivals {
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

    fn choices_into(&self, _: &(), _: &mut Vec<u8>) {
        unreachable!("AbandonedRivals names each player's actions through choices_for_into")
    }

    fn choices_for_into(&self, _: &(), player: u8, out: &mut Vec<u8>) {
        out.extend([0, 1]);
        if player == 0 && !self.banned {
            out.push(FORBIDDEN_ACTION);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("AbandonedRivals has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff =
            Some(ABANDONED_RIVALS_PAYOFFS[*joint.get(0) as usize][*joint.get(1) as usize]);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let banned = self.banned;
        let payoff = *self.payoff.get_or_insert_with(|| {
            let mine = below(rng, if banned { 2 } else { 3 }) as usize;
            ABANDONED_RIVALS_PAYOFFS[mine][below(rng, 2) as usize]
        });
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
        dest.banned = false;
    }
}

/// Payoff to player 0 at [`ForbiddenFavourite`], indexed
/// `[player 0's action][player 1's action]`. Player 1 scores `1.0 - payoff`.
///
/// The two actions player 0 keeps are matching pennies at stakes of 0.25: that
/// sub-game's equilibrium is mixed and worth 0.5 to either player, so which of
/// the two leads on visits keeps changing for as long as the search runs.
/// [`FORBIDDEN_ACTION`] pays a flat 0.55, which is what makes it player 0's
/// equilibrium action — against player 1 playing the second column with
/// probability `q` the kept rows are worth `0.25 + 0.5q` and `0.75 - 0.5q`, so
/// 0.55 is at least as good as both exactly when `q` lies in `[0.4, 0.6]`, and a
/// flat row leaves player 1 indifferent over every such `q`. It clears the
/// sub-game's 0.5 by little enough that the kept rows keep being explored rather
/// than starving.
pub const FORBIDDEN_FAVOURITE_PAYOFFS: [[f64; 2]; 3] = [[0.25, 0.75], [0.75, 0.25], [0.55, 0.55]];

/// A simultaneous ply whose equilibrium action for player 0 is illegal in the
/// real position and legal in every determinization of it.
///
/// Imperfect information does exactly this, and [`VariableRps`] is the mild
/// version of it: arms accumulate every action any sampled world offered, so the
/// marginal carrying the most visits can be one the player cannot play here.
/// Every consumer of the arms — the root extraction, the root-parallel merge,
/// the early-termination proof — has to mask them against the real position, and
/// this fixture is what tells a masked consumer from an unmasked one, because
/// the unplayable action is precisely the one an unmasked consumer picks.
#[derive(Clone, Default)]
pub struct ForbiddenFavourite {
    payoff: Option<f64>,
    banned: bool,
}

impl ForbiddenFavourite {
    /// The position player 0 is actually in: [`FORBIDDEN_ACTION`] is not
    /// available, though every determinization says otherwise.
    pub fn banned() -> Self {
        Self {
            payoff: None,
            banned: true,
        }
    }
}

impl Game for ForbiddenFavourite {
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

    fn choices_into(&self, _: &(), _: &mut Vec<u8>) {
        unreachable!("ForbiddenFavourite names each player's actions through choices_for_into")
    }

    fn choices_for_into(&self, _: &(), player: u8, out: &mut Vec<u8>) {
        out.extend([0, 1]);
        if player == 0 && !self.banned {
            out.push(FORBIDDEN_ACTION);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("ForbiddenFavourite has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff =
            Some(FORBIDDEN_FAVOURITE_PAYOFFS[*joint.get(0) as usize][*joint.get(1) as usize]);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let banned = self.banned;
        let payoff = *self.payoff.get_or_insert_with(|| {
            let mine = below(rng, if banned { 2 } else { 3 }) as usize;
            FORBIDDEN_FAVOURITE_PAYOFFS[mine][below(rng, 2) as usize]
        });
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
        dest.banned = false;
    }
}
