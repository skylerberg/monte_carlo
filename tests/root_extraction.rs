//! How the answer is read off a finished root: which candidates are eligible,
//! and how they are ranked against each other.

use mcts::rand_core::{Rng, SeedableRng};
use mcts::{
    Config, Game, JointChoices, PlayerSet, RootPolicy, Searcher, SimultaneousPolicy, Status,
};
use wyrand::WyRand;

fn rng(seed: u64) -> WyRand {
    WyRand::seed_from_u64(seed)
}

/// Uniform draw from `0..n`, by the multiply-shift the crate uses internally.
fn below<R: Rng + ?Sized>(rng: &mut R, n: u64) -> u64 {
    ((rng.next_u64() as u128 * n as u128) >> 64) as u64
}

/// A one-ply root whose second choice is only dealt to a third of the
/// determinizations, and pays better than the first wherever it is.
///
/// The real position holds both. Nothing here is noisy: each choice leads
/// straight to a terminal state paying a fixed amount, so a search that returns
/// choice 0 is not unlucky, it is ranking by how often the choice was legal.
#[derive(Clone)]
struct RareBest {
    played: Option<usize>,
    rare_legal: bool,
}

impl RareBest {
    const PAYOFFS: [f64; 2] = [0.5, 0.9];

    fn new() -> Self {
        Self {
            played: None,
            rare_legal: true,
        }
    }
}

impl Game for RareBest {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.played {
            Some(choice) => {
                let payoff = Self::PAYOFFS[choice];
                Status::Terminal([payoff, 1.0 - payoff])
            }
            None => Status::Active { player: 0 },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.push(0);
        if self.rare_legal {
            out.push(1);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, _: &mut R) {
        self.played = Some(*choice);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        match self.status(&()) {
            Status::Terminal(rewards) => rewards,
            _ => unreachable!("every choice ends the game"),
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.rare_legal = below(rng, 3) == 0;
    }
}

/// A move a third of the worlds offer, played on nearly every iteration that
/// offered it, must not lose the root to a move played on two thirds of the
/// iterations because it was legal in all of them. Raw visits decide that
/// contest by legality rate, and no budget closes the gap — it is structural.
/// The mean reward decides it by value, which is what the two moves differ in.
#[test]
fn a_rarely_legal_but_better_move_wins_the_root() {
    const BUDGET: u32 = 3_000;

    for seed in [1u64, 7, 21, 99] {
        let game = RareBest::new();
        let cfg = Config {
            iterations: BUDGET,
            ..Config::default()
        };
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));

        let visits: Vec<u32> = searcher
            .tree()
            .expect("a search leaves a tree")
            .children()
            .iter()
            .map(|child| child.visits())
            .collect();
        assert!(
            visits[0] > visits[1],
            "seed {seed}: the always-legal move must lead on raw visits, or this \
             fixture tests nothing: {visits:?}"
        );
        assert_eq!(
            result.choice, 1,
            "seed {seed}: returned the dominated move with visits {visits:?}"
        );
        assert!(
            (result.best_mean_reward - 0.9).abs() < 1e-9,
            "seed {seed}: reported {} for a move worth 0.9",
            result.best_mean_reward
        );
    }
}

/// The same shape as [`RareBest`] with the rare choice as the *worst* of the
/// three: two solid always-legal moves, and a decoy one determinization in
/// `rarity` offers, worth `decoy`.
///
/// The real position holds all three, so nothing filters the decoy out; it has
/// to lose on the ranking or not at all.
#[derive(Clone)]
struct RareDecoy {
    played: Option<usize>,
    rare_legal: bool,
    rarity: u64,
    decoy: f64,
}

impl RareDecoy {
    const BEST: f64 = 0.90;
    /// The bands the rate-ranked design was broken on: rare enough that the
    /// decoy is taken on nearly every iteration that offers it, common enough
    /// that it clears the evidence bar comfortably.
    const RARITY_BAND: &'static [(u64, f64)] = &[(100, 0.75), (250, 0.60), (2_000, 0.50)];

    fn new(rarity: u64, decoy: f64) -> Self {
        Self {
            played: None,
            rare_legal: true,
            rarity,
            decoy,
        }
    }

    fn payoff(&self, choice: usize) -> f64 {
        match choice {
            0 => Self::BEST,
            1 => 0.85,
            _ => self.decoy,
        }
    }
}

impl Game for RareDecoy {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.played {
            Some(choice) => {
                let payoff = self.payoff(choice);
                Status::Terminal([payoff, 1.0 - payoff])
            }
            None => Status::Active { player: 0 },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend([0, 1]);
        if self.rare_legal {
            out.push(2);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, _: &mut R) {
        self.played = Some(*choice);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        match self.status(&()) {
            Status::Terminal(rewards) => rewards,
            _ => unreachable!("every choice ends the game"),
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.rare_legal = below(rng, self.rarity) == 0;
    }
}

/// The constraint that sits opposite [`a_rarely_legal_but_better_move_wins_the_root`],
/// and the reason the ranking is not the selection rate.
///
/// Selection takes an unvisited candidate on sight and keeps taking it while
/// its exploration bonus covers the gap to the leader, so a candidate offered a
/// hundred times is taken on essentially all hundred *whatever it is worth*.
/// Its rate is then 1.000 — the ceiling — off enough selections to clear any
/// evidence bar, while the genuinely best move splits the root's budget with a
/// rival and comes in at 0.987. A rate-ranked root answers the decoy in every
/// seed, stamped `Proven`. The two fixtures are the same shape with opposite
/// correct verdicts, and what separates them is the value the search measured,
/// which is the statistic the ranking leads with.
///
/// The rarity band matters: [`RareDecoy::RARITY_BAND`] is where the decoy is
/// rare enough to saturate its rate and common enough to be well over the bar,
/// which is exactly where a bar alone cannot help.
#[test]
fn a_rarely_legal_but_worse_move_does_not_win_the_root() {
    const BUDGET: u32 = 10_000;

    for &(rarity, decoy) in RareDecoy::RARITY_BAND {
        for seed in [1u64, 7, 21, 99] {
            let game = RareDecoy::new(rarity, decoy);
            let cfg = Config {
                iterations: BUDGET,
                ..Config::default()
            };
            let mut searcher = Searcher::new(&game);
            let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));

            let root = searcher.tree().expect("a search leaves a tree");
            let stats: Vec<(u32, u32)> = root
                .children()
                .iter()
                .map(|child| (child.visits(), child.availability()))
                .collect();
            let rate = |i: usize| stats[i].0 as f64 / stats[i].1.max(1) as f64;
            assert!(
                rate(2) > rate(0),
                "1/{rarity}, seed {seed}: the decoy must hold the higher selection \
                 rate, or this fixture tests nothing: {stats:?}"
            );
            assert_eq!(
                result.choice,
                0,
                "1/{rarity}, seed {seed}: answered a move worth {} over one worth \
                 {}, on {stats:?}",
                game.payoff(result.choice),
                RareDecoy::BEST
            );
            assert!(
                (result.best_mean_reward - RareDecoy::BEST).abs() < 1e-9,
                "1/{rarity}, seed {seed}: reported {} for a move worth {}",
                result.best_mean_reward,
                RareDecoy::BEST
            );
        }
    }
}

/// A one-ply root where three quarters of the determinizations offer a third
/// choice the real position does not, and it is the best of the three.
#[derive(Clone)]
struct PhantomBest {
    played: Option<usize>,
    phantom_legal: bool,
}

impl PhantomBest {
    const PAYOFFS: [f64; 3] = [0.4, 0.5, 1.0];

    fn new() -> Self {
        Self {
            played: None,
            phantom_legal: false,
        }
    }
}

impl Game for PhantomBest {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.played {
            Some(choice) => {
                let payoff = Self::PAYOFFS[choice];
                Status::Terminal([payoff, 1.0 - payoff])
            }
            None => Status::Active { player: 0 },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend([0, 1]);
        if self.phantom_legal {
            out.push(2);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, _: &mut R) {
        self.played = Some(*choice);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        match self.status(&()) {
            Status::Terminal(rewards) => rewards,
            _ => unreachable!("every choice ends the game"),
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.phantom_legal = below(rng, 4) != 0;
    }
}

/// The tree holds the union of every determinization's choices, and the answer
/// has to be a move the player actually holds. Returning the union's leader
/// hands the caller a move `apply_choice` was never going to accept — and with
/// early termination on, hands it back stamped `Proven`.
#[test]
fn the_root_never_answers_a_move_the_position_withholds() {
    const BUDGET: u32 = 3_000;

    for seed in [1u64, 7, 21, 99] {
        let game = PhantomBest::new();
        let cfg = Config {
            iterations: BUDGET,
            ..Config::default()
        };
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));

        let root = searcher.tree().expect("a search leaves a tree");
        let phantom = root
            .children()
            .iter()
            .find(|child| *child.edge().choice().expect("a choice") == 2)
            .expect("the tree must learn about the phantom choice");
        let held: u32 = root
            .children()
            .iter()
            .filter(|child| *child.edge().choice().expect("a choice") != 2)
            .map(|child| child.visits())
            .max()
            .expect("the root has children");
        assert!(
            phantom.visits() > held,
            "seed {seed}: the phantom move must lead the tree, or this fixture tests \
             nothing: {} against {held}",
            phantom.visits()
        );
        assert_eq!(
            result.choice, 1,
            "seed {seed}: answered {} where the position offers only 0 and 1",
            result.choice
        );
    }
}

/// A one-ply root of [`WideTerminal::WIDTH`] choices, exactly one of which
/// wins, each leading straight to a terminal state.
#[derive(Clone, Default)]
struct WideTerminal {
    played: Option<usize>,
}

impl WideTerminal {
    const WIDTH: usize = 64;
    const WINNER: usize = 10;
}

impl Game for WideTerminal {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.played {
            Some(choice) if choice == Self::WINNER => Status::Terminal([1.0, 0.0]),
            Some(_) => Status::Terminal([0.0, 1.0]),
            None => Status::Active { player: 0 },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend(0..Self::WIDTH);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, _: &mut R) {
        self.played = Some(*choice);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        match self.status(&()) {
            Status::Terminal(rewards) => rewards,
            _ => unreachable!("every choice ends the game"),
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// A root wider than the iteration budget opens every child on iteration one
/// and then visits them one at a time, so every child the search reached ends
/// on exactly one visit and a visit argmax returns child zero whatever it
/// measured there. The estimates here are exact terminal payoffs, not noise.
#[test]
fn a_root_wider_than_the_budget_returns_the_best_thing_it_measured() {
    const BUDGET: u32 = 40;

    let game = WideTerminal::default();
    let cfg = Config {
        iterations: BUDGET,
        early_termination: false,
        ..Config::default()
    };

    for seed in [1u64, 7, 21] {
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));

        let root = searcher.tree().expect("a search leaves a tree");
        assert!(
            root.children().iter().all(|child| child.visits() <= 1),
            "seed {seed}: the budget must be spent opening children, or this fixture \
             tests nothing"
        );
        assert_eq!(
            result.choice,
            WideTerminal::WINNER,
            "seed {seed}: returned {} with mean {}, where {} was measured at 1.0",
            result.choice,
            result.best_mean_reward,
            WideTerminal::WINNER
        );
    }
}

/// A simultaneous root where player 0's second action is dealt to a quarter of
/// the determinizations and strictly dominates the first wherever it is.
///
/// Player 1's action never matters, so this isolates the marginal extraction:
/// decoupled UCB1 plays action 1 on essentially every iteration that offers it,
/// and it still ends with a quarter of action 0's raw visits.
#[derive(Clone, Default)]
struct RareBestSimul {
    payoff: Option<f64>,
    rare_legal: bool,
}

impl RareBestSimul {
    fn new() -> Self {
        Self {
            payoff: None,
            rare_legal: true,
        }
    }
}

impl Game for RareBestSimul {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    const SIMULTANEOUS_POLICY: SimultaneousPolicy = SimultaneousPolicy::Duct;

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

    fn choices_for_into(&self, _: &(), player: u8, out: &mut Vec<u8>) {
        out.push(0);
        if player == 1 || self.rare_legal {
            out.push(1);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("RareBestSimul has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(if *joint.get(0) == 1 { 0.9 } else { 0.4 });
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        let payoff = self.payoff.expect("every joint action ends the game");
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.rare_legal = below(rng, 4) == 0;
    }
}

/// The simultaneous half of the same rule, where the loss is sharper: `Duct`'s
/// *selection* is availability-corrected and had already identified the
/// dominant arm, so a raw-visit extraction throws away an answer the search
/// itself had right.
#[test]
fn a_rarely_legal_but_better_arm_wins_the_simultaneous_root() {
    const BUDGET: u32 = 4_000;

    for seed in [1u64, 7, 21] {
        let game = RareBestSimul::new();
        let cfg = Config {
            iterations: BUDGET,
            simultaneous: mcts::SimultaneousConfig {
                root_policy: RootPolicy::MostVisited,
                ..Default::default()
            },
            ..Config::default()
        };
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));

        let root = searcher.tree().expect("a search leaves a tree");
        let marginals = root.marginals(0).expect("player 0 acts here");
        assert!(
            marginals.visits(0) > marginals.visits(1),
            "seed {seed}: the always-legal arm must lead on raw visits, or this \
             fixture tests nothing: {} against {}",
            marginals.visits(0),
            marginals.visits(1)
        );
        assert_eq!(
            result.choice,
            1,
            "seed {seed}: returned the dominated arm, visits {} / availability {} \
             against {} / {}",
            marginals.visits(0),
            marginals.availability(0),
            marginals.visits(1),
            marginals.availability(1)
        );
    }
}

/// A simultaneous root where player 0's two actions depend on a hidden value, so
/// a determinization's world often names a pair of actions the real position
/// does not hold at all.
#[derive(Clone, Default)]
struct HiddenHand {
    payoff: Option<f64>,
    hand: u8,
}

impl HiddenHand {
    const HANDS: u64 = 4;
}

impl Game for HiddenHand {
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

    fn choices_for_into(&self, _: &(), player: u8, out: &mut Vec<u8>) {
        if player == 0 {
            out.extend([self.hand * 2, self.hand * 2 + 1]);
        } else {
            out.extend([0, 1]);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("HiddenHand has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.payoff = Some(if joint.get(0) % 2 == *joint.get(1) {
            1.0
        } else {
            0.0
        });
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        let payoff = self.payoff.expect("every joint action ends the game");
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.hand = below(rng, Self::HANDS) as u8;
    }
}

/// `root_policy_into` promises pairs summing to 1 over the actions legal in the
/// position whenever it answers true. A one-iteration search whose single world
/// dealt a different hand leaves a root whose arms are disjoint from this
/// position's legal set, and the extraction has nothing of the search's to
/// report — which is a real state at low budgets, not an error, and a caller
/// taking the argmax of an empty policy cannot tell it from a healthy call.
#[test]
fn the_root_policy_is_never_reported_empty() {
    let game = HiddenHand::default();
    let legal = [0u8, 1];

    let mut answered = 0;
    let mut disjoint = 0;
    for seed in 0..64u64 {
        let cfg = Config {
            iterations: 1,
            ..Config::default()
        };
        let mut searcher = Searcher::new(&game);
        searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));

        let root = searcher.tree().expect("a search leaves a tree");
        let marginals = root.marginals(0).expect("player 0 acts here");
        if (0..marginals.len()).all(|arm| !legal.contains(marginals.choice(arm))) {
            disjoint += 1;
        }

        let mut out = Vec::new();
        if !searcher.root_policy_into(&game, &(), 0, &mut out) {
            continue;
        }
        answered += 1;
        assert!(!out.is_empty(), "seed {seed}: answered true with no policy");
        let sum: f64 = out.iter().map(|&(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "seed {seed}: {out:?} sums to {sum}"
        );
        for &(choice, _) in &out {
            assert!(
                legal.contains(&choice),
                "seed {seed}: reported action {choice}, which this position withholds"
            );
        }
    }
    assert!(
        answered > 0,
        "no seed answered, so this test compares nothing"
    );
    assert!(
        disjoint > 0,
        "no world was disjoint from the real position, so this test compares nothing"
    );
}

/// A pool is a pool *of searchers*, and it merges what they hold. Both of the
/// corrections the single-threaded root applies have to survive that merge, or
/// one position has two answers depending on how many threads asked.
#[cfg(feature = "parallel")]
#[test]
fn a_pooled_root_answers_what_its_searchers_would() {
    use mcts::RootParallel;

    const BUDGET: u32 = 3_000;
    let cfg = Config {
        iterations: BUDGET,
        ..Config::default()
    };

    for threads in [1usize, 3] {
        let rare = RareBest::new();
        let mut pool = RootParallel::new(threads, &rare, |worker| rng(worker as u64 + 1));
        let result = pool.search(&rare, &(), 0, &cfg, None);
        assert_eq!(
            result.choice, 1,
            "{threads} workers returned the dominated move"
        );

        let phantom = PhantomBest::new();
        let mut pool = RootParallel::new(threads, &phantom, |worker| rng(worker as u64 + 1));
        let result = pool.search(&phantom, &(), 0, &cfg, None);
        assert_eq!(
            result.choice, 1,
            "{threads} workers answered a move the position withholds"
        );
    }
}

/// A root of [`WideDecoy::WIDTH`] choices where one solid move is clearly best,
/// one decoy is nearly as good and offered by one determinization in
/// [`WideDecoy::RARITY`], and the rest are poor.
///
/// The decoy is dominated wherever both are legal, and the real position offers
/// it, so nothing filters it out — it has to lose on the ranking or not at all.
#[derive(Clone)]
struct WideDecoy {
    played: Option<usize>,
    decoy_legal: bool,
}

impl WideDecoy {
    const WIDTH: usize = 64;
    const BEST: usize = 7;
    const DECOY: usize = 63;
    const RARITY: u64 = 500;

    fn new() -> Self {
        Self {
            played: None,
            decoy_legal: true,
        }
    }

    fn payoff(choice: usize) -> f64 {
        match choice {
            Self::BEST => 0.90,
            Self::DECOY => 0.85,
            _ => 0.40,
        }
    }
}

impl Game for WideDecoy {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.played {
            Some(choice) => {
                let payoff = Self::payoff(choice);
                Status::Terminal([payoff, 1.0 - payoff])
            }
            None => Status::Active { player: 0 },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend(0..Self::WIDTH - 1);
        if self.decoy_legal {
            out.push(Self::DECOY);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, _: &mut R) {
        self.played = Some(*choice);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        match self.status(&()) {
            Status::Terminal(rewards) => rewards,
            _ => unreachable!("every choice ends the game"),
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.decoy_legal = below(rng, Self::RARITY) == 0;
    }
}

/// The same decoy at a root wide enough that the leader's own rate is nowhere
/// near 1.0 — where ranking on the rate is at its most wrong, because
/// sixty-four candidates share the opportunities and one of them does not.
///
/// The decoy is offered a few dozen times in ten thousand iterations and taken
/// on essentially every one of them, so it holds the best selection rate at the
/// root by a distance no discount of that rate closes cheaply. What rules it out
/// is the thing the ranking leads with: it is worth 0.85 where the answer is
/// worth 0.90, and the mean has no ceiling for a saturated rate to hide behind.
#[test]
fn a_manufactured_rate_does_not_win_a_wide_root() {
    const BUDGET: u32 = 10_000;

    for seed in [1u64, 7, 21] {
        let game = WideDecoy::new();
        let cfg = Config {
            iterations: BUDGET,
            ..Config::default()
        };
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));

        let root = searcher.tree().expect("a search leaves a tree");
        let rate = |choice: usize| {
            let child = root
                .children()
                .iter()
                .find(|child| *child.edge().choice().expect("a choice") == choice)
                .expect("every choice is in the tree");
            (
                child.visits(),
                child.visits() as f64 / child.availability().max(1) as f64,
            )
        };
        let (decoy_visits, decoy_rate) = rate(WideDecoy::DECOY);
        let (best_visits, best_rate) = rate(WideDecoy::BEST);
        assert!(
            decoy_rate > best_rate,
            "seed {seed}: the decoy must hold the higher selection rate, or this \
             fixture tests nothing: {decoy_rate} against {best_rate}"
        );
        assert!(
            decoy_visits * 20 < best_visits,
            "seed {seed}: the decoy must be backed by far less evidence than the \
             leader, or this fixture tests nothing: {decoy_visits} against \
             {best_visits}"
        );
        assert_eq!(
            result.choice,
            WideDecoy::BEST,
            "seed {seed}: answered {} (worth {}) on a rate of {decoy_rate} against \
             the leader's {best_rate}",
            result.choice,
            WideDecoy::payoff(result.choice)
        );
    }
}

/// A one-ply root of two equally worthless moves the position holds, and a third
/// the search sees in all but one world in [`PhantomLead::RARITY`] and this
/// position does not offer.
///
/// The phantom pays 1.0 where the two real moves pay nothing, and it is legal in
/// nearly every determinization, so the search abandons the real moves after a
/// handful of selections: they end on about a dozen visits each against the
/// phantom's two thousand. That is the shape the surviving early-termination
/// proof fires on — the leader established, every rival stuck under the evidence
/// bar — and here the leader is a move the player does not hold.
#[derive(Clone)]
struct PhantomLead {
    played: Option<usize>,
    phantom_legal: bool,
}

impl PhantomLead {
    const PHANTOM: usize = 2;
    /// One determinization in this many withholds the phantom, which is what
    /// leaves the two real moves any visits at all.
    const RARITY: u64 = 128;

    fn new() -> Self {
        Self {
            played: None,
            phantom_legal: false,
        }
    }
}

impl Game for PhantomLead {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.played {
            Some(choice) => {
                let payoff = if choice == Self::PHANTOM { 1.0 } else { 0.0 };
                Status::Terminal([payoff, 1.0 - payoff])
            }
            None => Status::Active { player: 0 },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend([0, 1]);
        if self.phantom_legal {
            out.push(Self::PHANTOM);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, _: &mut R) {
        self.played = Some(*choice);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        match self.status(&()) {
            Status::Terminal(rewards) => rewards,
            _ => unreachable!("every choice ends the game"),
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.phantom_legal = below(rng, Self::RARITY) != 0;
    }
}

/// The early-termination proof has to rank the candidates the answer is drawn
/// from, not the ones the tree happens to hold.
///
/// Ranked over all three children, the phantom has cleared the evidence bar and
/// both of its rivals are stuck far under it, so the proof fires and stamps
/// `Proven` on a move the player cannot play — while the contest that actually
/// decides the answer, between two children the search barely sampled, is
/// settled by nothing at all. Ranked over the two legal children, neither has
/// cleared the bar and there is nothing to prove. `StopReason::Budget` here *is*
/// the mask.
#[test]
fn a_proof_is_not_read_off_a_move_the_position_withholds() {
    const BUDGET: u32 = 2_000;
    // `rank::MIN_EVIDENCE`, which is not public.
    const EVIDENCE_BAR: u32 = 32;

    for seed in [1u64, 7, 21] {
        let game = PhantomLead::new();
        let cfg = Config {
            iterations: BUDGET,
            ..Config::default()
        };
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));

        let visits: Vec<u32> = searcher
            .tree()
            .expect("a search leaves a tree")
            .children()
            .iter()
            .map(|child| child.visits())
            .collect();
        let held = visits[0].max(visits[1]);
        assert!(
            visits[PhantomLead::PHANTOM] >= EVIDENCE_BAR && held + 1 < EVIDENCE_BAR,
            "seed {seed}: an unmasked proof has to be available for this seed to say \
             anything — the phantom needs the evidence and the legal moves must not \
             have it, with an iteration to spare: {visits:?}"
        );
        assert_eq!(
            result.stop_reason,
            mcts::StopReason::Budget,
            "seed {seed}: proved an answer it never established, on {visits:?}"
        );
        assert_eq!(
            result.iterations_used,
            BUDGET,
            "seed {seed}: stopped {} iterations early",
            BUDGET - result.iterations_used
        );
    }
}

/// Two plies of `arms` actions, where a hidden coin decides in each
/// determinization whether the upper half of the action list was dealt at all,
/// and the payoff is a coin flip either side of the first action's value.
///
/// Three properties at once, and the sweep below needs all three: the legal set
/// varies between determinizations, so the root accumulates candidates the real
/// position may not hold; the rewards land in `[-1, 1]` while
/// [`Config::min_reward`] and [`Config::max_reward`] are left at their `[0, 1]`
/// defaults, so any proof that read those numbers as a clamp would be proving
/// against an interval the rewards leave; and every action's mean is a sample
/// mean rather than a constant, so no ranking here is settled by the first
/// rollout.
#[derive(Clone)]
struct Straddle {
    arms: usize,
    depth: u8,
    first: Option<usize>,
    payoff: f64,
    upper_half_dealt: bool,
    player: u8,
}

impl Straddle {
    const PLIES: u8 = 4;

    fn new(arms: usize) -> Self {
        Self {
            arms,
            depth: 0,
            first: None,
            payoff: 0.0,
            upper_half_dealt: true,
            player: 0,
        }
    }
}

impl Game for Straddle {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.depth >= Self::PLIES {
            Status::Terminal([self.payoff, -self.payoff])
        } else {
            Status::Active {
                player: self.player,
            }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        let dealt = if self.upper_half_dealt {
            self.arms
        } else {
            self.arms / 2 + 1
        };
        out.extend(0..dealt);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, rng: &mut R) {
        if self.first.is_none() {
            self.first = Some(*choice);
            let mean = if *choice == 1 { 0.5 } else { -0.1 };
            self.payoff = mean + if below(rng, 2) == 0 { -0.5 } else { 0.5 };
        }
        self.depth += 1;
        self.player ^= 1;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &(), rng: &mut R) -> [f64; 2] {
        let mut choices = Vec::new();
        while self.depth < Self::PLIES {
            choices.clear();
            self.choices_into(ctx, &mut choices);
            let choice = choices[below(rng, choices.len() as u64) as usize];
            self.apply_choice(ctx, &choice, rng);
        }
        [self.payoff, -self.payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.upper_half_dealt = below(rng, 3) != 0;
    }
}

/// A root of `arms` actions where action 1 is worth 0.05 and the rest 0.00,
/// each paid as a coin flip half a point either side of its mean.
///
/// The signal is a tenth of the noise, so which action leads changes hands
/// throughout the search: exactly the root where a proof that overstates what
/// it knows is caught, and the reason the sweep below runs a shape with nothing
/// to find alongside one with something.
#[derive(Clone)]
struct LowSignal {
    arms: usize,
    played: Option<usize>,
    payoff: f64,
}

impl Game for LowSignal {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.played {
            Some(_) => Status::Terminal([self.payoff, -self.payoff]),
            None => Status::Active { player: 0 },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend(0..self.arms);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, rng: &mut R) {
        self.played = Some(*choice);
        let mean = if *choice == 1 { 0.05 } else { 0.0 };
        self.payoff = mean + if below(rng, 2) == 0 { -0.5 } else { 0.5 };
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        [self.payoff, -self.payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// One search that stopped on a proof, and the same search run to the end of
/// its budget instead.
struct Proof {
    stopped: mcts::SearchResult<usize>,
    full: mcts::SearchResult<usize>,
}

/// `None` if the search spent its budget rather than proving anything.
fn proof_of<G: Game<Choice = usize, Context = ()>>(
    game: &G,
    budget: u32,
    seed: u64,
) -> Option<Proof> {
    let cfg = |early_termination| Config {
        iterations: budget,
        early_termination,
        ..Config::default()
    };
    let stopped = Searcher::new(game).search(game, &(), 0, &cfg(true), None, &mut rng(seed));
    if stopped.stop_reason != mcts::StopReason::Proven {
        return None;
    }
    let full = Searcher::new(game).search(game, &(), 0, &cfg(false), None, &mut rng(seed));
    Some(Proof { stopped, full })
}

/// What [`mcts::StopReason::Proven`] claims, swept rather than sampled: the
/// iterations the search did not spend could not have changed the answer.
///
/// The single-fixture early-termination tests each pin one shape of the proof.
/// This pins the claim itself over every budget that straddles the evidence bar
/// — the same search run to the end of its budget must answer what the proof
/// let it stop on, for every seed on which the proof fired. A false `Proven` is
/// the worst thing this crate can return, and it is worth a sweep rather than a
/// seed.
#[test]
fn a_proof_answers_what_the_whole_budget_would() {
    let mut proofs = 0;
    let mut low_signal_proofs = 0;
    let mut wrong = Vec::new();
    let mut record = |name: &str, arms: usize, budget: u32, seed: u64, proof: Proof| {
        if proof.stopped.choice != proof.full.choice {
            wrong.push(format!(
                "{name} arms={arms} budget={budget} seed={seed}: proved {} after {} \
                 iterations where the whole budget answers {}",
                proof.stopped.choice, proof.stopped.iterations_used, proof.full.choice
            ));
        }
    };

    for arms in [2usize, 3, 5, 8] {
        for budget in [35u32, 45, 60, 100, 300, 1_000] {
            for seed in 1u64..=20 {
                if let Some(proof) = proof_of(&Straddle::new(arms), budget, seed) {
                    proofs += 1;
                    record("straddle", arms, budget, seed, proof);
                }
            }
        }
    }

    for arms in [60usize, 120, 200, 400] {
        for budget in [300u32, 500, 800, 1_200, 2_000] {
            for seed in 1u64..=20 {
                let game = LowSignal {
                    arms,
                    played: None,
                    payoff: 0.0,
                };
                if let Some(proof) = proof_of(&game, budget, seed) {
                    proofs += 1;
                    low_signal_proofs += 1;
                    record("low-signal", arms, budget, seed, proof);
                }
            }
        }
    }

    assert!(
        proofs >= 100 && low_signal_proofs > 0,
        "the sweep proved {proofs} answers, {low_signal_proofs} of them at a root whose \
         leader changes hands, which is too few for its silence to mean anything"
    );
    assert!(wrong.is_empty(), "{} false proofs: {wrong:#?}", wrong.len());
}
