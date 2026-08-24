//! The simultaneous arm index's cost, as a comparison count.
//!
//! A simultaneous node keeps one hash index over every participant's arms, and
//! `Game::Choice`'s `Eq` is what a lookup costs when that index is missing: the
//! whole of a slot's arm range, per choice, per determinization. This counts
//! those comparisons through the public API and holds them to a bound that only
//! an index which is both *built* at the node's first visit and *maintained*
//! across arm growth can meet. Those are two separate mechanisms, so there are
//! two measured regions:
//!
//! - [`DrawnHands`] grows its arm set for most of the search, and gates the
//!   maintenance: an index discarded on every newly discovered action.
//! - [`FixedHands`] never grows its arm set after the first visit, and gates
//!   the build: a node enters expansion with no arms at all, so an index built
//!   only on entry to a later expansion pass leaves the whole first pass
//!   scanning.
//!
//! Counts rather than wall clock, for the reason `tests/allocation.rs` gives:
//! a comparison count is reproducible on a machine that is doing other things.
//!
//! One `#[test]` on purpose — the counter is global, so a second test running
//! concurrently on another thread would be counted into this one's region.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, JointChoices, PlayerSet, Searcher, Status};
use wyrand::WyRand;

static COMPARISONS: AtomicU64 = AtomicU64::new(0);

/// A `Choice` that counts every equality test the library performs on it.
///
/// `Hash` reads the same field `Eq` does, so the index sees an ordinary
/// well-behaved key and the count measures lookup strategy alone.
#[derive(Clone, Copy, Eq, Debug)]
struct Action(u16);

impl PartialEq for Action {
    fn eq(&self, other: &Self) -> bool {
        COMPARISONS.fetch_add(1, Ordering::Relaxed);
        self.0 == other.0
    }
}

impl Hash for Action {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Actions in the pool a determinization draws from.
const POOL: u16 = 2_000;
/// How many of them any one determinization makes legal, per player.
const LEGAL: u16 = 50;
/// Coprime to [`POOL`], so a determinization's `LEGAL` actions are distinct.
const STRIDE: u16 = 3;
const ITERATIONS: u32 = 2_000;

/// One simultaneous ply between two players who each hold `LEGAL` of a pool of
/// `POOL` actions, redrawn every determinization.
///
/// The Information Set MCTS shape the arm index exists for: the tree's arm set
/// grows for as long as determinizations keep revealing actions no earlier one
/// offered, which is most of the search, and every lookup in between has to
/// find its arm among all the arms discovered so far.
#[derive(Clone, Default)]
struct DrawnHands {
    payoff: Option<f64>,
    deal: u64,
}

impl DrawnHands {
    /// The `k`th action player `p` holds in deal `deal`.
    fn action(deal: u64, p: u8, k: u16) -> Action {
        let mut x = deal ^ (u64::from(p) << 40);
        x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 31;
        Action(((x % u64::from(POOL)) as u16).wrapping_add(k.wrapping_mul(STRIDE)) % POOL)
    }
}

impl Game for DrawnHands {
    type Choice = Action;
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

    fn choices_into(&self, _: &(), out: &mut Vec<Action>) {
        self.choices_for_into(&(), 0, out);
    }

    fn choices_for_into(&self, _: &(), player: u8, out: &mut Vec<Action>) {
        out.extend((0..LEGAL).map(|k| Self::action(self.deal, player, k)));
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &Action, _: &mut R) {
        unreachable!("DrawnHands has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, Action>, _: &mut R) {
        let (a, b) = (joint.get(0).0, joint.get(1).0);
        self.payoff = Some(f64::from((a % 11 + b % 7) % 11) / 10.0);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let deal = self.deal;
        let payoff = *self.payoff.get_or_insert_with(|| {
            let a = Self::action(deal, 0, (rng.next_u64() % u64::from(LEGAL)) as u16).0;
            let b = Self::action(deal, 1, (rng.next_u64() % u64::from(LEGAL)) as u16).0;
            f64::from((a % 11 + b % 7) % 11) / 10.0
        });
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.deal = rng.next_u64();
    }
}

/// Actions each player holds at [`FixedHands`]'s only node.
const WIDE: u16 = 512;
/// How many times the fixed node is visited.
const VISITS: u32 = 2;

/// One simultaneous ply between two players who each hold the same [`WIDE`]
/// actions, on every visit.
///
/// The shape that has no growth to amortize anything over: the node's whole arm
/// set is discovered on its first visit, and every action enumerated in that
/// pass has to be looked up among the arms the pass has already created. A node
/// starts with no arms, so it is below [`Game::CHILD_INDEX_THRESHOLD`] when
/// expansion begins and can only cross it while growing.
#[derive(Clone, Default)]
struct FixedHands {
    payoff: Option<f64>,
}

impl FixedHands {
    fn resolve(a: u16, b: u16) -> f64 {
        f64::from((a % 11 + b % 7) % 11) / 10.0
    }
}

impl Game for FixedHands {
    type Choice = Action;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    /// Small and explicit: the arms a node enumerates before it crosses this
    /// are the one part of the count that stays quadratic, so the ceiling has
    /// to name the number rather than inherit whatever the default is.
    const CHILD_INDEX_THRESHOLD: usize = 8;

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.payoff {
            Some(payoff) => Status::Terminal([payoff, 1.0 - payoff]),
            None => Status::Simultaneous {
                players: PlayerSet::first_n(2),
            },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<Action>) {
        self.choices_for_into(&(), 0, out);
    }

    fn choices_for_into(&self, _: &(), _player: u8, out: &mut Vec<Action>) {
        out.extend((0..WIDE).map(Action));
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &Action, _: &mut R) {
        unreachable!("FixedHands has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, Action>, _: &mut R) {
        self.payoff = Some(Self::resolve(joint.get(0).0, joint.get(1).0));
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let payoff = *self.payoff.get_or_insert_with(|| {
            let a = (rng.next_u64() % u64::from(WIDE)) as u16;
            let b = (rng.next_u64() % u64::from(WIDE)) as u16;
            Self::resolve(a, b)
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

/// Two regions, because the counter is global and two `#[test]`s would run
/// concurrently into it. Each stores zero before its own search.
#[test]
fn arm_lookup_stays_linear_in_the_legal_set() {
    growth_keeps_the_index();
    a_first_visit_builds_the_index();
}

/// An arm set that keeps growing: the index has to survive each new arm.
fn growth_keeps_the_index() {
    let game = DrawnHands::default();
    let mut searcher = Searcher::new(&game);
    let mut rng = WyRand::seed_from_u64(0x5EED_1234);
    let cfg = Config {
        iterations: ITERATIONS,
        early_termination: false,
        ..Config::default()
    };

    COMPARISONS.store(0, Ordering::Relaxed);
    searcher.search(&game, &(), 0, &cfg, None, &mut rng);
    let comparisons = COMPARISONS.load(Ordering::Relaxed);

    let root = searcher.tree().expect("a search leaves a tree");
    let arms: usize = (0..2)
        .map(|slot| {
            root.marginals(slot)
                .expect("a simultaneous root has marginals")
                .len()
        })
        .sum();
    assert_eq!(
        arms,
        2 * usize::from(POOL),
        "the two players hold {POOL} actions each and the search must have met every one \
         of them; an arm set that stopped growing would meet the bound below by doing \
         less work rather than by doing it better"
    );

    // What the work is: two participants, `LEGAL` lookups each, per iteration,
    // one indexed probe apiece, so 200,000 probes costing about one comparison
    // each once the arm exists. The other term is the debug-only root check,
    // which compares its two enumerations of the perspective player's `LEGAL`
    // actions once per iteration. Nothing else recurs: `early_termination` is
    // off, so `refresh_root_legal` runs once after the loop rather than per
    // iteration, and the sub-threshold scans belong to the node's first visit.
    // That puts the honest ceiling at 2 comparisons per lookup. Measured:
    // 296,186 (1.48 per lookup) with the index maintained across growth,
    // 5,477,504 with it discarded on every newly discovered action (18x), and
    // that ratio grows with the pool.
    let ceiling = 2 * u64::from(LEGAL) * 2 * u64::from(ITERATIONS);
    assert!(
        comparisons <= ceiling,
        "the search compared {comparisons} choices, over the {ceiling} that {ITERATIONS} \
         iterations of two {LEGAL}-action hands can account for. An arm lookup is scanning \
         the slot's whole arm range instead of hitting the index."
    );
}

/// A node's first visit, where the arm set does not exist yet.
fn a_first_visit_builds_the_index() {
    let game = FixedHands::default();
    let mut searcher = Searcher::new(&game);
    let mut rng = WyRand::seed_from_u64(0x5EED_1234);
    let cfg = Config {
        iterations: VISITS,
        early_termination: false,
        ..Config::default()
    };

    COMPARISONS.store(0, Ordering::Relaxed);
    searcher.search(&game, &(), 0, &cfg, None, &mut rng);
    let comparisons = COMPARISONS.load(Ordering::Relaxed);

    let root = searcher.tree().expect("a search leaves a tree");
    let arms: usize = (0..2)
        .map(|slot| {
            root.marginals(slot)
                .expect("a simultaneous root has marginals")
                .len()
        })
        .sum();
    assert_eq!(
        arms,
        2 * usize::from(WIDE),
        "both players' whole hands must have become arms on the first visit"
    );

    // What the work is: `VISITS` passes of 2 slots x `WIDE` lookups, plus a
    // post-loop `refresh_root_legal` over one slot and the debug-only root
    // check's list equality over one slot per visit — call it `VISITS + 2`
    // passes of 2 x `WIDE`, at one comparison each. The only quadratic term
    // left is the sub-threshold prefix of the very first pass: the arms
    // enumerated before the node crosses `CHILD_INDEX_THRESHOLD` are scanned,
    // T*(T+1)/2 = 36 comparisons, once for the node's whole life. Measured:
    // 2,597. With no build in `grow_slot` it is 133,376: slot 0's first pass
    // scans linearly, 512*511/2 = 130,816 comparisons of it, because a node
    // holds no arms at all — and so no index — when that pass begins.
    let ceiling = 2 * u64::from(WIDE) * u64::from(VISITS + 2);
    assert!(
        comparisons <= ceiling,
        "the first visit of a {WIDE}-arm node compared {comparisons} choices, over the \
         {ceiling} that {VISITS} visits of two {WIDE}-action hands can account for. The \
         index is not being built until after the pass that creates the arms."
    );
}
