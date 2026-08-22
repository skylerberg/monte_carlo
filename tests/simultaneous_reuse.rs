//! Tree reuse at a simultaneous root.
//!
//! A simultaneous node's successor is a tuple, not a choice, so
//! `reuse_subtree` cannot name it and `reuse_joint` mostly cannot find it: a
//! node with `prod(|A_i|)` successors materializes at most one per iteration.
//! These tests pin both halves of that — what survives a hit, and that a miss
//! is survivable — because the hit path is the only place in the crate where
//! statistics outlive the node that owns them.

// Each test binary compiles `tests/common` on its own, and this one uses the
// simultaneous fixtures rather than the sequential ones.
#[allow(dead_code)]
mod common;

use common::{ManyArm, Rps, MANY_ARM_ACTIONS, PENNIES_PAYOFFS};
use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, JointChoices, Node, NodeKind, PlayerSet, Searcher, Status};
use wyrand::WyRand;

fn rng(seed: u64) -> WyRand {
    WyRand::seed_from_u64(seed)
}

fn config(iterations: u32) -> Config {
    Config {
        iterations,
        early_termination: false,
        ..Config::default()
    }
}

/// Uniform draw from `0..n`, by the same multiply-shift the crate uses.
fn below<R: Rng + ?Sized>(rng: &mut R, n: u64) -> u64 {
    ((rng.next_u64() as u128 * n as u128) >> 64) as u64
}

/// Player 1's third action in the second round of [`TwoRounds`], offered only
/// by the determinizations that allow it.
const EDGE: u8 = 2;

/// What [`EDGE`] pays player 0, and so worth `0.75` to player 1 — more than
/// the `0.5` the pennies subgame is worth them, so the arm usually carries
/// real statistics rather than noise.
///
/// Only *usually*: no assertion in this file requires the arm to have been
/// selected at all. Regret matching under bandit feedback estimates an
/// unsampled arm's utility as zero and its regret falls monotonically, so
/// reaching it again depends entirely on the exploration floor. That used to be
/// a real hazard: under the `0.1 / sqrt(t)` floor this crate first shipped, the
/// arm ended with **zero** visits in 4 of 17 seeds at this node's ~1300 visits,
/// even at this payoff. Under the floor `duct.rs`'s `exploration_floor` now
/// uses it is 0 of 17, with 85 to 729 visits against 570 to 866 availabilities.
/// The hedge stays in this comment because nothing here asserts on it, not
/// because it is still expected to fail.
const EDGE_PAYOFF: f64 = 0.25;

/// Two simultaneous rounds of matching pennies, the second offering player 1 a
/// third action that only half the determinizations allow.
///
/// Local rather than shared because nothing in `tests/common` has two
/// simultaneous plies in a row, and every assertion here about a *promoted*
/// root needs the node `reuse_joint` promotes to own arms: a joint successor
/// of `Pennies`, `Rps` or `SeqThenSimul` is terminal, so promoting one leaves
/// a leaf with nothing to check and nothing to search.
///
/// The variable third action is what makes an arm's availability a different
/// number from its visits and from its node's visit count. Without it every
/// availability equals the node's visits, and a promotion that zeroed
/// availability while keeping visits — or reset it to the node's own
/// re-rooted `1` — would still satisfy an assertion that only compared the
/// two.
#[derive(Clone, Default)]
struct TwoRounds {
    round: u8,
    score: f64,
    edge_legal: bool,
}

impl TwoRounds {
    /// Resolve one round outside the search, so a test can walk the real game
    /// to the position it has just re-rooted the tree onto.
    fn play(&mut self, first: u8, second: u8) {
        self.score += if second == EDGE {
            EDGE_PAYOFF
        } else {
            PENNIES_PAYOFFS[first as usize][second as usize]
        };
        self.round += 1;
    }
}

impl Game for TwoRounds {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.round >= 2 {
            let payoff = self.score / 2.0;
            Status::Terminal([payoff, 1.0 - payoff])
        } else {
            Status::Simultaneous {
                players: PlayerSet::first_n(2),
            }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend([0, 1]);
    }

    fn choices_for_into(&self, ctx: &(), player: u8, out: &mut Vec<u8>) {
        self.choices_into(ctx, out);
        if self.round == 1 && player == 1 && self.edge_legal {
            out.push(EDGE);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("TwoRounds has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, _: &mut R) {
        self.play(*joint.get(0), *joint.get(1));
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        while self.round < 2 {
            self.play(below(rng, 2) as u8, below(rng, 2) as u8);
        }
        let payoff = self.score / 2.0;
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
        dest.clone_from(self);
        dest.edge_legal = rng.next_u64() & 1 == 1;
    }
}

/// Every joint successor that materialized under `node`, as its child index
/// and the tuple of actions that reaches it, most-visited first.
///
/// A `reuse_joint` miss is the common case by design, so a test that wants a
/// hit must read a tuple back out of the tree — through `Node::joint_arm` and
/// `Marginals::choice`, the only public route from a joint child to the
/// actions that made it — rather than assume any particular tuple is there.
fn materialized_tuples(node: &Node<u8>) -> Vec<(usize, Vec<(u8, u8)>)> {
    let players = node.simultaneous_players().expect("a simultaneous node");
    let mut tuples: Vec<(usize, Vec<(u8, u8)>)> = node
        .children()
        .iter()
        .enumerate()
        .map(|(child, _)| {
            let played = players
                .iter()
                .map(|player| {
                    let arm = node.joint_arm(child, player).expect("a joint successor");
                    let choice = *node.marginals(player).expect("a participant").choice(arm);
                    (player, choice)
                })
                .collect();
            (child, played)
        })
        .collect();
    tuples.sort_by_key(|&(child, _)| std::cmp::Reverse(node.children()[child].visits()));
    tuples
}

/// Every participant's arms at `node`, as `(choice, visits, availability)`.
fn arms(node: &Node<u8>) -> Vec<Vec<(u8, u32, u32)>> {
    let players = node.simultaneous_players().expect("a simultaneous node");
    players
        .iter()
        .map(|player| {
            let marginals = node.marginals(player).expect("a participant");
            (0..marginals.len())
                .map(|arm| {
                    (
                        *marginals.choice(arm),
                        marginals.visits(arm),
                        marginals.availability(arm),
                    )
                })
                .collect()
        })
        .collect()
}

/// Every participant's extracted mixed strategy at `node`.
///
/// `strategy_sum` is not directly readable, and it is the accumulator a
/// promotion could drop without moving a visit count. Its ratio against
/// availability is exactly what `policy_into` returns, so comparing these
/// before and after is the one public check that both survived together.
fn policies(node: &Node<u8>) -> Vec<Vec<f64>> {
    let players = node.simultaneous_players().expect("a simultaneous node");
    players
        .iter()
        .map(|player| {
            let mut out = Vec::new();
            node.marginals(player)
                .expect("a participant")
                .policy_into(&mut out);
            out
        })
        .collect()
}

/// A searcher holding a tree whose root is one simultaneous ply, for the
/// argument-validation tests, which never get as far as looking one up.
fn searched_rps() -> Searcher<Rps> {
    let game = Rps::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(200), None, &mut rng(19));
    searcher
}

/// The promotion is only worth having if the promoted node arrives intact: its
/// visits become the next search's `reused_iterations`, and its arms are the
/// bandits that search continues from. An arm that came back with its counts
/// zeroed would keep its mean and lose its exploration term, and would sit
/// frozen at whatever it happened to have learned.
#[test]
fn reuse_joint_carries_arm_statistics_forward() {
    let mut game = TwoRounds::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(4_000), None, &mut rng(3));

    let root = searcher.tree().expect("a search leaves a tree");
    let (child, played) = materialized_tuples(root)
        .into_iter()
        .next()
        .expect("the search materialized at least one joint successor");
    let promoted = &root.children()[child];
    let carried = promoted.visits();
    let before = arms(promoted);
    let strategies = policies(promoted);
    assert!(carried > 0);
    assert_eq!(before[0].len(), 2, "player 0's arms: {:?}", before[0]);
    assert_eq!(
        before[1].len(),
        3,
        "player 1's third action never showed up, so both slots hold the same arms \
         and this run does not exercise an asymmetric promotion: {:?}",
        before[1]
    );

    assert!(searcher.reuse_joint(&played));
    let root = searcher.tree().expect("the promoted node is the new root");
    assert_eq!(root.kind(), NodeKind::Root);
    assert!(root.choice().is_none());
    assert_eq!(root.visits(), carried);
    assert_eq!(
        arms(root),
        before,
        "arm statistics did not survive the promotion"
    );
    assert_eq!(
        policies(root),
        strategies,
        "the extracted strategy moved, so strategy_sum and availability did not \
         survive together"
    );

    game.play(played[0].1, played[1].1);
    let target = carried + 2_000;
    let second = searcher.search(&game, &(), 0, &config(target), None, &mut rng(5));
    assert_eq!(second.reused_iterations, carried);
    assert_eq!(second.iterations_used, 2_000);
    assert_eq!(second.root_visits, target);
    assert!(second.choice < 2, "chose {}", second.choice);
}

/// Every arm keeps the availability it earned at this node, while the node's
/// own availability is reset to 1 by the re-rooting. The asymmetry is the
/// point: a node's availability counts iterations at the parent that was just
/// discarded, an arm's counts iterations at the node itself, and those are
/// still the denominator of its exploration term and the divisor that makes
/// its strategy comparable to a more often legal arm's.
#[test]
fn a_reused_simultaneous_root_keeps_arm_availability() {
    let game = TwoRounds::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(4_000), None, &mut rng(17));

    let root = searcher.tree().expect("a search leaves a tree");
    let (child, played) = materialized_tuples(root)
        .into_iter()
        .next()
        .expect("the search materialized at least one joint successor");
    let promoted = &root.children()[child];
    let visits = promoted.visits();
    let before = arms(promoted);

    let &(_, _, edge_availability) = before[1]
        .iter()
        .find(|&&(choice, _, _)| choice == EDGE)
        .expect("player 1's variable action is what separates availability from visits");
    let always_legal = before[1][0].2;
    assert!(
        edge_availability > 0 && edge_availability < always_legal,
        "the variable action was legal in {edge_availability} of {visits} visits \
         against an always-legal arm's {always_legal}, so availability here is \
         indistinguishable from the node's own visit count"
    );
    let most_selected = before[1]
        .iter()
        .map(|&(_, visits, _)| visits)
        .max()
        .expect("player 1 has arms");
    assert!(
        most_selected > 0 && most_selected < always_legal,
        "player 1's most-selected arm took {most_selected} of {always_legal} \
         availabilities, so no arm's visits differ from its availability and an \
         assertion that availability survived cannot tell the two counters apart"
    );

    assert!(searcher.reuse_joint(&played));
    let root = searcher.tree().expect("the promoted node is the new root");
    let after = arms(root);
    assert_eq!(after, before, "availability was rewritten by promotion");
    for (slot, player_arms) in after.iter().enumerate() {
        for &(choice, _, availability) in player_arms {
            assert!(
                availability > 1,
                "slot {slot}'s action {choice} came back with availability \
                 {availability}, which is the node's own re-rooted value rather than \
                 the arm's"
            );
        }
    }
}

/// A miss is the common case, not an error, so it has to cost a fresh search
/// rather than a wrong one. Leaving the tree in place would root the next
/// search at last round's position.
#[test]
fn reuse_joint_misses_are_survivable() {
    let game = ManyArm::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(400), None, &mut rng(11));

    let root = searcher.tree().expect("a search leaves a tree");
    let visited: Vec<Vec<(u8, u8)>> = materialized_tuples(root)
        .into_iter()
        .map(|(_, played)| played)
        .collect();
    let possible = usize::from(MANY_ARM_ACTIONS).pow(2);
    assert!(!visited.is_empty());
    assert!(
        visited.len() < possible,
        "{} of {possible} tuples materialized, so there is no miss to test",
        visited.len()
    );

    let missing = (0..MANY_ARM_ACTIONS)
        .flat_map(|a| (0..MANY_ARM_ACTIONS).map(move |b| vec![(0u8, a), (1u8, b)]))
        .find(|tuple| !visited.contains(tuple))
        .expect("400 iterations cannot materialize 1600 joint successors");

    assert!(!searcher.reuse_joint(&missing));
    assert!(
        searcher.tree().is_none(),
        "a miss left the previous round's tree in place"
    );

    let next = searcher.search(&game, &(), 0, &config(400), None, &mut rng(12));
    assert_eq!(next.reused_iterations, 0);
    assert!(
        next.choice < MANY_ARM_ACTIONS,
        "chose {}, which is not a legal action",
        next.choice
    );
}

/// `reuse_subtree` names a successor by one player's choice, which at a
/// simultaneous root does not identify one. In debug that is a named panic; in
/// release it is a refusal that drops the tree, so the caller gets a fresh
/// search rather than a tree rooted at a position nobody reached.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "reuse_subtree was called on a simultaneous root")]
fn reuse_subtree_refuses_a_simultaneous_root() {
    let game = Rps::default();
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &config(200), None, &mut rng(13));
    searcher.reuse_subtree(&result.choice);
}

/// See the debug-build test of the same name: this is the release half, where
/// the refusal is a return value rather than a panic.
#[cfg(not(debug_assertions))]
#[test]
fn reuse_subtree_refuses_a_simultaneous_root() {
    let game = Rps::default();
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &config(200), None, &mut rng(13));

    assert!(!searcher.reuse_subtree(&result.choice));
    assert!(searcher.tree().is_none());

    let next = searcher.search(&game, &(), 0, &config(200), None, &mut rng(14));
    assert_eq!(next.reused_iterations, 0);
    assert!(next.choice < 3);
}

/// A tuple naming somebody who does not act here is a caller error, not a
/// miss: silently ignoring it would re-root at whatever the remaining entries
/// happened to match.
#[test]
#[should_panic(expected = "reuse_joint names player 3, who does not act at this node")]
fn reuse_joint_rejects_a_non_participant() {
    searched_rps().reuse_joint(&[(0, 0), (3, 1)]);
}

/// Naming one player twice leaves another player's slot at its default, which
/// would re-root at a tuple nobody played.
#[test]
#[should_panic(expected = "reuse_joint names player 1 twice")]
fn reuse_joint_rejects_a_repeated_player() {
    searched_rps().reuse_joint(&[(1, 0), (1, 1)]);
}

/// The same trap reached by the shorter route: too few entries cannot name
/// every participant.
#[test]
#[should_panic(expected = "reuse_joint was given 1 actions for 2 participants")]
fn reuse_joint_rejects_a_tuple_of_the_wrong_length() {
    searched_rps().reuse_joint(&[(0, 0)]);
}
