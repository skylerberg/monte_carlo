//! The `Game::ROOT_CHOICES_INVARIANT` fast path, against a re-rooted tree.
//!
//! The flag promises that the *root's* choice set does not vary across
//! determinizations, and the fast path it buys skips move generation and marks
//! every arm — or every child — available. What it must not do is skip the
//! proof that the tree's action set is the position's: a node one ply down
//! accumulates the union of every chance outcome that reached it, and
//! `reuse_subtree` and `reuse_joint` promote exactly such a node to root. The
//! flag being a true claim about determinizations says nothing about the arms
//! that promotion carried in.
//!
//! Both fixtures here determinize by cloning, so the invariant they declare is
//! genuinely true and the failures below cannot be blamed on the game. They are
//! local rather than shared because nothing in `tests/common` puts a chance
//! draw between two plies, which is what makes a node's action set depend on
//! how it was reached.

use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, JointChoices, Node, PlayerSet, Searcher, Status};
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

/// The action the chance draw grants or withholds at the second ply, and the
/// one that pays best, so a search allowed to reach it goes there.
const EDGE: u8 = 2;

fn payoff(choice: u8) -> f64 {
    match choice {
        EDGE => 1.0,
        1 => 0.6,
        _ => 0.4,
    }
}

/// Two simultaneous plies, with a chance draw resolved by the first deciding
/// whether player 0 still holds [`EDGE`] at the second.
///
/// The determinization is an exact clone, so `ROOT_CHOICES_INVARIANT` is a true
/// claim at both roots: at the first ply nobody holds `EDGE`, and at the second
/// the draw has already happened and every determinization of that position
/// agrees about it. The tree does not: the ply-2 node is reached under both
/// outcomes and its arms are their union.
#[derive(Clone, Default)]
struct ChanceRounds {
    round: u8,
    score: f64,
    edge_legal: bool,
}

impl ChanceRounds {
    /// Resolve one round outside the search with the draw forced, so a test can
    /// walk the real game to the position it has re-rooted the tree onto.
    fn play(&mut self, mine: u8, theirs: u8, edge_legal: bool) {
        self.score += payoff(mine) * if mine == theirs { 1.0 } else { 0.5 };
        self.round += 1;
        self.edge_legal = edge_legal;
    }
}

impl Game for ChanceRounds {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    const ROOT_CHOICES_INVARIANT: bool = true;

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

    fn choices_into(&self, _: &(), _: &mut Vec<u8>) {
        unreachable!("ChanceRounds names each player's actions through choices_for_into")
    }

    fn choices_for_into(&self, _: &(), player: u8, out: &mut Vec<u8>) {
        out.extend([0, 1]);
        if self.round == 1 && player == 0 && self.edge_legal {
            out.push(EDGE);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &u8, _: &mut R) {
        unreachable!("ChanceRounds has no sequential node")
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), joint: JointChoices<'_, u8>, rng: &mut R) {
        let draw = rng.next_u64() & 1 == 1;
        self.play(*joint.get(0), *joint.get(1), draw);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        while self.round < 2 {
            let mine = below(rng, 2) as u8;
            let theirs = below(rng, 2) as u8;
            let draw = rng.next_u64() & 1 == 1;
            self.play(mine, theirs, draw);
        }
        let payoff = self.score / 2.0;
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// The same shape sequentially: one player, two plies, and a chance draw at the
/// first deciding whether [`EDGE`] is on offer at the second.
#[derive(Clone, Default)]
struct ChanceLine {
    round: u8,
    score: f64,
    edge_legal: bool,
}

impl ChanceLine {
    fn play(&mut self, choice: u8, edge_legal: bool) {
        self.score += payoff(choice);
        self.round += 1;
        self.edge_legal = edge_legal;
    }
}

impl Game for ChanceLine {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    const ROOT_CHOICES_INVARIANT: bool = true;

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.round >= 2 {
            let payoff = self.score / 2.0;
            Status::Terminal([payoff, 1.0 - payoff])
        } else {
            Status::Active { player: 0 }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend([0, 1]);
        if self.round == 1 && self.edge_legal {
            out.push(EDGE);
        }
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &u8, rng: &mut R) {
        let draw = rng.next_u64() & 1 == 1;
        self.play(*choice, draw);
    }

    fn apply_joint<R: Rng + ?Sized>(&mut self, _: &(), _: JointChoices<'_, u8>, _: &mut R) {
        unreachable!("ChanceLine has no simultaneous node")
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        while self.round < 2 {
            let choice = below(rng, 2) as u8;
            let draw = rng.next_u64() & 1 == 1;
            self.play(choice, draw);
        }
        let payoff = self.score / 2.0;
        [payoff, 1.0 - payoff]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// The most-visited joint successor of `node`, as its child index and the
/// actions that reach it.
///
/// A `reuse_joint` hit needs a tuple that actually materialized, and the only
/// public route from a joint child back to the actions that made it is
/// [`Node::joint_arm`] with `Marginals::choice`.
fn most_visited_tuple(node: &Node<u8>) -> (usize, Vec<(u8, u8)>) {
    let players = node.simultaneous_players().expect("a simultaneous node");
    let child = (0..node.children().len())
        .max_by_key(|&child| node.children()[child].visits())
        .expect("the search materialized a joint successor");
    let played = players
        .iter()
        .map(|player| {
            let arm = node.joint_arm(child, player).expect("a joint successor");
            let choice = *node.marginals(player).expect("a participant").choice(arm);
            (player, choice)
        })
        .collect();
    (child, played)
}

/// `player`'s `(visits, availability)` for `action` at `node`, or `None` if the
/// node holds no arm for it.
fn arm(node: &Node<u8>, player: u8, action: u8) -> Option<(u32, u32)> {
    let marginals = node.marginals(player)?;
    (0..marginals.len())
        .find(|&arm| *marginals.choice(arm) == action)
        .map(|arm| (marginals.visits(arm), marginals.availability(arm)))
}

/// A node promoted by `reuse_joint` carries the arms of every determinization
/// that reached it at depth, and the position it is now the root of need not
/// offer them all. Arming the fast path on iteration 1 restamps those leftovers
/// as available, and from iteration 2 the search selects and applies an action
/// the player does not have — in debug tripping the fast path's own consistency
/// check, which then blames the game for a claim the game made correctly.
///
/// Arming it only once a full pass has *proved* the sets equal costs a promoted
/// root the fast path it cannot have, and leaves the withheld arm exactly where
/// promotion left it: never available, so never selected.
#[test]
fn a_promoted_simultaneous_root_never_offers_an_arm_the_position_withholds() {
    let game = ChanceRounds::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(4_000), None, &mut rng(7));

    let root = searcher.tree().expect("a search leaves a tree");
    let (child, played) = most_visited_tuple(root);
    let before = arm(&root.children()[child], 0, EDGE).expect(
        "the promoted node never saw the withheld action, so there is no leftover arm \
         to restamp and this test proves nothing",
    );

    let mut real = ChanceRounds::default();
    real.play(played[0].1, played[1].1, false);
    let mut legal = Vec::new();
    real.choices_for_into(&(), 0, &mut legal);
    assert!(
        !legal.contains(&EDGE),
        "the real position offers the withheld action, so nothing is being withheld"
    );

    assert!(
        searcher.reuse_joint(&played),
        "reuse_joint missed the tuple read out of the tree"
    );
    let promoted = arm(searcher.tree().expect("a promoted tree"), 0, EDGE)
        .expect("promotion carried the arm forward");

    let result = searcher.search(&real, &(), 0, &config(3_000), None, &mut rng(11));

    assert!(
        legal.contains(&result.choice),
        "the search returned {}, which is not one of player 0's actions {legal:?}",
        result.choice
    );
    assert_eq!(
        arm(searcher.tree().expect("a search leaves a tree"), 0, EDGE),
        Some(promoted),
        "the withheld arm was {promoted:?} (visits, availability) when the search \
         started and it moved, so the root was marked fully expanded on a pass that \
         never proved the tree's arms are the position's"
    );
    assert!(
        before.0 <= promoted.0,
        "sanity: promotion should not lose the arm's visits"
    );
}

/// The sequential twin of the test above, and the worse of the two: a
/// sequential root's answer is an unfiltered `most_visited` over the children,
/// so a leftover child that the fast path keeps marking available can win the
/// search outright and hand the caller a move it cannot play.
#[test]
fn a_re_rooted_sequential_root_never_offers_a_child_the_position_withholds() {
    let game = ChanceLine::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(2_000), None, &mut rng(4));

    let root = searcher.tree().expect("a search leaves a tree");
    let first = *root
        .children()
        .iter()
        .max_by_key(|child| child.visits())
        .expect("the search expanded the root")
        .choice()
        .expect("a root child has a choice");

    let mut real = ChanceLine::default();
    real.play(first, false);
    let mut legal = Vec::new();
    real.choices_into(&(), &mut legal);
    assert!(
        !legal.contains(&EDGE),
        "the real position offers the withheld action, so nothing is being withheld"
    );

    assert!(searcher.reuse_subtree(&first), "the child is a root child");

    let promoted = searcher.tree().expect("a promoted tree");
    let withheld = promoted
        .children()
        .iter()
        .position(|child| child.choice() == Some(&EDGE))
        .expect(
            "the promoted node never saw the withheld action, so there is no leftover \
             child to mark available and this test proves nothing",
        );
    let before = promoted.children()[withheld].visits();

    let result = searcher.search(&real, &(), 0, &config(2_000), None, &mut rng(9));

    assert!(
        legal.contains(&result.choice),
        "the search returned {}, which is not one of the position's choices {legal:?}",
        result.choice
    );
    let after = searcher
        .tree()
        .expect("a search leaves a tree")
        .children()
        .iter()
        .find(|child| child.choice() == Some(&EDGE))
        .expect("the leftover child is still there")
        .visits();
    assert_eq!(
        after, before,
        "the withheld child had {before} visits when the search started and has {after} \
         now, so the fast path kept marking it available"
    );
}
