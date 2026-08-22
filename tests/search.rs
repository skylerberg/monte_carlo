mod common;

use common::{AlwaysWin, CountToThree, GameTree, TreeNode};
use mcts::rand_core::SeedableRng;
use mcts::{Config, Node, Searcher, StopReason};
use wyrand::WyRand;

fn rng(seed: u64) -> WyRand {
    WyRand::seed_from_u64(seed)
}

fn config(iterations: u32) -> Config {
    Config {
        iterations,
        exploration_constant: 0.4,
        early_termination: false,
        ..Config::default()
    }
}

fn visit_vector(root: &Node<usize>) -> Vec<(usize, u32)> {
    root.children()
        .iter()
        .map(|child| (*child.edge().choice().unwrap(), child.visits()))
        .collect()
}

#[test]
fn explores_each_option_once() {
    let game = AlwaysWin { ply: 0 };
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 1, &config(3), None, &mut rng(1));

    let root = searcher.tree().unwrap();
    assert_eq!(root.children().len(), 3);
    assert!(root.children().iter().all(|child| child.visits() == 1));
    assert!(root
        .children()
        .iter()
        .all(|child| child.cumulative_reward() == 1.0));
}

#[test]
fn explores_evenly_when_nothing_distinguishes_choices() {
    let game = AlwaysWin { ply: 0 };
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 1, &config(81), None, &mut rng(1));

    let root = searcher.tree().unwrap();
    assert_eq!(root.children().len(), 3);
    assert!(
        root.children().iter().all(|child| child.visits() == 27),
        "expected 27 visits each, got {:?}",
        visit_vector(root)
    );
}

#[test]
fn finds_the_only_winning_line() {
    let game = CountToThree::new();
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &config(200), None, &mut rng(7));
    assert_eq!(result.choice, 1);
}

#[test]
fn defeats_a_shallow_trap() {
    let game = GameTree::minimal_trap();
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &config(200), None, &mut rng(11));
    assert_eq!(
        result.choice, 1,
        "took the branch that looks good but always loses"
    );
}

#[test]
fn same_seed_gives_the_same_search() {
    let game = GameTree::minimal_trap();

    let run = || {
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &config(150), None, &mut rng(99));
        let visits = visit_vector(searcher.tree().unwrap());
        (result.choice, result.root_visits, visits)
    };

    assert_eq!(run(), run());
}

#[test]
fn a_forced_move_skips_the_search() {
    let game = GameTree {
        state: TreeNode::branch(vec![TreeNode::winner(0)]),
        active_player: 0,
        player_count: 2,
    };
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &config(10_000), None, &mut rng(3));

    assert_eq!(result.stop_reason, StopReason::SingleChoice);
    assert_eq!(result.iterations_used, 0);
    assert_eq!(result.choice, 0);
}

#[test]
fn reused_subtree_carries_its_statistics_forward() {
    let mut game = GameTree::minimal_trap();
    let mut searcher = Searcher::new(&game);

    let first = searcher.search(&game, &(), 0, &config(100), None, &mut rng(5));
    let carried = searcher
        .tree()
        .unwrap()
        .children()
        .iter()
        .find(|child| *child.edge().choice().unwrap() == first.choice)
        .unwrap()
        .visits();
    assert!(carried > 0);

    assert!(searcher.reuse_subtree(&first.choice));
    game.apply(&first.choice);

    let second = searcher.search(&game, &(), 0, &config(200), None, &mut rng(5));
    assert_eq!(second.reused_iterations, carried);
    assert!(second.iterations_used < 200);
    assert_eq!(second.root_visits, 200);
}

/// A retained tree describes the position it was built for. Searching a new
/// position without re-rooting must not inherit it — otherwise the search
/// returns a choice that was legal at the previous position and may not even
/// exist at this one.
#[test]
fn a_tree_is_not_reused_unless_asked_for() {
    let mut game = GameTree::minimal_trap();
    let mut searcher = Searcher::new(&game);

    let first = searcher.search(&game, &(), 0, &config(100), None, &mut rng(5));
    game.apply(&first.choice);

    // Deliberately no reuse_subtree call here.
    let second = searcher.search(&game, &(), 1, &config(100), None, &mut rng(5));

    assert_eq!(
        second.reused_iterations, 0,
        "stale tree was carried forward"
    );
    assert_eq!(second.root_visits, 100);

    let legal: Vec<usize> = (0..game.state.children.len()).collect();
    assert!(
        legal.contains(&second.choice),
        "chose {} which is not legal here; legal choices are {legal:?}",
        second.choice
    );
}

/// A re-rooted subtree carries only the children progressive expansion happened
/// to create at the previous position. The next search must still open every
/// legal choice at the new one — otherwise it can only ever return a move from
/// that stale subset, which may no longer be legal.
#[test]
fn a_reused_root_still_expands_every_choice() {
    let mut game = GameTree::wide_two_ply(40);
    let mut searcher = Searcher::new(&game);

    let first = searcher.search(&game, &(), 0, &config(60), None, &mut rng(17));
    assert!(searcher.reuse_subtree(&first.choice));
    game.apply(&first.choice);

    // The subtree kept far fewer children than the new position has choices.
    let carried = searcher.tree().unwrap().children().len();
    let legal = game.state.children.len();
    assert!(
        carried < legal,
        "test is not exercising the gap: carried {carried}, legal {legal}"
    );

    searcher.search(&game, &(), 1, &config(300), None, &mut rng(19));
    assert_eq!(
        searcher.tree().unwrap().children().len(),
        legal,
        "reused root was never fully expanded, so most legal choices are unreachable"
    );
}

#[test]
fn taking_the_tree_leaves_the_searcher_empty() {
    let game = GameTree::minimal_trap();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(80), None, &mut rng(4));

    let tree = searcher.take_tree().expect("a search leaves a tree");
    assert_eq!(tree.visits(), 80);
    assert!(searcher.tree().is_none());

    // Having taken it, the next search must start from nothing.
    let next = searcher.search(&game, &(), 0, &config(80), None, &mut rng(4));
    assert_eq!(next.reused_iterations, 0);
}

#[test]
fn wide_nodes_track_every_child() {
    const WIDTH: usize = 200;
    let game = GameTree::wide(WIDTH);
    let mut searcher = Searcher::new(&game);

    // Enough iterations to open every child and then favour the winning one.
    let result = searcher.search(&game, &(), 0, &config(1_000), None, &mut rng(13));

    let root = searcher.tree().unwrap();
    assert_eq!(
        root.children().len(),
        WIDTH,
        "every choice must get a child"
    );

    let mut seen: Vec<usize> = root
        .children()
        .iter()
        .map(|child| *child.edge().choice().unwrap())
        .collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), WIDTH, "children must be distinct");

    assert_eq!(result.choice, WIDTH - 1, "should settle on the only win");
}

#[test]
fn early_termination_does_not_change_the_answer() {
    let game = GameTree::minimal_trap();

    let full = {
        let mut searcher = Searcher::new(&game);
        searcher
            .search(&game, &(), 0, &config(400), None, &mut rng(21))
            .choice
    };

    let stopped = {
        let mut searcher = Searcher::new(&game);
        let cfg = Config {
            early_termination: true,
            ..config(400)
        };
        searcher.search(&game, &(), 0, &cfg, None, &mut rng(21))
    };

    assert_eq!(stopped.choice, full);
    assert!(stopped.root_visits <= 400);
}
