mod common;

use common::{AlwaysWin, CountToThree, GameTree, TreeNode};
use mcts::rand_core::SeedableRng;
use mcts::{Config, Node, Searcher, StopReason};
use std::panic::{catch_unwind, AssertUnwindSafe};
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

/// `width` choices, then a ply with exactly one, then `width` again — the shape
/// a search needs to reach a forced move with a re-rooted tree in hand and
/// still have a position to search afterwards.
fn forced_middle_ply(width: usize) -> GameTree {
    let third_ply = || {
        let mut leaves: Vec<TreeNode> = (0..width - 1).map(|_| TreeNode::winner(1)).collect();
        leaves.push(TreeNode::winner(0));
        TreeNode::branch(leaves)
    };
    let forced = || TreeNode::branch(vec![third_ply()]);
    GameTree {
        state: TreeNode::branch((0..width).map(|_| forced()).collect()),
        active_player: 0,
        player_count: 2,
    }
}

/// A forced move returns without searching, so it never asks whether the tree
/// it is holding still describes anything. Leaving it armed hands the *next*
/// search — one nobody asked to reuse anything for — a tree rooted two
/// positions back, whose children can name moves that do not exist here.
#[test]
fn a_forced_move_disarms_the_tree() {
    let mut game = forced_middle_ply(4);
    let mut searcher = Searcher::new(&game);

    let first = searcher.search(&game, &(), 0, &config(100), None, &mut rng(5));
    assert!(searcher.reuse_subtree(&first.choice));
    game.apply(&first.choice);

    let forced = searcher.search(&game, &(), 1, &config(100), None, &mut rng(5));
    assert_eq!(forced.stop_reason, StopReason::SingleChoice);
    game.apply(&forced.choice);

    // Deliberately no reuse_subtree call here.
    let third = searcher.search(&game, &(), 0, &config(100), None, &mut rng(5));

    assert_eq!(
        third.reused_iterations, 0,
        "the forced move left the tree armed, so the next search inherited it"
    );
    assert_eq!(third.root_visits, 100);

    let legal: Vec<usize> = (0..game.state.children.len()).collect();
    assert!(
        legal.contains(&third.choice),
        "chose {} which is not legal here; legal choices are {legal:?}",
        third.choice
    );
}

/// The disarm has to happen above the forced-move return, not just at it: a
/// tree the caller never re-rooted describes an older position still, and
/// leaving it there would let a `reuse_subtree` after the forced move arm a
/// root two positions behind.
#[test]
fn a_forced_move_on_an_unarmed_searcher_drops_the_stale_tree() {
    let mut game = forced_middle_ply(4);
    let mut searcher = Searcher::new(&game);

    let first = searcher.search(&game, &(), 0, &config(100), None, &mut rng(5));
    // Deliberately no reuse_subtree call here.
    game.apply(&first.choice);

    let forced = searcher.search(&game, &(), 1, &config(100), None, &mut rng(5));
    assert_eq!(forced.stop_reason, StopReason::SingleChoice);
    assert!(
        searcher.tree().is_none(),
        "the forced move kept a tree describing an older position"
    );
}

/// Disarming is not discarding. A forced move is still a move the caller can
/// re-root on, and dropping the tree there would throw away a whole search for
/// a ply that had nothing to decide.
#[test]
fn a_forced_move_still_leaves_a_reusable_tree() {
    let mut game = forced_middle_ply(4);
    let mut searcher = Searcher::new(&game);

    let first = searcher.search(&game, &(), 0, &config(200), None, &mut rng(5));
    assert!(searcher.reuse_subtree(&first.choice));
    game.apply(&first.choice);

    let forced = searcher.search(&game, &(), 1, &config(200), None, &mut rng(5));
    assert_eq!(forced.stop_reason, StopReason::SingleChoice);

    assert!(
        searcher.reuse_subtree(&forced.choice),
        "the forced move's successor is still in the tree"
    );
    game.apply(&forced.choice);

    let third = searcher.search(&game, &(), 0, &config(200), None, &mut rng(5));
    assert!(
        third.reused_iterations > 0,
        "re-rooting after a forced move carried nothing forward"
    );
}

/// Three plies of three choices, so a search, a caught panic and a further move
/// all have somewhere to go.
fn three_plies() -> GameTree {
    let leaves = || TreeNode::branch((0..3).map(|_| TreeNode::winner(0)).collect());
    let middle = || TreeNode::branch((0..3).map(|_| leaves()).collect());
    GameTree {
        state: TreeNode::branch((0..3).map(|_| middle()).collect()),
        active_player: 0,
        player_count: 2,
    }
}

/// A search that unwinds never reaches its own last statement, so a flag
/// cleared there stays set. A caller who catches the panic and searches again
/// then gets the abandoned position's statistics pooled into the answer.
#[test]
fn a_panicking_search_disarms_the_tree() {
    let mut game = three_plies();
    let mut searcher = Searcher::new(&game);

    let first = searcher.search(&game, &(), 0, &config(100), None, &mut rng(5));
    assert!(searcher.reuse_subtree(&first.choice));
    game.apply(&first.choice);

    // The documented terminal-state panic, so no panicking fixture is needed.
    let terminal = GameTree {
        state: TreeNode::winner(0),
        active_player: 1,
        player_count: 2,
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        searcher.search(&terminal, &(), 1, &config(100), None, &mut rng(5));
    }));
    assert!(outcome.is_err(), "searching a terminal state must panic");

    game.apply(&0);
    let next = searcher.search(&game, &(), 0, &config(100), None, &mut rng(5));

    assert_eq!(
        next.reused_iterations, 0,
        "the abandoned search left the tree armed"
    );
    assert_eq!(next.root_visits, 100);
}

/// Three plies of three choices whose context is a fuse: every `status` call
/// burns one, and the call that finds it empty panics. Local rather than
/// shared, because a pooled search needs a `Sync` context and nothing in
/// `tests/common` has one.
#[cfg(feature = "parallel")]
#[derive(Clone, Default)]
struct FusedFork {
    ply: u8,
}

#[cfg(feature = "parallel")]
impl mcts::Game for FusedFork {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = std::sync::atomic::AtomicU32;
    type Side = ();

    fn status(&self, fuse: &Self::Context) -> mcts::Status<[f64; 2]> {
        use std::sync::atomic::Ordering::Relaxed;
        // Saturating, so the fuse stays burned out for every worker that
        // reaches it rather than wrapping back to a full one.
        fuse.fetch_update(Relaxed, Relaxed, |left| left.checked_sub(1))
            .expect("the fuse burned out");
        if self.ply >= 3 {
            mcts::Status::Terminal([0.5, 0.5])
        } else {
            mcts::Status::Active { player: 0 }
        }
    }

    fn choices_into(&self, _: &Self::Context, out: &mut Vec<u8>) {
        out.extend(0..3);
    }

    fn apply_choice<R: mcts::rand_core::Rng + ?Sized>(
        &mut self,
        _: &Self::Context,
        _: &u8,
        _: &mut R,
    ) {
        self.ply += 1;
    }

    fn rollout<R: mcts::rand_core::Rng + ?Sized>(
        &mut self,
        _: &Self::Context,
        _: &mut R,
    ) -> [f64; 2] {
        self.ply = 3;
        [0.5, 0.5]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: mcts::rand_core::Rng + ?Sized>(
        &self,
        dest: &mut Self,
        _: &Self::Context,
        _: u8,
        _: &mut R,
    ) {
        dest.clone_from(self);
    }
}

/// The pool's disarm is per worker and there is no pool-level one: `search`
/// re-raises a worker's panic straight out of the join, leaving every
/// `Searcher` in `self.workers` exactly as its thread left it. The workers that
/// finished had disarmed themselves; the abandoned one had not, so the next
/// pooled search — a new position, no reuse asked for — pooled the abandoned
/// position's visits and rewards into the merged answer.
#[cfg(feature = "parallel")]
#[test]
fn a_panicking_pooled_search_disarms_every_worker() {
    use mcts::RootParallel;
    use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

    const THREADS: usize = 2;
    const BUDGET: u32 = 100;

    let fuse = AtomicU32::new(u32::MAX);
    let mut game = FusedFork::default();
    let mut pool = RootParallel::new(THREADS, &game, |worker| rng(worker as u64 + 1));

    let first = pool.search(&game, &fuse, 0, &config(BUDGET), None);
    pool.reuse_subtree(&first.choice);
    game.ply = 1;

    // Long enough for every worker to arm itself and grow a tree worth
    // pooling, short enough that none of them spends its budget.
    fuse.store(300, Relaxed);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        pool.search(&game, &fuse, 0, &config(BUDGET), None);
    }));
    assert!(outcome.is_err(), "the fuse must burn out mid-search");

    fuse.store(u32::MAX, Relaxed);
    game.ply = 2;
    // Deliberately no reuse_subtree call here.
    let next = pool.search(&game, &fuse, 0, &config(BUDGET), None);

    assert_eq!(
        next.reused_iterations, 0,
        "an abandoned worker carried the previous position's tree into the merge"
    );
    assert_eq!(next.root_visits, BUDGET * THREADS as u32);
    assert!(
        next.choice < 3,
        "chose {}, which is not one of this position's choices",
        next.choice
    );
}

/// `Game::advance` may fast-forward past decisions the tree does not model, but
/// not past the one being searched: the root player and the answer list are
/// read before it and the tree is built after it, so consuming the root
/// decision returns somebody else's move.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "Game::advance moved the root past the decision")]
fn advance_may_not_consume_the_root_decision() {
    use mcts::rand_core::Rng;
    use mcts::{Game, Status};

    #[derive(Clone, Default)]
    struct SkipsTheRoot {
        ply: u8,
    }

    impl Game for SkipsTheRoot {
        type Choice = usize;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        fn status(&self, _: &()) -> Status<[f64; 2]> {
            if self.ply >= 2 {
                Status::Terminal([0.5, 0.5])
            } else {
                Status::Active { player: self.ply }
            }
        }

        fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
            out.extend([0, 1]);
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &usize, _: &mut R) {
            self.ply += 1;
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
            self.ply = 2;
            [0.5, 0.5]
        }

        fn new_buffer(&self) -> Self {
            self.clone()
        }

        fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
            dest.clone_from(self);
        }

        fn advance<R: Rng + ?Sized>(&mut self, _: &(), _: &mut (), _: u8, _: &mut R) {
            self.ply = self.ply.max(1);
        }
    }

    let game = SkipsTheRoot::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(50), None, &mut rng(2));
}

/// Same player, different decision — the case a side model is likeliest to
/// reach, and the one a status comparison alone cannot see. The tree's root
/// children are this position's moves, so an `advance` that resolves the root
/// ply returns one of the *next* position's.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "on a different decision")]
fn advance_may_not_swap_the_root_decision_for_the_next_one() {
    use mcts::rand_core::Rng;
    use mcts::{Game, Status};

    #[derive(Clone, Default)]
    struct ResolvesTheRootPly {
        ply: u8,
    }

    impl Game for ResolvesTheRootPly {
        type Choice = usize;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        fn status(&self, _: &()) -> Status<[f64; 2]> {
            if self.ply >= 2 {
                Status::Terminal([0.5, 0.5])
            } else {
                Status::Active { player: 1 }
            }
        }

        fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
            match self.ply {
                0 => out.extend([0, 1, 2]),
                _ => out.extend([10, 11, 12]),
            }
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &usize, _: &mut R) {
            self.ply += 1;
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
            self.ply = 2;
            [0.5, 0.5]
        }

        fn new_buffer(&self) -> Self {
            self.clone()
        }

        fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
            dest.clone_from(self);
        }

        fn advance<R: Rng + ?Sized>(&mut self, _: &(), _: &mut (), _: u8, _: &mut R) {
            self.ply = self.ply.max(1);
        }
    }

    let game = ResolvesTheRootPly::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 1, &config(50), None, &mut rng(2));
}

/// A side model that resolves a hidden opponent decision does not resolve it in
/// every world: the root decision can survive one determinization and be
/// consumed by the next. Checking a single iteration would clear such a game
/// whenever the first world sampled happens to behave.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "on a different decision")]
fn advance_may_not_consume_the_root_decision_in_only_some_worlds() {
    use mcts::rand_core::Rng;
    use mcts::{Game, Status};

    #[derive(Clone, Default)]
    struct ResolvesTheRootPlyLater {
        ply: u8,
    }

    impl Game for ResolvesTheRootPlyLater {
        type Choice = usize;
        type Rewards = [f64; 2];
        type Context = ();
        /// Advance calls so far, standing in for the hidden state a side model
        /// would consult.
        type Side = u32;

        fn status(&self, _: &()) -> Status<[f64; 2]> {
            if self.ply >= 2 {
                Status::Terminal([0.5, 0.5])
            } else {
                Status::Active { player: 1 }
            }
        }

        fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
            match self.ply {
                0 => out.extend([0, 1, 2]),
                _ => out.extend([10, 11, 12]),
            }
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &usize, _: &mut R) {
            self.ply += 1;
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
            self.ply = 2;
            [0.5, 0.5]
        }

        fn new_buffer(&self) -> Self {
            self.clone()
        }

        fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
            dest.clone_from(self);
        }

        fn advance<R: Rng + ?Sized>(&mut self, _: &(), side: &mut u32, _: u8, _: &mut R) {
            *side += 1;
            if *side > 1 {
                self.ply = self.ply.max(1);
            }
        }
    }

    let game = ResolvesTheRootPlyLater::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 1, &config(50), None, &mut rng(2));
}

/// The root check reads the determinized state across `advance`, not against
/// the state the caller handed in, because those two are not required to agree:
/// a game whose terminality depends on hidden information samples worlds that
/// are already over. An iteration scores such a world and moves on, and blaming
/// `Game::advance` for it points at a method the game need not have written.
#[test]
fn a_determinization_may_sample_a_world_that_is_already_over() {
    use mcts::rand_core::Rng;
    use mcts::{Game, Status};

    #[derive(Clone, Default)]
    struct MaybeOver {
        ply: u8,
        over: bool,
    }

    impl Game for MaybeOver {
        type Choice = usize;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        fn status(&self, _: &()) -> Status<[f64; 2]> {
            if self.over || self.ply >= 2 {
                Status::Terminal([0.5, 0.5])
            } else {
                Status::Active { player: 0 }
            }
        }

        fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
            out.extend([0, 1]);
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &usize, _: &mut R) {
            self.ply += 1;
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
            self.ply = 2;
            [0.5, 0.5]
        }

        fn new_buffer(&self) -> Self {
            self.clone()
        }

        // One world in four is over before the search has played anything.
        fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, rng: &mut R) {
            dest.clone_from(self);
            dest.over = rng.next_u64() & 3 == 0;
        }
    }

    let game = MaybeOver::default();
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &config(200), None, &mut rng(4));
    assert!(result.choice < 2, "chose {}", result.choice);
    assert_eq!(result.root_visits, 200);
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

/// A one-ply root whose third choice only appears in a determinization once
/// `UNLOCK_AT` of them have been drawn — a choice the search cannot see yet,
/// which is the ordinary shape of an information set when the card that permits
/// a move is only sometimes in the deck's top half.
///
/// The real position holds all three throughout. It has to: the search may only
/// answer with a move the player actually has, so a fixture whose late choice
/// were illegal at the root would be asking for the answer to be filtered out
/// rather than proved.
#[derive(Clone)]
struct LateChoice {
    drawn: std::rc::Rc<std::cell::Cell<u32>>,
    unlocked: bool,
    played: Option<usize>,
}

impl LateChoice {
    const UNLOCK_AT: u32 = 1_024;
    const PAYOFFS: [f64; 3] = [0.9, 0.0, 1.0];

    fn new() -> Self {
        Self {
            drawn: std::rc::Rc::new(std::cell::Cell::new(0)),
            unlocked: true,
            played: None,
        }
    }
}

impl mcts::Game for LateChoice {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> mcts::Status<[f64; 2]> {
        match self.played {
            Some(choice) => {
                let payoff = Self::PAYOFFS[choice];
                mcts::Status::Terminal([payoff, 1.0 - payoff])
            }
            None => mcts::Status::Active { player: 0 },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend([0, 1]);
        if self.unlocked {
            out.push(2);
        }
    }

    fn apply_choice<R: mcts::rand_core::Rng + ?Sized>(
        &mut self,
        _: &(),
        choice: &usize,
        _: &mut R,
    ) {
        self.played = Some(*choice);
    }

    fn rollout<R: mcts::rand_core::Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        if self.played.is_none() {
            let width: u64 = if self.unlocked { 3 } else { 2 };
            let choice = ((rng.next_u64() as u128 * width as u128) >> 64) as usize;
            self.apply_choice(&(), &choice, rng);
        }
        match self.status(&()) {
            mcts::Status::Terminal(rewards) => rewards,
            _ => unreachable!("one choice ends the game"),
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: mcts::rand_core::Rng + ?Sized>(
        &self,
        dest: &mut Self,
        _: &(),
        _: u8,
        _: &mut R,
    ) {
        dest.clone_from(self);
        self.drawn.set(self.drawn.get() + 1);
        dest.unlocked = self.drawn.get() > Self::UNLOCK_AT;
    }
}

/// Runs one early-terminating search, checks any proof it reports against the
/// conserved-visit bound — one iteration adds one visit to one child, so a
/// child `gap` behind needs `gap` iterations to draw level — and reports
/// whether it proved at all, so a caller can tell a bound that held from one
/// that was never tested.
///
/// That bound is no longer what the crate proves; it is now a *consequence* of
/// what the crate proves, which is stricter. `settled` fires only when the
/// leader has cleared `rank::MIN_EVIDENCE` selections and every rival is short
/// of it by more than the iterations left, so the runner-up holds fewer than
/// `MIN_EVIDENCE - remaining` visits and the leader at least `MIN_EVIDENCE` —
/// which is the gap below and then some. Checking the weaker statement is
/// deliberate: it is the one a maintainer can restate from the design without
/// reading `rank.rs`, and any proof that breaks it is wrong under either rule.
fn proved_early<G>(name: &str, game: &G, budget: u32, seed: u64) -> bool
where
    G: mcts::Game<Choice = usize, Context = ()>,
{
    let cfg = Config {
        early_termination: true,
        ..config(budget)
    };
    let mut searcher = Searcher::new(game);
    let result = searcher.search(game, &(), 0, &cfg, None, &mut rng(seed));
    if result.stop_reason != StopReason::Proven {
        return false;
    }

    let mut visits: Vec<u32> = searcher
        .tree()
        .expect("a search leaves a tree")
        .children()
        .iter()
        .map(|child| child.visits())
        .collect();
    visits.sort_unstable_by(|a, b| b.cmp(a));
    let gap = visits[0] - visits.get(1).copied().unwrap_or(0);
    let remaining = budget - result.root_visits;
    assert!(
        gap > remaining,
        "{name} seed {seed}: proved after {} of {budget} iterations with the leader \
         only {gap} visits clear and {remaining} iterations left to spend",
        result.root_visits
    );
    true
}

/// Nothing weaker than the conserved-visit bound may be stamped
/// `StopReason::Proven`. A simulation of how the remaining iterations would
/// probably be spent is a guess about the future, and the guess this crate used
/// to make was wrong in both directions: it paid every challenger the maximum
/// reward at once, so challengers competed with each other for a budget the
/// question asks about one at a time, and it modelled only the children that
/// existed when it ran.
///
/// The `proofs > 0` guard is load-bearing now in a way it was not: the surviving
/// proof needs every rival locked under the evidence bar, so a fixture whose
/// children are all well sampled proves nothing and this test would compare
/// nothing.
#[test]
fn a_proof_is_never_stronger_than_the_visit_gap() {
    const BUDGET: u32 = 5_000;

    let mut proofs = 0;
    for seed in [1u64, 7, 21] {
        proofs += u32::from(proved_early(
            "count_to_three",
            &CountToThree::new(),
            BUDGET,
            seed,
        ));
        proofs += u32::from(proved_early(
            "trap",
            &GameTree::minimal_trap(),
            BUDGET,
            seed,
        ));
        proofs += u32::from(proved_early("wide200", &GameTree::wide(200), BUDGET, seed));
        proofs += u32::from(proved_early(
            "wide_two_ply40",
            &GameTree::wide_two_ply(40),
            BUDGET,
            seed,
        ));
    }
    assert!(
        proofs > 0,
        "no search proved anything, so this test compares nothing"
    );
}

/// Spending the budget is not a proof. `settled` is asked after the iteration
/// that brings the root up to `target`, and answering "yes, nothing can change"
/// there used to break the loop before its own budget test could — making
/// `StopReason::Budget` unreachable for every sequential search with early
/// termination on, and `Proven` a false positive on every full-budget one.
#[test]
fn a_spent_budget_reports_budget_not_a_proof() {
    // Three choices that all pay the same, so no gap ever opens and there is
    // nothing here to prove.
    let game = AlwaysWin { ply: 0 };
    let cfg = Config {
        early_termination: true,
        ..config(400)
    };
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 1, &cfg, None, &mut rng(1));

    assert_eq!(result.root_visits, 400, "the search must reach its budget");
    assert_eq!(result.stop_reason, StopReason::Budget);
}

/// A child that does not exist yet is still a challenger, and nothing in the
/// tree bounds it: created with `n` iterations left it can spend all of them,
/// clear the evidence bar, and be ranked on a mean the counts say nothing about.
/// A scan over the children present today cannot see it, and used to prove the
/// answer settled on the last iteration before the real winner appeared. The
/// `complete` flag `settled` takes is the refusal that covers this.
#[test]
fn a_choice_the_search_has_not_seen_yet_is_not_proven_away() {
    const BUDGET: u32 = 2_200;

    let answer = |early_termination| {
        let game = LateChoice::new();
        let mut searcher = Searcher::new(&game);
        let cfg = Config {
            early_termination,
            ..config(BUDGET)
        };
        searcher.search(&game, &(), 0, &cfg, None, &mut rng(1))
    };

    let full = answer(false);
    assert_eq!(
        full.choice, 2,
        "the fixture must be won by the late choice, or it tests nothing"
    );

    let stopped = answer(true);
    assert_eq!(
        stopped.choice, full.choice,
        "early termination answered {} after {} of {BUDGET} iterations, stopping for \
         {:?}, where the full budget answers {}",
        stopped.choice, stopped.root_visits, stopped.stop_reason, full.choice
    );
    assert!(
        stopped.root_visits > LateChoice::UNLOCK_AT,
        "the search stopped before the late choice was ever offered"
    );
}

/// A worker proves something about its own tree: that the argmax of *those*
/// visits cannot be overtaken in the iterations *it* had left. The merged
/// answer is an argmax over statistics pooled from every worker, which no
/// worker's proof ever saw, and a worker that stopped short contributed fewer
/// visits to the pool than it would have. So the merge reports the budget.
#[cfg(feature = "parallel")]
#[test]
fn a_pooled_merge_does_not_inherit_a_workers_proof() {
    use mcts::RootParallel;

    const THREADS: usize = 4;
    const BUDGET: u32 = 5_000;

    let game = GameTree::wide(200);
    let cfg = Config {
        early_termination: true,
        ..config(BUDGET)
    };

    // One worker is the whole tree the answer is read off, so its proof stands.
    // It is also what makes the pooled assertion below mean anything: without
    // it, a pool that reported `Budget` might simply never have proved.
    let mut solo = RootParallel::new(1, &game, |worker| rng(worker as u64 + 1));
    let solo_result = solo.search(&game, &(), 0, &cfg, None);
    assert_eq!(solo_result.stop_reason, StopReason::Proven);

    let mut pool = RootParallel::new(THREADS, &game, |worker| rng(worker as u64 + 1));
    let pooled = pool.search(&game, &(), 0, &cfg, None);
    assert_eq!(
        pooled.stop_reason,
        StopReason::Budget,
        "a {THREADS}-worker merge reported {:?} after {} iterations",
        pooled.stop_reason,
        pooled.iterations_used
    );
    assert!(
        pooled.iterations_used < BUDGET * THREADS as u32,
        "no worker stopped early, so nothing was merged over a truncated tree"
    );
}
