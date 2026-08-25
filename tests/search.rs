mod common;

use common::{
    AlwaysWin, CountToThree, GameTree, PriorTrap, RotatingThree, TreeNode, ROTATING_PAYOFFS,
};
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

/// `max^n` backup credits a node with the reward of the player who moved into
/// it, for any number of players — which is the claim `lib.rs` sells as "nothing
/// assumes two players or zero sum". Nothing else in this suite can see it above
/// player index 1: every other sequential fixture is two-player, so
/// `rewards.reward(node.player)` and `rewards.reward(node.player.min(1))` are
/// the same function, and an `assert!(node.player < 2)` at either backup site
/// never fires across the whole suite.
///
/// Checked at the nodes rather than through the move that comes back: this game
/// is small enough that the search opens all of it and solves it whatever the
/// interior means say, so the returned choice is no evidence about the backup.
#[test]
fn max_n_backup_credits_the_third_player_too() {
    let game = RotatingThree::new();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(4_000), None, &mut rng(4));

    let mut leaves = 0;
    check_rotating(searcher.tree().unwrap(), 0, 0, &mut leaves);
    assert_eq!(
        leaves,
        ROTATING_PAYOFFS.len(),
        "the search opened {leaves} of the eight leaves, so the players it did not \
         reach are unchecked"
    );
}

/// Walks a [`RotatingThree`] tree, checking every node's reward player and every
/// leaf's accumulator. A leaf is terminal, so what it was paid is not an average
/// over anything: its cumulative reward is its visit count times one number, and
/// which number says which player the backup credited.
fn check_rotating(node: &Node<usize>, depth: usize, path: usize, leaves: &mut usize) {
    assert_eq!(
        node.reward_player(),
        RotatingThree::PLAYER_AT_DEPTH[depth],
        "the node at depth {depth} reached by {path:0depth$b} is stamped for player {}",
        node.reward_player()
    );
    if depth == RotatingThree::PLAYER_AT_DEPTH.len() - 1 {
        *leaves += 1;
        let payoffs = ROTATING_PAYOFFS[path];
        assert!(node.visits() > 0, "leaf {path:03b} was never visited");
        assert!(
            (node.cumulative_reward() - node.visits() as f64 * payoffs[2]).abs() < 1e-9,
            "leaf {path:03b} accumulated {} over {} visits, which is {} a visit; it is \
             stamped for player 2 and pays {payoffs:?}",
            node.cumulative_reward(),
            node.visits(),
            node.mean_reward()
        );
        return;
    }
    for child in node.children() {
        let choice = *child
            .edge()
            .choice()
            .expect("a sequential root's children all carry a choice");
        check_rotating(child, depth + 1, path << 1 | choice, leaves);
    }
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

/// A config the pool refuses is the one exit that skips the workers entirely,
/// so it is the one exit that has to disarm them by hand.
///
/// `RootParallel::search` names an empty reward range on its own thread rather
/// than as N threads panicking at once — which means no worker's
/// `Searcher::search` runs, and it is that call which consumes a searcher's
/// arming. Refusing before disarming left every retained tree armed for a
/// position the pool never searched.
#[cfg(feature = "parallel")]
#[test]
fn a_pooled_search_refused_for_its_config_disarms_every_worker() {
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

    let empty_range = Config {
        min_reward: 1.0,
        max_reward: 1.0,
        ..config(BUDGET)
    };
    let unburned = fuse.load(Relaxed);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        pool.search(&game, &fuse, 0, &empty_range, None);
    }));
    assert!(outcome.is_err(), "an empty reward range must be refused");
    assert_eq!(
        fuse.load(Relaxed),
        unburned,
        "the refusal must come before any worker touches the game"
    );

    game.ply = 2;
    // Deliberately no reuse_subtree call here.
    let next = pool.search(&game, &fuse, 0, &config(BUDGET), None);

    assert_eq!(
        next.reused_iterations, 0,
        "the refused config left the previous position's tree armed"
    );
    assert_eq!(next.root_visits, BUDGET * THREADS as u32);
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

/// The whole claim of [`Config::early_termination`]: it buys iterations back
/// and it does not change the move.
///
/// Both halves have to be asserted, and one of them is not free. `choice ==
/// full` is a tautology on its own — with the flag inert the two searches are
/// bit-identical runs of the same seed — and `root_visits <= BUDGET` cannot
/// fail, because the budget is the ceiling. So this also pins that the search
/// really stopped: `Proven` and a visit count strictly under the budget, which
/// is what a build with early termination compiled out cannot produce. The
/// fixture proves at 380 of 400, twenty short of `rank::MIN_EVIDENCE`.
#[test]
fn early_termination_does_not_change_the_answer() {
    const BUDGET: u32 = 400;
    let game = GameTree::minimal_trap();

    let full = {
        let mut searcher = Searcher::new(&game);
        searcher
            .search(&game, &(), 0, &config(BUDGET), None, &mut rng(21))
            .choice
    };

    let stopped = {
        let mut searcher = Searcher::new(&game);
        let cfg = Config {
            early_termination: true,
            ..config(BUDGET)
        };
        searcher.search(&game, &(), 0, &cfg, None, &mut rng(21))
    };

    assert_eq!(stopped.choice, full);
    assert_eq!(
        stopped.stop_reason,
        StopReason::Proven,
        "the search spent {} of its {BUDGET} iterations and stopped for {:?}, so \
         nothing here was proved and the answer above is the same run twice",
        stopped.root_visits,
        stopped.stop_reason
    );
    assert!(
        stopped.root_visits < BUDGET,
        "the search reported a proof after spending its whole budget"
    );
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

/// A reward range with nothing in it is not a scale, and every consumer of the
/// declared range reads it as one: regret matching normalizes every payoff it
/// ever sees to a constant 0.5 there and its regrets carry no information at
/// all, decoupled UCB1 measures a tie tolerance against a width of zero, and
/// the degenerate empty-choice node has no in-range value to score. Measured on
/// biased rock-paper-scissors, `[0.5, 0.5]` leaves the extracted strategy
/// exploitable for 0.156 against 0.083 for playing uniformly at random — worse
/// than not searching. It is refused before any of the config is read.
#[test]
#[should_panic(expected = "declares the reward range [0.5, 0.5], which is empty")]
fn an_empty_reward_range_is_refused() {
    let game = AlwaysWin { ply: 0 };
    let cfg = Config {
        min_reward: 0.5,
        max_reward: 0.5,
        ..config(10)
    };
    Searcher::new(&game).search(&game, &(), 1, &cfg, None, &mut rng(1));
}

/// An inverted range takes the same branch bit for bit, so it is refused by the
/// same assertion rather than being read as the range its author meant.
#[test]
#[should_panic(expected = "declares the reward range [1, 0], which is empty")]
fn an_inverted_reward_range_is_refused() {
    let game = AlwaysWin { ply: 0 };
    let cfg = Config {
        min_reward: 1.0,
        max_reward: 0.0,
        ..config(10)
    };
    Searcher::new(&game).search(&game, &(), 1, &cfg, None, &mut rng(1));
}

/// A game that pays strictly inside `[MIN_PAY, MAX_PAY]`, with one root choice
/// leading to a node where the player to move has no legal choice at all.
///
/// The descent cannot continue there, so the iteration is scored with a payoff
/// the crate fabricates rather than one the game produced — and that payoff is
/// credited to every node on the path. Fabricating a zero put a number outside
/// the declared range into all of them.
#[derive(Clone, Default)]
struct DeadEnd {
    stuck: bool,
    resolved: bool,
}

impl DeadEnd {
    /// The declared floor. Above zero, which is the whole point of the fixture.
    const MIN_PAY: f64 = 1.0;
    const MAX_PAY: f64 = 2.0;
    /// What every line that reaches a terminal state pays both players.
    const PAYOUT: f64 = 1.5;
    /// The root choice that leads nowhere.
    const STUCK: usize = 0;
}

impl mcts::Game for DeadEnd {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> mcts::Status<[f64; 2]> {
        if self.resolved {
            mcts::Status::Terminal([Self::PAYOUT; 2])
        } else {
            mcts::Status::Active { player: 0 }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        if !self.stuck {
            out.extend([Self::STUCK, 1]);
        }
    }

    fn apply_choice<R: mcts::rand_core::Rng + ?Sized>(
        &mut self,
        _: &(),
        choice: &usize,
        _: &mut R,
    ) {
        if *choice == Self::STUCK {
            self.stuck = true;
        } else {
            self.resolved = true;
        }
    }

    fn rollout<R: mcts::rand_core::Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        [Self::PAYOUT; 2]
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
    }
}

fn dead_end_config() -> Config {
    Config {
        min_reward: DeadEnd::MIN_PAY,
        max_reward: DeadEnd::MAX_PAY,
        ..config(200)
    }
}

/// Every reward in the tree is one the game could have paid.
///
/// The value the crate fabricates for a node it cannot descend from is the
/// declared floor, so it lands inside the range the caller promised. A zero
/// there is a payoff no game paying in `[1, 2]` can produce, and it does not
/// stay local: it is credited to every node on the path, drags their means
/// below the range they are read against, and — under a simultaneous ancestor —
/// trips the reward-range assertion in a message that blames the game for it.
#[cfg(not(debug_assertions))]
#[test]
fn a_degenerate_node_scores_inside_the_declared_reward_range() {
    let game = DeadEnd::default();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &dead_end_config(), None, &mut rng(3));

    let root = searcher.tree().expect("the search retained its tree");
    let stuck = root
        .children()
        .iter()
        .find(|child| child.edge().choice() == Some(&DeadEnd::STUCK))
        .expect("the dead-end choice is a root child");
    assert!(
        stuck.visits() > 1,
        "the dead end was never descended into, so nothing was fabricated"
    );
    for (name, mean) in [
        ("the root", root.mean_reward()),
        ("the dead end", stuck.mean_reward()),
    ] {
        assert!(
            (DeadEnd::MIN_PAY..=DeadEnd::MAX_PAY).contains(&mean),
            "{name} holds a mean reward of {mean}, outside the declared \
             [{}, {}] range this game pays in",
            DeadEnd::MIN_PAY,
            DeadEnd::MAX_PAY
        );
    }
}

/// The same position in a debug build, where a node the descent cannot continue
/// from is a bug worth naming at the node it happened at. It used to be
/// reported, if at all, by the reward-range assertion at a simultaneous
/// ancestor accusing the game of a payoff the search had fabricated.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the descent cannot continue at this node")]
fn a_degenerate_sequential_node_is_a_debug_assertion() {
    let game = DeadEnd::default();
    Searcher::new(&game).search(&game, &(), 0, &dead_end_config(), None, &mut rng(3));
}

/// A prior is a statement about the *mover's* prospects, and the search adds it
/// to a mean it keeps in the mover's currency.
///
/// [`common::PriorTrap`]'s hook returns the successor's game-theoretic value to
/// the player who moved into it. Evaluated for the searching player instead —
/// which is what the crate used to pass — player 1's node is told that the
/// terminal paying player 1 nothing is the good one, player 1 obliges, and the
/// trap's mean climbs toward the 1.0 it would pay player 0 if the opponent
/// cooperated. This is the optimistic-opponent error the crate refuses to make
/// at a simultaneous root, made silently at every sequential opponent node.
#[test]
fn the_prior_is_read_in_the_movers_currency() {
    const BUDGET: u32 = 500;

    for seed in 1..=16u64 {
        let game = PriorTrap::new();
        let cfg = Config {
            iterations: BUDGET,
            progressive_bias_weight: 100.0,
            ..config(BUDGET)
        };
        let mut searcher = Searcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(seed));

        assert_eq!(
            result.choice,
            PriorTrap::SAFE,
            "seed {seed} walked into the trap, which player 1 punishes for 0.0 \
             against SAFE's 0.3"
        );

        let trap = searcher
            .tree()
            .expect("a search leaves a tree")
            .children()
            .iter()
            .find(|child| child.edge().choice() == Some(&PriorTrap::TRAP))
            .expect("the trap is a root child");
        let visits = |choice: usize| {
            trap.children()
                .iter()
                .find(|child| child.edge().choice() == Some(&choice))
                .map_or(0, |child| child.visits())
        };
        assert!(
            visits(1) > visits(0),
            "seed {seed}: player 1 spent {} visits on the reply paying them 0.0 and \
             {} on the reply paying them 1.0",
            visits(0),
            visits(1)
        );
    }
}

/// The progressive-bias term has to be observable, or nothing here is a test.
///
/// It was not: no fixture in the repo overrode `Game::heuristic_bias`, so
/// deleting the third term of `ucb_raw` outright left the whole suite green,
/// golden included. An honest prior valuing the trap at 0.0 and the safe move
/// at 0.3 must spend less of the budget on the trap than the same search with
/// the weight at zero — the same search in every other respect, down to the rng
/// draws, since evaluating the prior consumes none.
#[test]
fn a_non_zero_bias_weight_moves_the_search() {
    const BUDGET: u32 = 500;

    let trap_visits = |weight: f64| {
        let game = PriorTrap::new();
        let cfg = Config {
            iterations: BUDGET,
            progressive_bias_weight: weight,
            ..config(BUDGET)
        };
        let mut searcher = Searcher::new(&game);
        searcher.search(&game, &(), 0, &cfg, None, &mut rng(1));
        searcher
            .tree()
            .expect("a search leaves a tree")
            .children()
            .iter()
            .find(|child| child.edge().choice() == Some(&PriorTrap::TRAP))
            .expect("the trap is a root child")
            .visits()
    };

    let unbiased = trap_visits(0.0);
    let biased = trap_visits(100.0);
    assert!(
        biased < unbiased,
        "the prior changed nothing: the trap took {biased} of {BUDGET} iterations \
         with the weight at 100 and {unbiased} with it at zero"
    );
}

/// A non-finite knob does not mis-tune the search, it switches it off: every
/// UCB value becomes NaN, NaN loses every comparison `select` makes, so `select`
/// answers `None` at the first fully-opened node and the rest of the budget
/// grows nothing. Measured before the refusal: 2 000 iterations on
/// `minimal_trap` left a three-node tree and returned the losing choice.
///
/// `Config` derives `serde` and TOML, YAML and JSON-with-a-divide all spell
/// `nan` and `inf`, so this arrives without anyone typing `f64::NAN`.
#[test]
#[should_panic(expected = "Config::exploration_constant is NaN")]
fn a_non_finite_exploration_constant_is_refused() {
    let game = GameTree::minimal_trap();
    let cfg = Config {
        exploration_constant: f64::NAN,
        ..config(2000)
    };
    Searcher::new(&game).search(&game, &(), 0, &cfg, None, &mut rng(1));
}

/// `NaN * 0.0` is `NaN`, so the bias weight reaches every child's score even in
/// a game that never implements the hook.
#[test]
#[should_panic(expected = "Config::progressive_bias_weight is inf")]
fn a_non_finite_bias_weight_is_refused() {
    let game = GameTree::minimal_trap();
    let cfg = Config {
        progressive_bias_weight: f64::INFINITY,
        ..config(2000)
    };
    Searcher::new(&game).search(&game, &(), 0, &cfg, None, &mut rng(1));
}

/// A NaN bound is caught as a non-finite knob rather than as an empty range:
/// the range check is `max_reward <= min_reward`, and every comparison against
/// a NaN is false, so an empty-range test alone would let this through.
#[test]
#[should_panic(expected = "Config::min_reward is NaN")]
fn a_non_finite_reward_bound_is_refused() {
    let game = GameTree::minimal_trap();
    let cfg = Config {
        min_reward: f64::NAN,
        ..config(2000)
    };
    Searcher::new(&game).search(&game, &(), 0, &cfg, None, &mut rng(1));
}

/// The simultaneous knobs go through the same gate. `duct_exploration` is dead
/// at a sequential node, but a config is refused for what it declares, not for
/// what this position happens to read.
#[test]
#[should_panic(expected = "Config::simultaneous.duct_exploration is -inf")]
fn a_non_finite_simultaneous_knob_is_refused() {
    let game = GameTree::minimal_trap();
    let mut cfg = config(2000);
    cfg.simultaneous.duct_exploration = f64::NEG_INFINITY;
    Searcher::new(&game).search(&game, &(), 0, &cfg, None, &mut rng(1));
}

/// A node whose bad children are legal in only one determinization in
/// [`RareTraps::PERIOD`], reached through a root ply the game ignores.
///
/// The fixture for the rule that makes this Information Set MCTS rather than
/// UCT: below the root, a child's exploration term is measured against how often
/// that child was *available*, not against how often its parent was visited. The
/// two denominators differ by exactly the legality rate, so a fixture whose
/// `determinize_into` is a clone — which is every other sequential fixture here
/// — cannot tell them apart at all, and neither can the golden fingerprint.
///
/// Player 0 moves throughout. The first ply is a coin the game discards; it is
/// here because a root offering one choice is a forced move, and because the
/// root is the one level that deliberately scores its children against its own
/// visit count. Both of its moves lead to the same trap node, so neither can be
/// abandoned and the measurement happens one level down, where the availability
/// rule lives.
///
/// The trap node offers [`RareTraps::GOOD`], worth [`RareTraps::GOOD_PAYOFF`]
/// and legal in every world, and [`RareTraps::TRAPS`] traps worth nothing, each
/// legal in one world in [`RareTraps::PERIOD`]. UCB1 keeps returning to a
/// worthless child until its exploration term falls under the gap, which takes
/// about `(c / gap)^2 * ln(denominator)` selections — so what the traps cost is
/// a direct reading of which denominator the search used.
#[derive(Clone)]
struct RareTraps {
    /// Determinizations drawn so far, shared with every copy, so which traps a
    /// world offers is a fixed cycle rather than a draw: a trap's availability
    /// is then `visits / PERIOD` and not a sample of it.
    drawn: std::rc::Rc<std::cell::Cell<u32>>,
    world: u32,
    stage: RareStage,
}

#[derive(Clone, Copy)]
enum RareStage {
    Coin,
    Traps,
    Over(f64),
}

impl RareTraps {
    const GOOD: usize = 0;
    const TRAPS: usize = 400;
    const PERIOD: u32 = 50;
    const GOOD_PAYOFF: f64 = 0.6;

    fn new() -> Self {
        Self {
            drawn: std::rc::Rc::new(std::cell::Cell::new(0)),
            world: 0,
            stage: RareStage::Coin,
        }
    }

    fn offers(&self, trap: usize) -> bool {
        (self.world + trap as u32).is_multiple_of(Self::PERIOD)
    }
}

impl mcts::Game for RareTraps {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> mcts::Status<[f64; 2]> {
        match self.stage {
            RareStage::Over(payoff) => mcts::Status::Terminal([payoff, 1.0 - payoff]),
            _ => mcts::Status::Active { player: 0 },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        match self.stage {
            RareStage::Coin => out.extend([0, 1]),
            RareStage::Traps => {
                out.push(Self::GOOD);
                out.extend((1..=Self::TRAPS).filter(|&trap| self.offers(trap)));
            }
            RareStage::Over(_) => {}
        }
    }

    fn apply_choice<R: mcts::rand_core::Rng + ?Sized>(
        &mut self,
        _: &(),
        choice: &usize,
        _: &mut R,
    ) {
        self.stage = match (self.stage, *choice) {
            (RareStage::Coin, _) => RareStage::Traps,
            (RareStage::Traps, Self::GOOD) => RareStage::Over(Self::GOOD_PAYOFF),
            (RareStage::Traps, _) => RareStage::Over(0.0),
            (over, _) => over,
        };
    }

    fn rollout<R: mcts::rand_core::Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let mut choices = Vec::new();
        loop {
            if let RareStage::Over(payoff) = self.stage {
                return [payoff, 1.0 - payoff];
            }
            choices.clear();
            self.choices_into(&(), &mut choices);
            let k = ((rng.next_u64() as u128 * choices.len() as u128) >> 64) as usize;
            let choice = choices[k];
            self.apply_choice(&(), &choice, rng);
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
        dest.world = self.drawn.get();
    }
}

/// Below the root, the exploration term's denominator is the child's own
/// availability count, not its parent's visit count. Deleting that — scoring
/// every child against `ln(parent visits)` — is deleting Information Set MCTS
/// from the sequential path, and it leaves the rest of this suite, the golden
/// fingerprint included, entirely green: no other sequential fixture's
/// `determinize_into` varies the legal set, so the two denominators are the same
/// number everywhere else.
///
/// The reading is how much of the budget the traps absorb. UCB1 keeps returning
/// to a child worth nothing while its exploration term covers the gap to the
/// leader, which is `c * sqrt(ln(denominator) / visits) > gap` — so it spends
/// `(c / gap)^2 * ln(denominator)` selections on each of them and then stops.
/// The traps here are legal one world in [`RareTraps::PERIOD`], so the two
/// candidate denominators differ by that factor and the two rules predict a
/// different integer. Measured: three selections per trap under the availability
/// rule and five under the parent's visit count, in every seed.
#[test]
fn a_rarely_legal_child_is_explored_against_its_own_availability() {
    const BUDGET: u32 = 40_000;

    let game = RareTraps::new();
    let mut searcher = Searcher::new(&game);
    searcher.search(&game, &(), 0, &config(BUDGET), None, &mut rng(1));

    // The busier of the two opening moves: they lead to the same position, and
    // which of them the search settles on is not what is being measured.
    let node = searcher
        .tree()
        .expect("a search leaves a tree")
        .children()
        .iter()
        .max_by_key(|child| child.visits())
        .expect("the root has both opening moves");
    let traps: Vec<&Node<usize>> = node
        .children()
        .iter()
        .filter(|child| *child.edge().choice().unwrap() != RareTraps::GOOD)
        .collect();
    assert_eq!(
        traps.len(),
        RareTraps::TRAPS,
        "the node the search built holds {} of the {} traps, so most of them were \
         never offered and there is nothing here to measure",
        traps.len(),
        RareTraps::TRAPS
    );

    // `(c / gap)^2 * ln(denominator)`, rounded up: the selections UCB1 spends on
    // a child worth nothing before its exploration term stops covering the gap
    // to a leader worth `GOOD_PAYOFF`.
    let budget_for = |ln_total: f64| {
        let ratio = config(BUDGET).exploration_constant / RareTraps::GOOD_PAYOFF;
        (ratio * ratio * ln_total).ceil() as u32
    };
    let availability =
        traps.iter().map(|trap| trap.availability()).sum::<u32>() / RareTraps::TRAPS as u32;
    let by_availability = budget_for((availability as f64).ln());
    let by_parent_visits = budget_for((node.visits() as f64).ln());
    assert!(
        by_availability < by_parent_visits,
        "the two denominators predict the same {by_availability} selections a trap, so \
         this fixture cannot tell them apart"
    );

    let spent = traps.iter().map(|trap| trap.visits()).max().unwrap_or(0);
    assert!(
        spent < by_parent_visits,
        "the search spent {spent} selections on a trap legal one world in {}, which is \
         what `ln(parent visits)` = {by_parent_visits} buys rather than what the trap's \
         own {availability} opportunities = {by_availability} do",
        RareTraps::PERIOD
    );
    assert!(
        spent <= by_availability,
        "the search spent {spent} selections on a trap against the {by_availability} its \
         own availability predicts"
    );
}

/// A cancellation flag and the countdown that trips it, handed to the search as
/// one context.
///
/// [`Searcher::search`] takes the flag by reference and polls it once an
/// iteration, so a test that wants it set *during* a search needs something
/// inside the loop to set it. `determinize_into` runs once an iteration, right
/// after the poll, which makes the arithmetic exact: the determinization that
/// empties the fuse is the last one, and the poll at the top of the next
/// iteration is what breaks.
struct Fuse {
    left: std::sync::atomic::AtomicU32,
    cancel: std::sync::atomic::AtomicBool,
}

impl Fuse {
    fn new(iterations: u32) -> Self {
        Self {
            left: std::sync::atomic::AtomicU32::new(iterations),
            cancel: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

/// Three plies of three choices and nothing else, over a [`Fuse`] context.
#[derive(Clone)]
struct Countdown {
    ply: u32,
}

impl mcts::Game for Countdown {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = Fuse;
    type Side = ();

    fn status(&self, _: &Fuse) -> mcts::Status<[f64; 2]> {
        if self.ply >= 3 {
            mcts::Status::Terminal([0.5, 0.5])
        } else {
            mcts::Status::Active { player: 0 }
        }
    }

    fn choices_into(&self, _: &Fuse, out: &mut Vec<usize>) {
        out.extend([0, 1, 2]);
    }

    fn apply_choice<R: mcts::rand_core::Rng + ?Sized>(&mut self, _: &Fuse, _: &usize, _: &mut R) {
        self.ply += 1;
    }

    fn rollout<R: mcts::rand_core::Rng + ?Sized>(&mut self, _: &Fuse, _: &mut R) -> [f64; 2] {
        self.ply = 3;
        [0.5, 0.5]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: mcts::rand_core::Rng + ?Sized>(
        &self,
        dest: &mut Self,
        fuse: &Fuse,
        _: u8,
        _: &mut R,
    ) {
        use std::sync::atomic::Ordering::Relaxed;
        dest.clone_from(self);
        if fuse
            .left
            .fetch_update(Relaxed, Relaxed, |left| left.checked_sub(1))
            .is_ok_and(|left| left == 1)
        {
            fuse.cancel.store(true, Relaxed);
        }
    }
}

/// The cancellation flag is one of the three budget kinds `lib.rs` sells, and
/// every `search` call in this suite used to pass `None` for it —
/// [`StopReason::Cancelled`] was unreachable from the tests.
///
/// The flag is polled at the top of the iteration, so a flag set by the
/// `CUTOFF`-th determinization stops the search at exactly `CUTOFF` iterations,
/// with the whole of a 5 000-iteration budget still unspent.
#[test]
fn a_cancellation_flag_cuts_the_budget_short() {
    const CUTOFF: u32 = 25;
    const BUDGET: u32 = 5_000;

    let fuse = Fuse::new(CUTOFF);
    let game = Countdown { ply: 0 };
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(
        &game,
        &fuse,
        0,
        &config(BUDGET),
        Some(&fuse.cancel),
        &mut rng(3),
    );

    assert_eq!(result.stop_reason, StopReason::Cancelled);
    assert_eq!(
        result.iterations_used, CUTOFF,
        "the flag was set by determinization {CUTOFF} and the poll that reads it is the \
         next iteration's first act"
    );
    assert_eq!(result.root_visits, CUTOFF);
    assert!(
        (0..3).contains(&result.choice),
        "a cancelled search still answers with a legal move, not {}",
        result.choice
    );
}

/// A flag already set completes no iterations at all, which is the exit that
/// leaves the root with no visited child: the answer is then a draw from the
/// position's own choice list rather than a reading of a tree that does not
/// exist.
#[test]
fn a_search_cancelled_before_it_starts_still_answers_legally() {
    let fuse = Fuse::new(u32::MAX);
    fuse.cancel
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let game = Countdown { ply: 0 };
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(
        &game,
        &fuse,
        0,
        &config(5_000),
        Some(&fuse.cancel),
        &mut rng(3),
    );

    assert_eq!(result.stop_reason, StopReason::Cancelled);
    assert_eq!(result.iterations_used, 0);
    assert_eq!(result.root_visits, 0);
    assert_eq!(result.best_visits, 0);
    assert!(
        (0..3).contains(&result.choice),
        "a search that ran no iterations answered {}, which is not one of this \
         position's choices",
        result.choice
    );
}

/// A wall-clock budget is the other half of `Config`'s documented pair, and
/// nothing in this suite ever set one: `time_limit_ms` and
/// [`StopReason::Deadline`] were unreachable from the tests.
///
/// The iteration budget is set far higher than the deadline can reach, so the
/// clock is what stops the search — and a build with the deadline check removed
/// fails this by running the whole of it rather than by hanging.
#[cfg(feature = "time")]
#[test]
fn a_wall_clock_budget_stops_the_search() {
    const BUDGET: u32 = 2_000_000;

    let fuse = Fuse::new(u32::MAX);
    let game = Countdown { ply: 0 };
    let cfg = Config {
        time_limit_ms: Some(20),
        ..config(BUDGET)
    };
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &fuse, 0, &cfg, None, &mut rng(3));

    assert_eq!(result.stop_reason, StopReason::Deadline);
    assert!(
        result.iterations_used > 0,
        "the search stopped before its first iteration, so it measured no clock"
    );
    assert!(
        result.iterations_used < BUDGET,
        "the search spent its whole {BUDGET}-iteration budget, so the deadline never \
         stopped anything"
    );
}

/// A deadline already in the past is the deterministic half of the same knob:
/// the clock is read before the first determinization, so the search completes
/// no iterations and still answers with a move this position holds.
///
/// `iterations: 0` is the configuration `Config::iterations`' doc points at with
/// "zero means run until `time_limit_ms` expires", which leaves the clock as the
/// only thing that can end this loop. The fuse is passed as a watchdog for
/// exactly that reason: a build with the deadline check removed then reports
/// `Cancelled` after 100 000 iterations and fails these assertions, instead of
/// running until someone kills it.
#[cfg(feature = "time")]
#[test]
fn a_deadline_already_past_completes_no_iterations() {
    let fuse = Fuse::new(100_000);
    let game = Countdown { ply: 0 };
    let cfg = Config {
        time_limit_ms: Some(0),
        ..config(0)
    };
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &fuse, 0, &cfg, Some(&fuse.cancel), &mut rng(3));

    assert_eq!(result.stop_reason, StopReason::Deadline);
    assert_eq!(result.iterations_used, 0);
    assert_eq!(result.root_visits, 0);
    assert!(
        (0..3).contains(&result.choice),
        "a search that ran no iterations answered {}, which is not one of this \
         position's choices",
        result.choice
    );
}

/// Every worker's tree is re-rooted, not merely one of them.
///
/// [`mcts::RootParallel::reuse_subtree`] is a loop over the workers, and nothing
/// in this suite ever read the result of it: an empty body, or one that re-roots
/// worker 0 and returns, left the whole suite green. `reused_iterations` cannot
/// catch either — it is one merged scalar, so a single worker carrying its
/// subtree forward already makes it positive — so the claim is asserted where it
/// is made, per tree, against the visits each worker held for that move before
/// the call.
#[cfg(feature = "parallel")]
#[test]
fn pooled_reuse_re_roots_every_worker() {
    use mcts::RootParallel;

    const THREADS: usize = 4;
    const BUDGET: u32 = 300;

    let mut game = three_plies();
    let mut pool = RootParallel::new(THREADS, &game, |worker| rng(worker as u64 + 1));
    let first = pool.search(&game, &(), 0, &config(BUDGET), None);

    let carried: Vec<u32> = pool
        .trees()
        .map(|root| {
            root.children()
                .iter()
                .find(|child| child.edge().choice() == Some(&first.choice))
                .map_or(0, |child| child.visits())
        })
        .collect();
    assert_eq!(carried.len(), THREADS, "a worker finished without a tree");
    assert!(
        carried.iter().all(|&visits| visits > 0),
        "the workers hold {carried:?} visits for the merged answer, so at least one of \
         them has nothing to carry forward and re-rooting it is unobservable"
    );

    pool.reuse_subtree(&first.choice);
    let rooted: Vec<u32> = pool.trees().map(|root| root.visits()).collect();
    assert_eq!(
        rooted, carried,
        "after re-rooting, each worker's root should be the subtree it held for the \
         answer; the pool holds {rooted:?} against the {carried:?} its workers had"
    );

    game.apply(&first.choice);
    let second = pool.search(&game, &(), 0, &config(BUDGET), None);
    assert_eq!(
        second.reused_iterations,
        carried.iter().sum::<u32>(),
        "the merged reuse count is every worker's, not one worker's"
    );
    for (worker, root) in pool.trees().enumerate() {
        assert_eq!(
            root.visits(),
            BUDGET,
            "worker {worker} reached {} of the {BUDGET} visits the budget asks for, \
             counting the ones it inherited",
            root.visits()
        );
    }
}

/// Two plies of ten choices, searched by a pool: wide enough that the workers'
/// seeds send them down different children, which is what lets one named worker
/// be given a budget it has already met.
#[cfg(feature = "parallel")]
const POOL_WIDTH: usize = 10;
#[cfg(feature = "parallel")]
const POOL_THREADS: usize = 4;

/// A pool that has searched once and re-rooted every worker at `choice`,
/// returned with the position it re-rooted into and the visits each worker
/// carried across.
///
/// A search whose budget is that carry is over before the worker holding it
/// runs an iteration — `root.visits >= target` is the first thing the loop
/// asks, ahead of the cancellation flag and the clock — so a budget set to one
/// worker's carry stops that worker for [`StopReason::Budget`] and leaves every
/// other worker to stop for whatever the caller passed in.
#[cfg(feature = "parallel")]
fn pool_carrying(choice: usize) -> (mcts::RootParallel<GameTree, WyRand>, GameTree, Vec<u32>) {
    const FIRST: u32 = 200;

    let mut game = GameTree::wide_two_ply(POOL_WIDTH);
    let mut pool = mcts::RootParallel::new(POOL_THREADS, &game, |worker| rng(worker as u64 + 1));
    pool.search(&game, &(), 0, &config(FIRST), None);
    let carried: Vec<u32> = pool
        .trees()
        .map(|root| {
            root.children()
                .iter()
                .find(|child| child.edge().choice() == Some(&choice))
                .map_or(0, |child| child.visits())
        })
        .collect();
    pool.reuse_subtree(&choice);
    game.apply(&choice);
    (pool, game, carried)
}

/// A root choice `worker` carried as much of as any worker did, and at least
/// one worker carried less of — so a budget of that carry stops `worker` for
/// [`StopReason::Budget`] and leaves somebody else to stop for the caller's
/// flag or clock.
#[cfg(feature = "parallel")]
fn budget_carried_by(worker: usize) -> (usize, u32) {
    (0..POOL_WIDTH)
        .find_map(|choice| {
            let carried = pool_carrying(choice).2;
            let mine = carried[worker];
            let short = carried.iter().any(|&visits| visits < mine);
            (mine > 1 && short && carried.iter().all(|&visits| visits <= mine))
                .then_some((choice, mine))
        })
        .unwrap_or_else(|| {
            panic!("no root choice of {POOL_WIDTH} leaves worker {worker} carrying the most")
        })
}

/// A pooled search reports the most informative reason any worker stopped for.
///
/// Every `search` call in this suite passed `None` for `cancel` and set no
/// `time_limit_ms`, pooled ones included, so `merged_stop_reason`'s `Cancelled`
/// and `Deadline` arms were unreachable: a pool answering `Budget` for a
/// cancelled run would have shipped unnoticed.
///
/// A pool whose workers all stop for the same reason cannot see a ranking at
/// all — `max_by_key` keeps the last maximum, so ranking `Cancelled` level with
/// `Budget` still answers `Cancelled` when every worker said `Cancelled`. So one
/// worker is stopped for `Budget` instead, and which one it is decides what the
/// case pins: with it last, a merge that has stopped ranking the flag above the
/// budget answers `Budget`, and with it first, so does a merge that copies
/// worker 0's reason out.
#[cfg(feature = "parallel")]
#[test]
fn a_pooled_search_reports_the_cancellation_a_worker_stopped_for() {
    use std::sync::atomic::AtomicBool;

    for worker in [POOL_THREADS - 1, 0] {
        let (choice, target) = budget_carried_by(worker);
        let (mut pool, game, carried) = pool_carrying(choice);
        let cancel = AtomicBool::new(true);
        let result = pool.search(&game, &(), 0, &config(target), Some(&cancel));

        assert_eq!(
            result.stop_reason,
            StopReason::Cancelled,
            "worker {worker} carried the whole {target}-visit budget and stopped for it, \
             but the workers that carried less of {choice} stopped for the flag"
        );
        assert_eq!(
            result.iterations_used, 0,
            "a flag set before the search leaves every worker nothing to do"
        );
        assert_eq!(
            result.reused_iterations,
            carried.iter().sum::<u32>(),
            "the merged reuse count is every worker's carry, {carried:?}"
        );
        assert!(
            result.choice < POOL_WIDTH,
            "the pool answered {}, which this position does not offer",
            result.choice
        );
    }
}

/// The clock's half of the same claim, and the same construction: one worker's
/// budget is already met, so it stops for `Budget` while the rest read a
/// deadline that has already passed.
#[cfg(all(feature = "parallel", feature = "time"))]
#[test]
fn a_pooled_search_reports_the_deadline_a_worker_stopped_for() {
    for worker in [POOL_THREADS - 1, 0] {
        let (choice, target) = budget_carried_by(worker);
        let (mut pool, game, _) = pool_carrying(choice);
        let cfg = Config {
            time_limit_ms: Some(0),
            ..config(target)
        };
        let result = pool.search(&game, &(), 0, &cfg, None);

        assert_eq!(
            result.stop_reason,
            StopReason::Deadline,
            "worker {worker} stopped for its budget and the rest for the clock"
        );
        assert_eq!(result.iterations_used, 0);
    }
}

/// A game that pays outside the range it declares.
///
/// Not a contrived one: a heuristic evaluation that is meant to be normalised
/// and is not, or a caller who declares `[0, 1]` and returns a raw score, both
/// land here. The declared range is what `Duct`'s tie tolerance, regret
/// matching's normalisation and the early-termination proof are all measured
/// against, so a payoff outside it is not a slightly-off number, it is a number
/// those three read as a different quantity.
#[derive(Clone, Default)]
struct PaysOutOfRange {
    done: bool,
}

impl PaysOutOfRange {
    const DECLARED_MAX: f64 = 1.0;
    /// Five times the declared ceiling.
    const ACTUAL_PAY: f64 = 5.0;
}

impl mcts::Game for PaysOutOfRange {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> mcts::Status<[f64; 2]> {
        if self.done {
            mcts::Status::Terminal([Self::ACTUAL_PAY; 2])
        } else {
            mcts::Status::Active { player: 0 }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        if !self.done {
            out.extend([0, 1]);
        }
    }

    fn apply_choice<R: mcts::rand_core::Rng + ?Sized>(&mut self, _: &(), _: &usize, _: &mut R) {
        self.done = true;
    }

    fn rollout<R: mcts::rand_core::Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        [Self::ACTUAL_PAY; 2]
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
    }
}

/// The declared range is enforced, not merely believed.
///
/// Without the clamp the root's mean is the 5.0 the game pays, five times the
/// ceiling it declared, and every quantity measured against that range is
/// reading a number off a different scale. The early-termination proof is the
/// sharpest case: it bounds a rival's mean by the declared ceiling, which is
/// only a bound if the accumulators respect it.
#[test]
fn a_reward_outside_the_declared_range_is_clamped_into_it() {
    let game = PaysOutOfRange::default();
    let cfg = Config {
        min_reward: 0.0,
        max_reward: PaysOutOfRange::DECLARED_MAX,
        ..config(200)
    };
    let mut searcher = Searcher::new(&game);
    let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng(1));

    let root = searcher.tree().expect("a search leaves a tree");
    assert!(
        root.mean_reward() <= PaysOutOfRange::DECLARED_MAX,
        "the root's mean is {}, above the declared ceiling of {}",
        root.mean_reward(),
        PaysOutOfRange::DECLARED_MAX
    );
    assert_eq!(
        result.best_mean_reward,
        PaysOutOfRange::DECLARED_MAX,
        "every payoff is above the ceiling, so every clamped one is the ceiling"
    );
}
