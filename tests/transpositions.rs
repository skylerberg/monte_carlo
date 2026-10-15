//! `GraphSearcher`: searching a graph of positions rather than a tree of move
//! sequences.
//!
//! Two kinds of evidence. Where no position is ever reached two ways, the graph
//! search must *be* the tree search — the same answer, the same statistics at
//! every node, to the last bit — which pins every line of the port. Where
//! positions are reached two ways, the graph must hold each once, conserve its
//! counts, stop at repetitions, and refuse a key that reads hidden state.

#[allow(dead_code)]
mod common;

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use common::*;
use mcts::rand_core::SeedableRng;
use mcts::{
    Config, Game, GraphNode, GraphSearcher, Node, RootPolicy, SearchResult, Searcher,
    SimultaneousConfig, Status, StopReason, Transposition,
};
use wyrand::WyRand;

fn assert_same_result<C: PartialEq + Debug>(tree: &SearchResult<C>, graph: &SearchResult<C>) {
    assert_eq!(tree.choice, graph.choice, "choice");
    assert_eq!(
        tree.iterations_used, graph.iterations_used,
        "iterations_used"
    );
    assert_eq!(
        tree.reused_iterations, graph.reused_iterations,
        "reused_iterations"
    );
    assert_eq!(tree.stop_reason, graph.stop_reason, "stop_reason");
    assert_eq!(tree.root_visits, graph.root_visits, "root_visits");
    assert_eq!(tree.best_visits, graph.best_visits, "best_visits");
    assert_eq!(
        tree.best_mean_reward.to_bits(),
        graph.best_mean_reward.to_bits(),
        "best_mean_reward"
    );
}

/// The tree under `tree` and the graph under `graph` hold the same statistics
/// everywhere: every child is a link with the same counts, every visited child
/// is a position with the same totals, and every simultaneous node has the same
/// bandits. Children are compared in creation order, which both build by the
/// same reservoir draws.
fn assert_same_shape<C: Eq + Hash + Clone + Debug, K>(tree: &Node<C>, graph: GraphNode<'_, C, K>) {
    assert_eq!(tree.visits(), graph.visits(), "visits");
    assert_eq!(
        tree.cumulative_reward().to_bits(),
        graph.cumulative_reward().to_bits(),
        "cumulative reward"
    );
    assert_eq!(tree.reward_player(), graph.reward_player(), "reward player");

    match tree.simultaneous_players() {
        None => {
            assert_eq!(graph.simultaneous_players(), None);
            let edges: Vec<_> = graph.edges().collect();
            assert_eq!(tree.children().len(), edges.len(), "children against links");
            for (child, edge) in tree.children().iter().zip(&edges) {
                assert_eq!(child.edge().choice(), Some(edge.choice()));
                assert_eq!(child.visits(), edge.visits(), "link visits");
                assert_eq!(
                    child.times_available(),
                    edge.times_available(),
                    "link availability"
                );
                match edge.target() {
                    Some(target) => assert_same_shape(child, target),
                    None => assert_eq!(child.visits(), 0, "an untaken link has no successor"),
                }
            }
        }
        Some(players) => {
            assert_eq!(graph.simultaneous_players(), Some(players));
            for player in players.iter() {
                let (a, b) = (
                    tree.marginals(player).expect("a participant has marginals"),
                    graph
                        .marginals(player)
                        .expect("a participant has marginals"),
                );
                assert_eq!(a.len(), b.len(), "arms");
                for arm in 0..a.len() {
                    assert_eq!(a.choice(arm), b.choice(arm));
                    assert_eq!(a.visits(arm), b.visits(arm));
                    assert_eq!(a.times_available(arm), b.times_available(arm));
                    assert_eq!(a.mean_reward(arm).to_bits(), b.mean_reward(arm).to_bits());
                }
                let (mut pa, mut pb) = (Vec::new(), Vec::new());
                a.policy_into(&mut pa);
                b.policy_into(&mut pb);
                assert_eq!(
                    pa.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    pb.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
            let successors: Vec<_> = graph.joint_successors().collect();
            assert_eq!(tree.children().len(), successors.len(), "joint successors");
            for (i, (child, target)) in tree.children().iter().zip(successors).enumerate() {
                for player in players.iter() {
                    assert_eq!(tree.joint_arm(i, player), graph.joint_arm(i, player));
                }
                match target {
                    Some(target) => assert_same_shape(child, target),
                    None => assert_eq!(child.visits(), 0),
                }
            }
        }
    }
}

/// Search `game` as a tree and its wrapped twin as a graph, one seed each, and
/// require the two to agree to the bit.
fn assert_same_search<G, W>(game: &G, wrapped: &W, perspective: u8, cfg: &Config, seed: u64)
where
    G: Game<Context = ()>,
    G::Choice: Debug,
    W: Transposition<Choice = G::Choice, Context = ()>,
{
    let mut tree = Searcher::new(game);
    let a = tree.search(
        game,
        &(),
        perspective,
        cfg,
        None,
        &mut WyRand::seed_from_u64(seed),
    );
    let mut graph = GraphSearcher::new(wrapped);
    let b = graph.search(
        wrapped,
        &(),
        perspective,
        cfg,
        None,
        &mut WyRand::seed_from_u64(seed),
    );
    assert_same_result(&a, &b);
    if let (Some(tree), Some(graph)) = (tree.tree(), graph.graph()) {
        assert_same_shape(tree, graph.root());
    }
}

fn fingerprint_configs() -> Vec<Config> {
    let mut configs = Vec::new();
    for iterations in [50u32, 500, 5000] {
        for &(early_termination, progressive_bias_weight) in
            &[(false, 0.0), (true, 0.0), (false, 0.75)]
        {
            configs.push(Config {
                iterations,
                exploration_constant: 0.4,
                early_termination,
                progressive_bias_weight,
                ..Config::default()
            });
        }
    }
    configs
}

fn both_wrappers<G>(game: &G, perspective: u8, cfg: &Config, seed: u64)
where
    G: Game<Context = ()> + Clone,
    G::Choice: Debug,
{
    assert_same_search(game, &Untransposed(game.clone()), perspective, cfg, seed);
    assert_same_search(game, &PathKeyed::new(game.clone()), perspective, cfg, seed);
}

/// The whole sequential fingerprint matrix, through both wrappers.
#[test]
fn a_graph_that_never_merges_searches_exactly_as_the_tree() {
    for seed in [1u64, 7, 21] {
        for cfg in fingerprint_configs() {
            both_wrappers(&AlwaysWin { ply: 0 }, 1, &cfg, seed);
            both_wrappers(&CountToThree::new(), 0, &cfg, seed);
            both_wrappers(&GameTree::minimal_trap(), 0, &cfg, seed);
            both_wrappers(&GameTree::wide(200), 0, &cfg, seed);
            both_wrappers(&GameTree::wide_two_ply(40), 0, &cfg, seed);
        }
    }
}

/// Re-rooting, through a graph that never merges, lands on the tree's subtree.
#[test]
fn re_rooting_a_graph_that_never_merges_agrees_with_the_tree() {
    for seed in [1u64, 7, 21] {
        for cfg in fingerprint_configs() {
            let game = GameTree::wide_two_ply(40);
            let mut wrapped = PathKeyed::new(game.clone());
            let mut tree_game = game.clone();
            let mut tree = Searcher::new(&tree_game);
            let mut graph = GraphSearcher::new(&wrapped);
            let mut a_rng = WyRand::seed_from_u64(seed);
            let mut b_rng = WyRand::seed_from_u64(seed);

            let a = tree.search(&tree_game, &(), 0, &cfg, None, &mut a_rng);
            let b = graph.search(&wrapped, &(), 0, &cfg, None, &mut b_rng);
            assert_same_result(&a, &b);
            assert_eq!(
                tree.reuse_subtree(&a.choice),
                graph.reuse_subtree(&b.choice)
            );
            tree_game.apply_choice(&(), &a.choice, &mut a_rng);
            wrapped.apply_choice(&(), &b.choice, &mut b_rng);

            let a = tree.search(&tree_game, &(), 1, &cfg, None, &mut a_rng);
            let b = graph.search(&wrapped, &(), 1, &cfg, None, &mut b_rng);
            assert_same_result(&a, &b);
            assert_same_shape(
                tree.tree().expect("a tree"),
                graph.graph().expect("a graph").root(),
            );
        }
    }
}

fn simultaneous_configs() -> Vec<Config> {
    let mut configs = Vec::new();
    for iterations in [10u32, 300, 3000] {
        for early_termination in [false, true] {
            for root_policy in [RootPolicy::Sampled, RootPolicy::MostVisited] {
                configs.push(Config {
                    iterations,
                    early_termination,
                    simultaneous: SimultaneousConfig {
                        root_policy,
                        ..SimultaneousConfig::default()
                    },
                    ..Config::default()
                });
            }
        }
    }
    configs
}

fn perspective_of<G: Game<Context = ()>>(game: &G) -> u8 {
    match game.status(&()) {
        Status::Active { player } => player,
        Status::Simultaneous { players } => players.iter().next().expect("a participant"),
        Status::Terminal(_) => unreachable!("fixtures start in play"),
    }
}

fn both_wrappers_simultaneous<G>(game: &G)
where
    G: Game<Context = ()> + Clone,
    G::Choice: Debug,
{
    let perspective = perspective_of(game);
    for seed in [3u64, 11] {
        for cfg in simultaneous_configs() {
            both_wrappers(game, perspective, &cfg, seed);
        }
    }
}

/// Every simultaneous fixture, under both policies and both root policies.
#[test]
fn simultaneous_nodes_search_exactly_as_the_tree() {
    both_wrappers_simultaneous(&Rps::default());
    both_wrappers_simultaneous(&BiasedRps::default());
    both_wrappers_simultaneous(&Pennies::default());
    both_wrappers_simultaneous(&SequentialPennies::default());
    both_wrappers_simultaneous(&ThreeWayParity::default());
    both_wrappers_simultaneous(&SeqThenSimul::default());
    both_wrappers_simultaneous(&AsymmetricBid::default());
    both_wrappers_simultaneous(&VariableRps::default());
    both_wrappers_simultaneous(&ManyArm::default());
    both_wrappers_simultaneous(&AbandonedRivals::default());
    both_wrappers_simultaneous(&ForbiddenFavourite::default());
}

/// Every route into a position must land in one vertex, and every vertex must
/// have been entered exactly as often as its incoming links were taken.
fn assert_merged_and_conserved<C, K: Eq + Hash + Debug>(graph: &mcts::Graph<C, K>) {
    assert_merged(graph, true);
}

/// [`assert_merged_and_conserved`], where `exact` is false after a re-root: a
/// survivor keeps the visits it took through parents the re-root discarded, so
/// it has been entered *at least* as often as its surviving links were taken.
fn assert_merged<C, K: Eq + Hash + Debug>(graph: &mcts::Graph<C, K>, exact: bool) {
    let mut seen = HashSet::new();
    let mut incoming: HashMap<usize, u32> = HashMap::new();
    for id in 0..graph.node_count() {
        let node = graph
            .node(id)
            .expect("every id below the count names a node");
        if !node.is_root() {
            if let Some(key) = node.key() {
                assert!(
                    seen.insert((key, node.reward_player())),
                    "position {key:?} has two vertices"
                );
            }
        }
        for edge in node.edges() {
            if let Some(target) = edge.target() {
                *incoming.entry(target.id()).or_default() += edge.visits();
            }
        }
    }
    for id in 0..graph.node_count() {
        let node = graph.node(id).expect("a node");
        if node.is_root() {
            continue;
        }
        let taken = incoming.get(&id).copied().unwrap_or(0);
        if exact {
            assert_eq!(
                node.visits(),
                taken,
                "a vertex was entered a different number of times than its links were taken"
            );
        } else {
            assert!(
                node.visits() >= taken,
                "a vertex was entered fewer times than its links were taken"
            );
        }
    }
}

#[test]
fn every_route_into_a_position_reads_one_vertex() {
    let game = CountToThree::new();
    let cfg = Config {
        iterations: 2_000,
        early_termination: false,
        ..Config::default()
    };
    let mut graph = GraphSearcher::new(&game);
    let result = graph.search(&game, &(), 0, &cfg, None, &mut WyRand::seed_from_u64(5));
    assert_eq!(result.choice, 1, "only the all-ones line wins");

    let graph = graph.graph().expect("a graph");
    // Ten positions: (ply, points) for points <= ply <= 3.
    assert!(graph.node_count() <= 10, "{} vertices", graph.node_count());
    assert_merged_and_conserved(graph);
    assert_eq!(graph.root().visits(), 2_000);

    let mut tree = Searcher::new(&game);
    tree.search(&game, &(), 0, &cfg, None, &mut WyRand::seed_from_u64(5));
    assert!(tree.tree().expect("a tree").node_count() > graph.node_count());
}

#[test]
fn a_game_with_one_position_per_ply_is_a_chain() {
    let game = AlwaysWin { ply: 0 };
    let cfg = Config {
        iterations: 300,
        early_termination: false,
        ..Config::default()
    };
    let mut searcher = GraphSearcher::new(&game);
    searcher.search(&game, &(), 1, &cfg, None, &mut WyRand::seed_from_u64(9));
    let graph = searcher.graph().expect("a graph");
    assert_eq!(graph.node_count(), 4, "one position per ply, plies 0 to 3");
    assert_eq!(
        graph.edge_count(),
        9,
        "three links into each of plies 1 to 3"
    );
    assert_eq!(graph.max_depth(), 4);
    let targets: HashSet<usize> = graph
        .root()
        .edges()
        .map(|edge| edge.target().expect("every root link was taken").id())
        .collect();
    assert_eq!(
        targets.len(),
        1,
        "every root choice reaches the same position"
    );
    assert_merged_and_conserved(graph);
}

#[test]
fn a_set_is_searched_once_whatever_order_it_was_picked_in() {
    let game = Collect::new(8, 3);
    let cfg = Config {
        iterations: 5_000,
        early_termination: false,
        ..Config::default()
    };
    let mut graph = GraphSearcher::new(&game);
    let result = graph.search(&game, &(), 0, &cfg, None, &mut WyRand::seed_from_u64(13));
    assert!(
        result.choice >= 5,
        "picked item {}, which is not in the best set",
        result.choice
    );
    let positions = 1 + 8 + 28 + 56;
    let held = graph.graph().expect("a graph").node_count();
    assert!(
        held <= positions,
        "{held} vertices for {positions} positions"
    );
    assert_merged_and_conserved(graph.graph().expect("a graph"));

    let mut tree = Searcher::new(&game);
    tree.search(&game, &(), 0, &cfg, None, &mut WyRand::seed_from_u64(13));
    assert!(tree.tree().expect("a tree").node_count() > 2 * held);
}

#[test]
fn a_repetition_ends_the_descent_rather_than_looping() {
    let cfg = Config {
        iterations: 3_000,
        early_termination: false,
        ..Config::default()
    };
    for (square, winning_step) in [(3u8, 2u8), (4, 1)] {
        let game = Ring::new(5, square);
        let mut searcher = GraphSearcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut WyRand::seed_from_u64(17));
        assert_eq!(result.choice, winning_step, "from square {square}");
        let graph = searcher.graph().expect("a graph");
        // Five squares, two players to move, and the two winning landings.
        assert!(
            graph.node_count() <= 5 * 2 + 4,
            "{} vertices",
            graph.node_count()
        );
        assert!(graph.max_depth() <= graph.node_count());
    }
}

#[test]
#[cfg_attr(not(debug_assertions), ignore = "the key check is a debug assertion")]
#[should_panic(expected = "transposition_key changed")]
fn a_key_that_reads_hidden_state_is_refused() {
    let game = LeakyKey {
        hidden: 0,
        moves: 0,
    };
    let cfg = Config {
        iterations: 200,
        early_termination: false,
        ..Config::default()
    };
    let mut searcher = GraphSearcher::new(&game);
    searcher.search(&game, &(), 0, &cfg, None, &mut WyRand::seed_from_u64(1));
}

#[test]
fn re_rooting_keeps_only_what_the_new_root_reaches() {
    let mut game = CountToThree::new();
    let cfg = Config {
        iterations: 1_000,
        early_termination: false,
        ..Config::default()
    };
    let mut searcher = GraphSearcher::new(&game);
    let mut rng = WyRand::seed_from_u64(23);
    searcher.search(&game, &(), 0, &cfg, None, &mut rng);
    let before = searcher.graph().expect("a graph").node_count();

    assert!(searcher.reuse_subtree(&1));
    game.apply_choice(&(), &1, &mut rng);
    let graph = searcher.graph().expect("a re-rooted graph");
    assert!(graph.node_count() < before);
    assert_eq!(graph.root().key(), Some(&(1, 1)));
    // Every survivor is reachable from the new root.
    let mut reached = HashSet::from([graph.root().id()]);
    let mut frontier = vec![graph.root()];
    while let Some(node) = frontier.pop() {
        for edge in node.edges() {
            if let Some(target) = edge.target() {
                if reached.insert(target.id()) {
                    frontier.push(target);
                }
            }
        }
    }
    assert_eq!(reached.len(), graph.node_count());

    let inherited = graph.root().visits();
    let result = searcher.search(&game, &(), 0, &cfg, None, &mut rng);
    assert_eq!(result.reused_iterations, inherited);
    assert_eq!(result.choice, 1);
    assert_merged(searcher.graph().expect("a graph"), false);

    assert!(!searcher.reuse_subtree(&7), "a choice the root never took");
    assert!(searcher.graph().is_none());
}

#[test]
fn one_seed_gives_one_search() {
    let game = Collect::new(8, 3);
    let cfg = Config {
        iterations: 2_000,
        ..Config::default()
    };
    let run = || {
        let mut searcher = GraphSearcher::new(&game);
        let result = searcher.search(&game, &(), 0, &cfg, None, &mut WyRand::seed_from_u64(29));
        let graph = searcher.graph().expect("a graph");
        (
            result.choice,
            result.iterations_used,
            result.best_mean_reward.to_bits(),
            graph.node_count(),
            graph.edge_count(),
        )
    };
    assert_eq!(run(), run());
}

#[test]
fn a_forced_move_returns_without_searching() {
    let game = Collect::new(3, 3);
    let mut after_two = game.clone();
    let mut rng = WyRand::seed_from_u64(31);
    after_two.apply_choice(&(), &0, &mut rng);
    after_two.apply_choice(&(), &1, &mut rng);
    let mut searcher = GraphSearcher::new(&after_two);
    let result = searcher.search(&after_two, &(), 0, &Config::default(), None, &mut rng);
    assert_eq!(result.stop_reason, StopReason::SingleChoice);
    assert_eq!(result.choice, 2);
}

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    use mcts::{GraphRootParallel, RootParallel};

    fn assert_same_pool<G>(game: &G, threads: usize, cfg: &Config)
    where
        G: Game<Context = ()> + Clone + Send + Sync,
        G::Choice: Debug + Send + Sync,
        G::Side: Send,
    {
        let perspective = perspective_of(game);
        let wrapped = Untransposed(game.clone());
        let mut trees = RootParallel::new(threads, game, |i| WyRand::seed_from_u64(i as u64 + 1));
        let mut graphs =
            GraphRootParallel::new(threads, &wrapped, |i| WyRand::seed_from_u64(i as u64 + 1));
        let a = trees.search(game, &(), perspective, cfg, None);
        let b = graphs.search(&wrapped, &(), perspective, cfg, None);
        assert_same_result(&a, &b);
        for (tree, graph) in trees.trees().zip(graphs.graphs()) {
            assert_same_shape(tree, graph.root());
        }
    }

    #[test]
    fn a_pool_of_graphs_that_never_merge_answers_as_a_pool_of_trees() {
        for threads in [1, 3] {
            for cfg in simultaneous_configs() {
                assert_same_pool(&GameTree::wide_two_ply(12), threads, &cfg);
                assert_same_pool(&Rps::default(), threads, &cfg);
                assert_same_pool(&BiasedRps::default(), threads, &cfg);
                assert_same_pool(&VariableRps::default(), threads, &cfg);
            }
        }
    }

    #[test]
    fn a_pool_of_merging_graphs_finds_the_best_set() {
        let game = Collect::new(8, 3);
        let mut pool = GraphRootParallel::new(3, &game, |i| WyRand::seed_from_u64(i as u64 + 41));
        let cfg = Config {
            iterations: 3_000,
            ..Config::default()
        };
        let result = pool.search(&game, &(), 0, &cfg, None);
        assert!(result.choice >= 5);
        for graph in pool.graphs() {
            assert_merged_and_conserved(graph);
        }
        pool.reuse_subtree(&result.choice);
        assert!(pool.graphs().count() > 0);
    }
}
