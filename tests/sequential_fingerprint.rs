//! A golden record of sequential search behaviour, captured before simultaneous-move
//! support existed.
//!
//! None of these games has a simultaneous node, so every number here must survive that
//! change byte for byte. If this test fails, the change reached the sequential path and
//! the change is wrong, not the golden file.
//!
//! Regenerate deliberately, never to make a red test green:
//! `MCTS_WRITE_GOLDEN=1 cargo test --test sequential_fingerprint`.

mod common;

use std::fmt::Write as _;

use common::{AlwaysWin, CountToThree, GameTree};
use mcts::rand_core::SeedableRng;
use mcts::{Config, Game, Searcher};
use wyrand::WyRand;

const GOLDEN: &str = include_str!("sequential_fingerprint.golden");
const PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/sequential_fingerprint.golden"
);

fn record<G: Game<Choice = usize, Context = ()>>(
    out: &mut String,
    name: &str,
    game: &G,
    perspective: u8,
    cfg: &Config,
    seed: u64,
) {
    let mut searcher = Searcher::new(game);
    let result = searcher.search(
        game,
        &(),
        perspective,
        cfg,
        None,
        &mut WyRand::seed_from_u64(seed),
    );
    let root = searcher.tree().expect("a search leaves a tree");

    writeln!(
        out,
        "{name} seed={seed} iters={} et={} pb={} => choice={} used={} reused={} stop={:?} \
         root_visits={} best_visits={} best_mean={:.12} nodes={} depth={}",
        cfg.iterations,
        cfg.early_termination,
        cfg.progressive_bias_weight,
        result.choice,
        result.iterations_used,
        result.reused_iterations,
        result.stop_reason,
        result.root_visits,
        result.best_visits,
        result.best_mean_reward,
        root.node_count(),
        root.max_depth(),
    )
    .unwrap();

    let mut children: Vec<(usize, u32, f64)> = root
        .children()
        .iter()
        .map(|c| {
            (
                *c.edge().choice().expect("a root child has a choice"),
                c.visits(),
                c.cumulative_reward(),
            )
        })
        .collect();
    children.sort_by_key(|&(choice, _, _)| choice);
    for (choice, visits, reward) in children {
        writeln!(out, "    {choice} {visits} {reward:.12}").unwrap();
    }
}

/// Re-rooting has its own arithmetic, so it gets its own record.
fn record_reuse(out: &mut String, cfg: &Config, seed: u64) {
    let mut game = GameTree::wide_two_ply(40);
    let mut searcher = Searcher::new(&game);
    let mut rng = WyRand::seed_from_u64(seed);

    let first = searcher.search(&game, &(), 0, cfg, None, &mut rng);
    let reused = searcher.reuse_subtree(&first.choice);
    game.apply(&first.choice);
    let second = searcher.search(&game, &(), 1, cfg, None, &mut rng);

    writeln!(
        out,
        "reuse seed={seed} iters={} => first={} reused={reused} second={} used={} inherited={} \
         root_visits={} best_mean={:.12}",
        cfg.iterations,
        first.choice,
        second.choice,
        second.iterations_used,
        second.reused_iterations,
        second.root_visits,
        second.best_mean_reward,
    )
    .unwrap();
}

fn fingerprint() -> String {
    let mut out = String::new();
    for seed in [1u64, 7, 21] {
        for iterations in [50u32, 500, 5000] {
            for &(early_termination, progressive_bias_weight) in
                &[(false, 0.0), (true, 0.0), (false, 0.75)]
            {
                let cfg = Config {
                    iterations,
                    exploration_constant: 0.4,
                    early_termination,
                    progressive_bias_weight,
                    ..Config::default()
                };
                record(&mut out, "always_win", &AlwaysWin { ply: 0 }, 1, &cfg, seed);
                record(
                    &mut out,
                    "count_to_three",
                    &CountToThree::new(),
                    0,
                    &cfg,
                    seed,
                );
                record(&mut out, "trap", &GameTree::minimal_trap(), 0, &cfg, seed);
                record(&mut out, "wide200", &GameTree::wide(200), 0, &cfg, seed);
                record(
                    &mut out,
                    "wide_two_ply40",
                    &GameTree::wide_two_ply(40),
                    0,
                    &cfg,
                    seed,
                );
                record_reuse(&mut out, &cfg, seed);
            }
        }
    }
    out
}

#[test]
fn sequential_behaviour_is_unchanged() {
    let actual = fingerprint();

    if std::env::var_os("MCTS_WRITE_GOLDEN").is_some() {
        std::fs::write(PATH, &actual).expect("write the golden file");
        return;
    }

    if actual == GOLDEN {
        return;
    }

    let mismatch = actual
        .lines()
        .zip(GOLDEN.lines())
        .enumerate()
        .find(|(_, (a, b))| a != b);
    match mismatch {
        Some((line, (a, b))) => panic!(
            "sequential behaviour changed at golden line {}:\n  expected: {b}\n  actual:   {a}",
            line + 1
        ),
        None => panic!(
            "sequential fingerprint differs in length: {} lines now, {} in the golden file",
            actual.lines().count(),
            GOLDEN.lines().count()
        ),
    }
}
