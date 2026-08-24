//! A golden record of sequential search behaviour, captured before simultaneous-move
//! support existed.
//!
//! None of these games has a simultaneous node, so every number here must survive that
//! change byte for byte. If this test fails, the change reached the sequential path and
//! the change is wrong, not the golden file.
//!
//! Regenerate deliberately, never to make a red test green:
//! `MCTS_WRITE_GOLDEN=1 cargo test --test sequential_fingerprint`.
//!
//! Regenerated once, for the early-termination soundness fix. Every `et=true`
//! row that spent its whole budget now reads `stop=Budget` rather than
//! `stop=Proven` — exhausting the budget was never a proof — and the six rows
//! that used to stop at exactly 2048 stop later, at the visit gap the crate can
//! actually prove, because the adversarial replay that produced 2048 was not an
//! upper bound. No `choice` moved, and no `et=false` row moved.
//!
//! Regenerated a second time, for the root extraction fix. A root now ranks its
//! children by **mean reward** among the children it selected at least
//! `rank::MIN_EVIDENCE` times, with the selection rate `visits / availability`
//! breaking ties, and a child under that bar never outranks one over it. The
//! early-termination proof is stated against the same ranking, and a mean is not
//! made of counts, so all that survives of it is the bar: a rival that cannot
//! reach `MIN_EVIDENCE` selections with the iterations left can never leave the
//! tier below the leader's.
//!
//! 106 of the 135 `record` rows are byte identical and 29 moved, of which only
//! **three** changed `choice`:
//!
//! - Twenty-six are `et=true` rows that prove later or not at all. A surviving
//!   proof needs every rival stuck under the bar, which needs fewer than 32
//!   iterations left, so no row here stops more than 30 short of its budget —
//!   where `count_to_three seed=1 iters=5000` used to stop at 2506. Three rows
//!   lose their proof entirely and report `Budget`; `choice` is unchanged in
//!   every one of the twenty-six, and `best_visits` and `best_mean` move only
//!   because the search ran longer.
//! - Three are the same `wide_two_ply40 seed=7 iters=500` search under all three
//!   configurations, where 40 children share 500 iterations and none of them
//!   reaches the bar. The answer moved from the child with 17 visits and a mean
//!   of 0.0588 to the one with 16 and a mean of 0.0625 — a root wider than its
//!   own budget answering with the best thing it measured, which is the whole
//!   point of the rule.
//!
//! All 27 `reuse` rows moved, in their `second` field and — for the one search
//! above — their `first`. The second search runs from the other player's side of
//! `wide_two_ply(40)`, where 39 of the 40 children pay the mover exactly 1.0, so
//! the answer is settled entirely by the rate tiebreak: a re-rooted node's
//! children carry the availabilities progressive expansion gave them at the node
//! that is now the root, and among equal means the child that needed fewer
//! opportunities wins. Every moved row still reports `best_mean=1.000000000000`,
//! so the answer changed identity, not quality.
//!
//! Regenerated a third time, to make the `pb=0.75` third of this file mean
//! something. No fixture in the repo overrode `Game::heuristic_bias`, so every
//! `pb=0.75` row was a byte-for-byte re-run of its `pb=0.0` twin and deleting
//! the progressive-bias term from `ucb_raw` left the whole suite green.
//! `GameTree` now returns a real prior — the share of a state's immediate
//! successors that the mover outright wins — so 14 of the 162 blocks moved,
//! every one of them a `pb=0.75` block, and none of the `pb=0.0` ones at any
//! seed or budget. Nine are `trap`, where the prior on a losing line delays the
//! revisit that UCB1's exploration term would otherwise schedule; the other
//! five are `wide_two_ply40 seed=21` and the `reuse` rows that share its search,
//! where 39 of 40 successors pay the mover 1.0 so the prior is near-uniform and
//! only breaks ties. One `choice` moved, `wide_two_ply40 seed=21 iters=50`, from
//! the child with 4 visits and a mean of 0.25 to the one with 3 and a mean of
//! 0.333 — 40 children sharing 50 iterations, decided in the tier below the
//! evidence bar.
//!
//! What moved is the prior's *presence*, not the fix that ships with it: this
//! file is insensitive to which player the prior is read for, because on these
//! fixtures the mover's prior always lands on the child that already leads.
//! `the_prior_is_read_in_the_movers_currency` in `tests/search.rs` is what pins
//! that.
//!
#[allow(dead_code)]
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
