# mcts

Monte Carlo Tree Search for board games, including imperfect-information games
via Information Set MCTS.

Built to be dropped into a real game engine without giving up the performance of
a hand-written search: no trait objects on the hot path, no allocation per
iteration beyond growing the tree, and no policy-inheritance layer to indirect
through.

## Using it

Implement `Game`, then drive a `Searcher`:

```rust
let mut searcher = Searcher::new(&state);
let result = searcher.search(&state, &ctx, player, &config, None, &mut rng);

state.apply(&result.choice);
searcher.reuse_subtree(&result.choice);   // next search inherits this subtree
```

`examples/tic_tac_toe.rs` is a complete implementation of a perfect-information
game; run it with `cargo run --release --example tic_tac_toe`.

## What it does

* **Information Set MCTS** — a fresh determinization per iteration, and an
  exploration term over how often a choice was *available* rather than how often
  its parent was visited, so a rarely-legal move is not mistaken for an
  under-explored one.
* **max^n backup** over per-player reward vectors. Any number of players, any
  reward scale; nothing assumes two players or zero sum.
* **Budgets** by iterations, wall clock, or an external `AtomicBool`, plus early
  termination once the remaining iterations provably cannot change the answer.
* **Tree reuse** across moves, and root parallelism behind the `parallel`
  feature.
* **Progressive bias** from a game-supplied prior, evaluated once per child.

## How it is put together

Everything a search varies on is either a field on `Config` or a hook on your own
game type. There is no policy trait to override, which is what keeps the hot path
monomorphic and inlinable across the crate boundary.

Two associated types carry what does not belong in the game state:

* `Context` — immutable for the whole search: evaluation tables, tuned
  parameters, move-filter flags. Keeping these out of the state means they are
  not copied on every determinization.
* `Side` — mutable, owned by the search, invisible to the tree. This is where a
  side model lives, such as a flat bandit over decisions you do not want to
  branch on.

Children are owned inline in a contiguous `Vec`, which is what selection scans.
A node builds a hash index only once it has more children than
`Game::CHILD_INDEX_THRESHOLD`; see `benchmarks/BASELINE.md` for the measured
crossover and why one strategy does not fit every branching factor.

## Features

| feature | default | effect |
|---|---|---|
| `time` | yes | wall-clock budgets. Turn it off for `wasm32-unknown-unknown`, where `Instant` panics. |
| `parallel` | no | `RootParallel`: independent trees per thread, merged at the root. |
| `serde` | no | `Config` is serialisable. |

## References

Most of the algorithm, and the vocabulary, comes from Browne et al.'s survey [1].
The Information Set treatment follows Cowling, Powley and Whitehouse [2].

[1] Browne, C. et al. (2012). *A Survey of Monte Carlo Tree Search Methods.*
IEEE Transactions on Computational Intelligence and AI in Games, 4(1), 1–43.

[2] Cowling, P., Powley, E. & Whitehouse, D. (2012). *Information Set Monte Carlo
Tree Search.* IEEE Transactions on Computational Intelligence and AI in Games,
4(2), 120–143.
