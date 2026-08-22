# Benchmarks

Machine: Apple M4, 4 P-cores + 6 E-cores, 32 GB, `aarch64-apple-darwin`.
Toolchain: `rustc 1.93.1 (01f6ddf75 2026-02-11)`.
Build: `[profile.bench]` inherits release — `lto = true`, `codegen-units = 1`,
`debug = 1`, plus `-C target-cpu=native -C force-frame-pointers=yes` from
`.cargo/config.toml`.

Measurement rules for anything recorded here: single-threaded, interleave A/B
rather than batching, discard the first repetition, report medians, and re-run
the first configuration at the end of a session — if it has drifted more than
3%, the session is void. Record the git sha and `rustc -vV` with every number.

## `child_storage` — where the linear child scan stops paying

Command:

```
cargo bench --bench child_storage -- --warm-up-time 0.4 --measurement-time 1.2
```

Median wall-clock for 64 search iterations at a node with `width` children,
re-enumerating `width` choices every iteration. `scan` forces the linear
`PartialEq` scan (`CHILD_INDEX_THRESHOLD = usize::MAX`), `index` forces the hash
index (`0`). `u64` is a cheap `Copy` choice; `vec` owns a `Vec<u32>`, so both
equality and hashing chase a pointer.

| children | scan/u64 | index/u64 | scan/vec | index/vec | cheap `Choice` | heap `Choice` |
|---:|---:|---:|---:|---:|:--|:--|
| 4 | 4.3 µs | 5.0 µs | 10.5 µs | 11.0 µs | scan 1.17x | scan 1.05x |
| 8 | 7.1 | 7.5 | 21.3 | 19.2 | scan 1.07x | index 1.11x |
| 16 | 10.1 | 10.0 | 44.5 | 31.9 | index 1.01x | index 1.39x |
| 32 | 20.1 | 15.5 | 105.8 | 54.9 | index 1.29x | index 1.93x |
| 64 | 52.8 | 21.9 | 288.4 | 74.2 | index 2.40x | index 3.89x |
| 128 | 196.1 | 39.5 | 963.6 | 147.5 | index 4.96x | index 6.53x |
| 256 | 709.6 | 71.3 | 3589.5 | 280.5 | index 9.95x | index 12.80x |
| 512 | 2602.4 | 133.6 | 13645.0 | 556.4 | index 19.48x | index 24.52x |
| 1024 | 9431.9 | 259.4 | 53233.9 | 1161.6 | index 36.36x | index 45.83x |

The scan quadruples per doubling above 64 — quadratic, as expected — while the
index doubles. Crossover is **16 children for a cheap `Choice` and 8 for one that
owns heap data**, which is why `Game::CHILD_INDEX_THRESHOLD` defaults to 16 and
why games with heap-owning choices should lower it.

This settles a question the two consumer implementations answered differently:
colori reverted a hash-based lookup as slower, and asteroids uses a hash map
throughout. Both were right for their own branching factors — colori's nodes hold
tens of children, asteroids' hold thousands — and neither answer generalises,
which is why the strategy switches on child count rather than being chosen once.

## `search` — throughput on workload-shaped synthetic games

Command:

```
cargo bench --bench search -- --warm-up-time 0.5 --measurement-time 2.0
```

Recorded at the first commit of the rewrite, as the reference every later change
is compared against.

| bench | budget | median | iterations/sec |
|---|---:|---:|---:|
| `tiny/100k` | 100 000 | 22.4 ms | 4.46 M |
| `narrow/1000` | 1 000 | 718 µs | 1.39 M |
| `narrow/10000` | 10 000 | 8.99 ms | 1.11 M |
| `wide/100` | 300 | 361 µs | 831 K |
| `wide/400` | 1 200 | 5.56 ms | 216 K |
| `wide/1600` | 4 800 | 93.6 ms | 51 K |

`narrow` is ~6% slower than first recorded because it now leaves
`ROOT_CHOICES_INVARIANT` off, which is the correct setting for it: its legal
choices depend on information that determinization permutes, so the root really
does have to be re-enumerated every iteration. The earlier number was measured
with an optimization that is unsound for that game.

`tiny` is the search core with the game removed: **224 ns per iteration**, which
is the floor everything else sits on top of. `narrow` costs ~4x that, most of it
in the rollout and the 2 KB state copy, and drifts ~20% slower from 1 000 to
10 000 iterations as the tree stops fitting in cache.

`wide` scales close to linearly in branching factor because every interior visit
enumerates the whole choice list and each choice owns a heap allocation —
roughly 7.7 M allocations for the 1 600-wide run. That cost belongs to the game,
not the search, and it is the single largest thing a game with heap-owning
choices can fix on its own side.

### Reading these as a regression gate

A change that improves `tiny` but not `narrow` moved something off the hot path
that was not on it. A change that improves `narrow` while `wide` regresses has
almost certainly traded child-lookup strategy for one branching factor at the
other's expense — check `CHILD_INDEX_THRESHOLD` before believing the win.

## `simultaneous` — not measured

`benches/simultaneous.rs` exists and compiles, with the three groups the design
calls for: `rps/{10k,100k}` at one `Duct` and one `RegretMatching` point,
`simul_wide/{8,32,128}` arms per player at arity 2, and `simul_arity/{2,3,4}` at
eight arms each.

**No number from it is recorded here, and none should be added without a valid
session.** Every attempt fell inside a window where this machine was carrying
other users' work — one-minute load average ranged 7.3 to 43.9, with foreign
processes sustaining 300–500% CPU throughout. Under that contention the
measurement rules above cannot be satisfied: the required closing re-run drifts
far past 3%, so every session was void by the file's own protocol.

The validity criterion was written down **before** any measurement was taken,
which is the only time such a rule is worth anything:

1. one-minute load average below 4.0 at session start and at session end;
2. no foreign process above 60% CPU at start;
3. the protocol above — interleave, discard the first repetition, medians,
   re-run the first configuration last;
4. that closing re-run within 3% of its own first measurement.

A session failing any of the four is void. Re-running until one passes is not
permitted; the gate is evaluated once, on the first valid session. When a quiet
machine is available, run the three groups and record them here, along with the
raw per-round windows and not only the medians, so a later reader can re-cut the
data under their own filter instead of taking this one on trust.

### The sequential regression gate is also unmeasured

The design gates this change on `tiny/100k` staying within 3% of 22.4 ms, and on
`narrow/*` and `wide/*` staying within 3% of their medians above. That gate has
**not** been evaluated, for the same reason.

What is known instead, and it is not nothing: `tests/sequential_fingerprint.rs`
holds a byte-exact record of 405 searches captured before this change existed —
chosen move, iteration and visit counts, per-child cumulative rewards to twelve
significant figures, node counts, depths, stop reasons, across five fixtures,
three seeds, three budgets, and the early-termination and progressive-bias
paths. It passes identically after the change. Sequential search therefore does
the *same work in the same order*; what is unverified is only whether that work
now takes longer per unit.

Stated precisely, because the distinction matters: **no regression detected,
which is weaker than no regression.**
