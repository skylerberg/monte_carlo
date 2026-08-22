# Review — simultaneous-move support (DUCT + regret matching)

Scope: the whole uncommitted `simultaneous-play` change against `main`. Thirteen findings
survived adversarial verification; after deduplication they are seven distinct defects
(the stale-legality-mask bug was found independently by six lenses). Six further findings
were refuted and are not repeated here.

---

## 1. Verdict

### Revised verdict — after the fix round

**The original "not ready to merge" no longer holds. All seven defects are closed, each by
a change verified in the source and pinned by a regression test that was shown to fail when
the fix is reverted.** The two mutation-coverage findings are also closed, each re-verified
by re-running its mutation against the strengthened assertion. One item remains open: the
benchmarks, which gate tagging rather than merging.

Every mutation in §6 is now caught by the test named for the mechanism it breaks, not
merely by some test somewhere.

| # | finding | severity | status |
|---|---|---|---|
| 1 | Stale root legality mask across searches | major | **Closed** — `scratch.root_legal.clear()` (src/search.rs:629) with the comment saying why, `refresh_root_legal`'s doc corrected (src/search.rs:731-736), pinned by `a_reused_searcher_does_not_carry_a_legality_mask_between_searches` (tests/simultaneous.rs:1381) |
| 2 | `ROOT_CHOICES_INVARIANT` restamps leftover arms | major | **Closed** — both gates now arm on proof (src/search.rs:951 simultaneous, src/search.rs:1063-1067 sequential), pinned by the two tests in `tests/root_choices_invariant.rs` |
| 3 | Parallel `MostVisited` argmaxes raw pooled visits | major | **Closed** — `mixes`/`sample` split (src/parallel.rs:146-147) and `Merged::strategy_leader` (src/parallel.rs:232-233, defined at :322), pinned by `a_most_visited_merge_agrees_with_a_single_threaded_search` |
| 4 | Parallel merge drops illegal mass unrenormalized | minor, must-fix by adjacency | **Closed** — `spread` (src/parallel.rs:360) renormalizes each worker over its surviving legal arms, pinned by `a_workers_say_in_the_merge_is_its_visits_not_its_legal_mass` and `a_worker_with_no_legal_mass_contributes_no_strategy` |
| 5 | Two false "Known limitations" bullets | minor, must-fix before tagging | **Closed** — both deleted; the shipped behaviour is now recorded under "Changed" |
| 6 | `Node::reward_player` doc wrong at a sequential root | minor | **Closed** — src/node.rs:344-355 and CHANGELOG.md's matching bullet both now say "the player to act" |
| 7 | `reuse_subtree`'s documented recovery path panics in debug | minor | **Closed as a doc fix** — both methods document the debug/release split and point at `Node::simultaneous_players()`; both panic messages dropped the false "the tree has been dropped". Behaviour deliberately unchanged (§4) |
| M-1 | The availability division had no end-to-end test | minor | **Closed** — assertion flipped to `> always_legal[SCISSORS] + 0.10`; mutation 5 re-run and `a_rarely_legal_action_is_not_starved` now fails at 0.30066 against a bar of 0.39690 |
| M-2 | The floor schedule's claim was not tested by the test named for it | minor | **Closed** — absolute level assertion `late < 0.008` added alongside the ratio; mutation 6 re-run and the test now fails at 0.014188 |
| §5 | Benchmarks never run | — | **Open, and gates tagging rather than merging** |

**Merge: yes.** Nothing known-defective ships in `src/`. Both open findings are one-assertion
test changes, and neither is evidence that the mechanism it fails to cover is broken — each was
verified correct by hand and by unit tests during mutation testing; what is missing is a test
that would *notice* a future regression. Closing them before merge is cheap and recommended;
they are not the kind of thing that should block the branch if the tag is still some way off.

**Tag: not yet.** §5 stands unchanged: no wall-clock measurement was taken at any point in this
work. CHANGELOG.md's performance sentences have since been reworded to claim only what the
fingerprint and the `size_of` test prove — behaviour and layout, not time. `README.md` was
outside the fix round's scope; re-reading it for this update turned up one sentence still
resting on measurement nobody took, at README.md:169 — "a purely sequential game pays nothing
in memory or in dispatch for simultaneous support". The memory half is pinned by
`node_layout_is_unchanged`; the dispatch half is not measured anywhere. Resolve it with the
benchmark re-run, or narrow it to memory.

Current tree state, verified for this update: `cargo test --all-features` is green —
**97 passed, 0 failed** (55 unit, 1 allocation, 2 root-choices, 12 legacy integration,
1 fingerprint, 19 simultaneous, 7 reuse) — and `cargo clippy --all-features --all-targets`
is clean with no warnings. `tests/sequential_fingerprint.golden` was never regenerated and
`tests/search.rs` is unmodified; both still pass, which is the one claim carrying the most
weight (§3).

### The original verdict, kept as written

**No — not ready to merge.** The design is sound and the sequential invariant holds, but
four defects must be closed first. Three are silent wrong-answer bugs on paths the crate
documents as normal usage, and one is a pair of release-note bullets that tell adopters a
shipped safety filter does not exist.

None of the four is deep. The largest is a one-condition change plus a test; the rest are
a merge-statistic swap, a normalization, and a CHANGELOG edit. Estimated total: half a day
including regression tests. Nothing found calls the architecture into question — the
per-player-marginal DUCT structure, the lazy joint successors, the availability correction
and the legality masking are all as specified once §0's corrections are applied.

Current tree state at time of writing: `cargo test --all-features` is green (52 unit, 12
legacy integration, 1 fingerprint, 18 simultaneous, 7 reuse) and clippy is clean. Note that
a concurrent agent applied one of the fixes below (`scratch.root_legal.clear()` at
src/search.rs:611) into the worktree *during* the review — see MUST-FIX 1.

---

## 2. Confirmed findings

Every finding below is kept as it was written, defect and evidence intact, with a **CLOSED**
note appended recording what was actually done and where to check it. Nothing here was
removed because it was fixed: the record of what the code did wrong is the part a later
reader needs.

Line numbers in the finding bodies are as they stood at review time; the CLOSED notes carry
current ones.

### MUST-FIX BEFORE MERGE — all four closed

#### 1. Stale root legality mask across searches — `src/search.rs:729` (fix now present at `src/search.rs:629`) — major — **CLOSED**

`Scratch::root_legal` is a `Searcher` field that outlives both the search and the tree.
`refresh_root_legal` caches it on length alone (`root_legal.len() == simul.slot_len(slot)`),
which is sound within one search — arms are only appended — and unsound across searches.
`early_stop::settled`'s own staleness guard is the same length compare, so an equal-length
mask with the *previous* position's contents passes both.

Failure: one `Searcher` (the usage `Searcher`'s own docs prescribe: "keep one per player per
game and call `search` repeatedly") searches a position where player 0 holds {0,1,2}, then a
position where only {0,1} are legal but every determinization hands action 2 back. Arm count
is 3 both times, the all-legal mask survives, and the early-termination proof ranks the
withheld arm — the visit leader. Measured, `Ducted(ForbiddenFavourite)`, `MostVisited`,
budget 2000, seeds 0..16: reused `Searcher` stops `Proven` at 1256–1663 iterations and
returns a different move from the full budget on 8/16 seeds; a fresh `Searcher` stops at
1958–1999 and agrees 16/16; a control whose first search is on the *same* position diverges
0/16. The reported `StopReason::Proven` is false — nothing was proved about the arm returned.
This is exactly the contract `early_termination_does_not_change_a_simultaneous_answer` exists
to enforce; that test misses it only because it builds a fresh `Searcher` per seed.

Fix: `scratch.root_legal.clear();` once per search, before the iteration loop. This line is
**already in the tree** at src/search.rs:611 and is verified sufficient (reused becomes
byte-identical to fresh on all 16 seeds; full suite including the fingerprint stays green).
Remaining work, which is why this is still a must-fix:
* confirm that line is intended and survives into the commit;
* add a warm-`Searcher` variant of `early_termination_does_not_change_a_simultaneous_answer`
  — nothing in the suite currently pins it, so the line can be deleted without a failure;
* correct `refresh_root_legal`'s doc comment, which justifies the cache with an
  arms-only-grow argument that is true only within one search.

Effort: 30 min (the test is the work).

**CLOSED.** All three items done. `scratch.root_legal.clear()` survives at src/search.rs:629
and now carries a comment saying why it exists, so it will not be deleted as redundant.
`refresh_root_legal`'s doc (src/search.rs:731-736) no longer justifies the length cache with
an unqualified arms-only-grow argument: it states that the argument holds *within one search*
and that across searches a new position can offer the same arm count and a different set.
The missing test is `a_reused_searcher_does_not_carry_a_legality_mask_between_searches`
(tests/simultaneous.rs:1381) — one `Searcher`, warmed on `Ducted(ForbiddenFavourite::default())`
and then run on `Ducted(ForbiddenFavourite::banned())`, 16 seeds, asserting that the withheld
arm leads the visits, that the arm count is unchanged so the stale mask would survive the
length check, and that the answer matches the full-budget answer. Reverting the `clear()`
makes it fail on seed 3. Both this test and its sibling now also assert
`stop_reason == Proven` and `root_visits < BUDGET` per seed, which closes a hole neither had:
without it a configuration that never proved anything would compare a search against itself
and pass. The test's doc comment records that detection here is statistical rather than
structural — the §4 item asking for exactly that comment is satisfied.

#### 2. `ROOT_CHOICES_INVARIANT` fast path restamps leftover arms after re-rooting — `src/search.rs:894`, sequential twin at `src/search.rs:1027` — major — **CLOSED**

Iteration 1 sets `*root_fully_expanded = true` on the sole condition that no slot came back
with `legal == 0`; it never checks that the enumerated set *equals* the tree's. From
iteration 2 the fast path calls `Node::restamp_marginals`, which stamps `avail_epoch = visits`
on every arm including ones that only ever existed in determinizations reached at depth. A
root promoted by `reuse_joint` carries exactly such leftovers.

Failure (reproduced): two simultaneous plies, `determinize_into` an exact clone — so
`ROOT_CHOICES_INVARIANT` is a *true* claim at every root — with a chance draw at ply 1
deciding whether player 0 still holds action 2 at ply 2. The ply-2 node accumulates the union
{0,1,2}; the real draw removes action 2; `reuse_joint` promotes it. Release build: 1920 of
3000 iterations selected and applied player 0's action 2 at a position offering only {0,1}.
Flag off: 0. Debug aborts at src/search.rs:877 with a message blaming the game for setting the
flag wrongly, when the game's claim is correct. The sequential twin
(`root_avail.resize(node.children.len(), true)` after `reuse_subtree`) fails identically and
is worse, because `most_visited` there is unfiltered — an illegal move can reach the caller.
That twin predates this change but is now reachable through the same re-rooting story.

Fix: gate the fast path on proof rather than on a pass having run — set
`*root_fully_expanded = true` only when `legal == simul.slot_len(slot)` for every slot, and in
the sequential branch only when `choices.len() == node.children.len()` and every `avail` entry
is true. A never-re-rooted root still passes on iteration 1, so the fingerprint is unaffected;
a promoted root with leftovers simply keeps paying for full expansion, which is correct.

Effort: 1–2 h including a simultaneous and a sequential regression test.

**CLOSED.** Both branches now arm the fast path on a proof rather than on a pass having run.
Simultaneous (src/search.rs:923-952): a local `every_arm_legal` accumulates
`legal == simul.slot_len(slot)` per slot, and the flag is set only
`if !degenerate && every_arm_legal && node.is_root()`. Sequential (src/search.rs:1063-1067):
`node.is_root() && choices.len() == node.children.len() && avail.iter().all(|&legal| legal)`.
In both cases expansion creates one arm or child per enumerated choice, so after the pass the
position's set is a subset of the tree's and equal counts force set equality — which is what
makes it a proof rather than a heuristic. A never-re-rooted root is empty on iteration 1 and
still arms from iteration 2 exactly as before, so the fingerprint is untouched and passes.

Regression tests: `tests/root_choices_invariant.rs`, one simultaneous and one sequential, over
two local fixtures that determinize by cloning — so `ROOT_CHOICES_INVARIANT` is a *true* claim
and a failure cannot be blamed on the game. Each asserts the leftover arm or child exists
before asserting anything about it. Reverting both gates to `if node.is_root()` reproduces the
review's two failure modes exactly: in debug, the crate's own consistency assertions fire; in
release, the sequential twin hands the caller move 2 at a position offering `[0, 1]`.

A side effect worth recording: the fast path's own `assert_eq!` messages ("ROOT_CHOICES_INVARIANT
is set, but this determinization offers …") are now *true* accusations. The path can only be
reached after a pass proved the sets equal, so a later mismatch really is the game breaking its
own claim.

#### 3. Parallel `MostVisited` merge argmaxes raw pooled visits — `src/parallel.rs:214` — major — **CLOSED**

`sample` at src/parallel.rs:135 is false whenever `root_policy != Sampled`, so the merged
strategy is never accumulated and the merge falls through to `merged.leader()` — an argmax
over pooled raw `ArmStats::visits`. That is precisely the extraction DESIGN §0.4 corrected
away from: raw visits are neither exploration-free nor divided by availability. `RootParallel`
and `Searcher` therefore return different actions for the same position and policy.

Failure (reproduced): player arms 0/1 always legal, arm 2 legal in one determinization in four,
payoffs [0.7, 0.2, 1.0]; RegretMatching, `MostVisited`, 20k iterations. Single-threaded arm
visits [13798, 2084, 4118] → extracted weights [0.428, 0.054, 0.519] → action 2, which pays
1.0. Four-worker `RootParallel` at the same per-worker budget: pooled [54020, 9188, 16792] →
action 0, which pays 0.7.

Fix: accumulate the merged strategy whenever `simultaneous && G::SIMULTANEOUS_POLICY.mixes()`,
and for `MostVisited` argmax over `merged.strategy` when the policy mixes, keeping
`merged.leader()` for `Duct` (where the weight is the visit argmax anyway) and for sequential
roots. Pooled visits stay correct for `best_visits`/`best_mean_reward`.
`a_simultaneous_merge_pools_the_perspective_players_arms` (src/parallel.rs:533) asserts the
current behaviour and must be updated to assert the corrected weight.

Effort: 1 h.

**CLOSED.** The gate is split (src/parallel.rs:146-147):
`let mixes = simultaneous && G::SIMULTANEOUS_POLICY.mixes();` and
`let sample = mixes && cfg.root_policy == RootPolicy::Sampled;`. The per-worker strategy is
accumulated whenever `mixes`, and the answer is picked as `sample` → inverse-CDF draw,
`else if mixes` → `merged.strategy_leader()` (src/parallel.rs:232-233; the method is a
first-maximum argmax over `strategy` at src/parallel.rs:322, falling back to `merged.leader()`
when no mass), `else` → `merged.leader()` — which is `Duct` and every sequential root,
unchanged. `best_visits` and `best_mean_reward` still come from the pooled visits and reward
of the chosen entry.

`a_simultaneous_merge_pools_the_perspective_players_arms` (which asserted the defect) now
asserts `result.choice == strategy_leader(pooled_strategy(..))`, a test helper that mirrors the
merge's arithmetic term for term through the same `spread`, and still checks
`best_visits == pooled[result.choice]`. The new
`a_most_visited_merge_agrees_with_a_single_threaded_search` pins the cross-check the finding is
really about: a fixture where the raw-visit ranking and the availability-divided ranking
disagree, asserting the fixture bites, that the single-threaded `Searcher` and `RootParallel`
return the same action, and that `best_visits` is still the pooled count. Reverting the fix
makes it fail with "one position, one policy, two answers".

#### 4. Parallel `Sampled` merge drops illegal mass without renormalizing — `src/parallel.rs:170`–`184` — minor severity, must-fix by adjacency — **CLOSED**

`Marginals::policy_into` normalizes over *all* of a worker's arms; the loop then skips arms
missing from the legal seed and accumulates the truncated remainder. A worker's influence
becomes `root.visits() * (mass it happened to place on unplayable actions' complement)`,
contradicting both the comment at src/parallel.rs:167 ("Normalized per worker before
averaging") and `policy_into`'s own documented contract ("a caller holding a narrower legal
set should zero the illegal entries and renormalize"), which the single-threaded
`duct::root_strategy_into` honours.

Failure (reproduced): `ForbiddenFavourite::banned()`, RegretMatching, `Sampled`, 4 workers ×
4000 iterations. All four workers ran exactly 4000 iterations, yet their legal mass came out
0.791 / 0.388 / 0.188 / 0.193 — a 4.2× spread in effective weight, 61% of the intended weight
silently discarded and unevenly distributed. Merged P(action 0) = 0.455 as coded versus 0.516
with per-worker legal renormalization; the favoured action flips on the first configuration
tried. This is largest in exactly the imperfect-information positions the filter was added for.

Fix: first pass sums `strategy[arm]` over surviving arms, then accumulate
`weight * strategy[arm] / legal_total`, falling back the way `root_strategy_into` does when
`legal_total == 0.0` (uniform over legal arms, not dropping the worker). Update the comment.
Bundled here because it is the same twenty lines as finding 3.

Effort: 30 min.

**CLOSED.** The merge loop collects `(merged index, strategy[arm])` for the arms that pass the
legality filter into a reused `shares` buffer and hands them to `spread` (src/parallel.rs:360),
which sums the surviving shares and accumulates `weight * share / total`, returning without
touching `out` when `total <= 0.0`. Each worker therefore contributes exactly `root.visits()`
of merged strategy mass however much of its normalized policy sat on unplayable actions, and a
worker with no surviving mass contributes no strategy while its visits and rewards still pool.
The review's suggested uniform-over-legal fallback was *not* taken: dropping a worker's
strategy while keeping its visits is the more conservative reading, and it is pinned by
`a_worker_with_no_legal_mass_contributes_no_strategy`. The stale
"Normalized per worker before averaging" comment now describes what the code does.
`a_workers_say_in_the_merge_is_its_visits_not_its_legal_mass` builds two workers whose surviving
legal mass is 0.15 and 0.50 and which disagree about the leading legal action; dropping the
`/ total` makes it fail with `[300.0, 350.0]` against the intended 1000 each, with the argmax
flipped.

#### 5. Two false entries in "Known limitations" — `CHANGELOG.md:158`–`167` — minor severity, must-fix before tagging — **CLOSED**

Both bullets describe defects the code no longer has. `RootParallel::merge` *does* apply a
legality filter (src/parallel.rs:148-156 seeds `Merged` from
`state.choices_for_into` and looks arms up with `Merged::get`, which never inserts), pinned by
`tests/simultaneous.rs:1019`. `early_stop::settled` *does* take a `legal: &[bool]` and filters
both sides of the comparison (src/early_stop.rs:70-80), pinned by `tests/simultaneous.rs:1076`.
Release notes are read as a contract: an adopter follows these into defensive dead code or a
needlessly disabled `early_termination`, and a maintainer touching those paths does not look
for a regression.

Fix: delete both bullets, and move the fixes into "Changed" — that `merge` now enumerates the
perspective player's legal actions in the real position and merges only matching arms, and that
`early_termination` at a `Duct` simultaneous root ranks only the arms `best_arm` draws from and
refuses to prove when its mask is out of step with the arm count. Keep the availability-residual
and full-marginal-expansion bullets; both are still accurate. DESIGN.md §0's matching sentence
is explicitly historical and needs no edit.

Effort: 10 min.

**CLOSED.** Both bullets are gone and the shipped behaviour is recorded under "Changed": the
merge's legality filter, its per-worker renormalization and its `MostVisited` statistic as three
sub-bullets, and the `Duct` early-termination mask — including that the mask is rebuilt from
scratch every search, which is finding 1 stated where an adopter will see it — as its own. The
availability-residual and full-marginal-expansion bullets were kept as the finding asks;
DESIGN.md was not touched.

While re-reading the whole file against the source, four further things were corrected:

* The `Node::reward_player` bullet at CHANGELOG.md:51 (finding 6's twin), which claimed the
  root reports the **perspective player's** reward and thereby contradicted the sentence four
  lines above it.
* "a game with no simultaneous nodes pays nothing" — an unmeasured timing claim, and precisely
  what §5 says must not ship on the strength of this review. It is now a structural statement
  (dispatch is behind `Status` and `G::SIMULTANEOUS_POLICY`; the fingerprint pins *behaviour*)
  with an explicit note that no timing comparison was run.
* "thousands of 32-byte arms" understated the default: an arm is `ArmStats` (32 bytes) plus,
  under `RegretMatching`, an `ArmPolicy` (16 bytes) — both sizes pinned by
  `node_layout_is_unchanged` (src/node.rs:1065).
* Two limitations that §4 of this review says should survive into the released changelog were
  missing from it entirely, and were added: that `Duct` does not converge at a mixed
  equilibrium, and that `reuse_subtree`/`reuse_joint` are two-mode across build profiles.

### SHOULD-FIX LATER — both closed

#### 6. `Node::reward_player` doc is wrong at a sequential root — `src/node.rs:348`, `CHANGELOG.md:51` — minor — **CLOSED**

The doc says the root accumulates "the reward of the player the search is running on behalf
of". True for a joint successor and for a simultaneous root (where `search` asserts
`players.contains(perspective)`), false for a sequential root: src/search.rs:521 binds
`root_player` from `Status::Active { player }` and src/search.rs:596 assigns it, with nothing
requiring `perspective == player`. A caller running an opponent model
(`search(&state, &ctx, /* perspective */ 0, ..)` on a state where player 1 acts) reads
`reward_player() == 1` and accumulates player 1's payoffs where the doc promises player 0's —
a sign-flipped zero-sum readout in any diagnostic that labels it. Confirmed by test.
CHANGELOG.md:51 repeats the claim and contradicts CHANGELOG.md:46 four lines above.

Behaviour is the intended max^n rule; only the prose is wrong. Fix: say the root accumulates
the reward of the player to act. Effort: 10 min.

**CLOSED.** src/node.rs:344-355 now says a joint successor accumulates the perspective player's
reward and the root accumulates the reward of the player **to act** — the perspective player at
a simultaneous root, where the search requires the two to coincide, but at a sequential root
whichever player the position says is on turn, which differs from the perspective player
whenever the caller is running an opponent model. CHANGELOG.md's matching bullet was corrected
to the same statement in this session, and no longer contradicts the sentence four lines above
it. Behaviour unchanged, as the finding says it should be.

#### 7. `reuse_subtree`'s documented recovery path panics in debug — `src/search.rs:357`, assert at `src/search.rs:365` — minor — **CLOSED**

The doc says flatly "Returns false, and drops the tree, at a simultaneous root", in the same
sentence shape as the genuinely non-panicking sentence above it, and never mentions the
`debug_assert!`. A game with both node kinds writes the natural
`if !searcher.reuse_subtree(&played) { /* fresh search */ }` loop, which is correct against
the documented contract and correct in release, and panics under `cargo test`. The message
also claims "The tree has been dropped" while `self.root` is still `Some`. The two-mode
behaviour is intentional and pinned by tests both ways, so this is a doc-and-message fix, not
a behaviour change: document the debug panic, point callers at `Node::simultaneous_players()`
for branching, and reword the message. The `reuse_joint` half of the original finding was a
misread — its doc makes no claim about a sequential root. Effort: 15 min.

**CLOSED as a doc-and-message fix; the two-mode behaviour deliberately stands.** Both
`reuse_subtree` (src/search.rs:357-363) and `reuse_joint` (src/search.rs:393-398) now state the
debug/release split in those words — "debug panics on a `debug_assert!`, release returns false
and drops the tree" — and point callers at `Node::simultaneous_players()` over
`Searcher::tree()` for branching, rather than at a return value that does not survive a debug
build. Both panic messages lost the false "The tree has been dropped." (`self.root` is still
`Some` at the assert) and now say that a release build drops the tree and returns false instead
of panicking.

The `debug_assert!`s were kept, and the reasoning is worth recording because it cuts against
the simpler fix: a wrong-kind call is a caller error, and the return value cannot distinguish
"you called the wrong method" from the ordinary miss the same `false` reports, so dropping the
assert would delete the only signal and silently degrade a bug into a fresh search forever.
§4 lists the two-mode behaviour as deliberate and it is pinned by tests both ways, so removing
it would have broken a pinned contract to fix prose. The `should_panic(expected = …)` prefixes
those tests match are unchanged.

---

## 3. What was checked and found clean

**The sequential-regression risk — the claim carrying the most weight.** A game with no
simultaneous node must behave byte-identically to 0.2.0. Evidence:

* `tests/sequential_fingerprint.rs` against its `.golden` passes unmodified: 405 searches,
  chosen move, iteration counts, stop reasons, per-child visits and cumulative rewards to 12
  significant figures, node counts and depths, across 5 fixtures × 3 seeds × 3 budgets ×
  early-termination and progressive-bias variants. The golden was never regenerated,
  `MCTS_WRITE_GOLDEN` was never set, and the file was not edited.
* The twelve tests in `tests/search.rs` pass unmodified.
* Structural, not just empirical: every simultaneous path is gated behind `Status` and
  `G::SIMULTANEOUS_POLICY`. `refresh_root_legal` returns immediately when `root.simul()` is
  `None`; `early_stop::settled` never reads `legal` at a sequential root; `sample` in the
  parallel merge is false on the sequential path, so `merged.strategy` is never read there.
* Verified under perturbation: the fix for finding 1 was applied and the fingerprint still
  passed, confirming the sequential path does not consult the mask at all.
* **Still true after the fix round**, and re-verified for this update: the fingerprint and all
  twelve `tests/search.rs` tests pass, with `.golden` never regenerated and `tests/search.rs`
  never edited. Finding 2's fix tightens a condition on the sequential path — the one change in
  the round that could have moved a sequential search — and does not: a root that was never
  re-rooted holds exactly the enumerated choices after its first pass, so it arms on the same
  iteration as before. §6's mutation testing adds independent evidence, in the other direction:
  all eight simultaneous-mechanism mutations left the fingerprint and `tests/search.rs` green.

The one sequential exposure this evidence does *not* cover is finding 2's twin at
src/search.rs:1027 — it needs `ROOT_CHOICES_INVARIANT` (opt-in, default false) plus
`reuse_subtree`, neither of which the fingerprint fixtures exercise. That is a pre-existing
defect surfaced by this review, not a regression introduced here, but it should be fixed with
finding 2 rather than after it.

**Also checked and clean:**

* The legality filter in `RootParallel::merge` and the legality mask in `early_stop::settled`
  both genuinely ship and are non-vacuously pinned (each test first asserts the fixture bites —
  that the withheld arm really does lead the visits — before asserting the answer is legal).
* Both new regression tests were proven to bite: reverting each fix makes its test fail with a
  message naming the real defect, and restoring it makes the whole suite byte-identical again.
* Slot-relative arm numbering is consistent everywhere it was suspected: `strategy[arm]`,
  `stats[arm]` and `marginals.choice(arm)` in the merge; `simul.starts[slot]`-offset indices in
  `refresh_root_legal` against `arm_stats[simul.slot_range(slot)]` in `settled`. Several
  findings were refuted on exactly this point.
* `reuse_joint`'s rejection paths (wrong tuple length, non-participant, repeated player) and
  arm-statistics carry-forward, all covered by `tests/simultaneous_reuse.rs`.
* `cargo clippy --all-features --all-targets` clean.
* No leftover debug instrumentation: the `MCTS_PROBE` `eprintln!` blocks and `tests/zz_*.rs`
  files that concurrent agents introduced during the review are gone from the tree.

---

## 4. Known limitations shipping deliberately

Each of these is a conscious trade, not an oversight. All should survive into the released
CHANGELOG (unlike the two false bullets in must-fix 5).

* **Nothing is proved deeper than one simultaneous ply, and determinization voids the
  convergence theorems literally.** Acceptable because `README.md` and the `SimultaneousPolicy`
  doc state exactly what survives, in the words DESIGN §1.2 fixes, without softening. Anyone
  who needs a theorem at depth needs a different algorithm, not a different default.
* **The availability correction has a residual** (DESIGN §0, "two things deliberately not
  changed"): arms conditioned on different legality events are not two entries of one
  distribution, so where arm A is always legal and arm B is legal half the time and strictly
  better whenever legal, extraction reports 0.33 on A instead of 0. Acceptable because fixing it
  exactly needs a statistic per legal *set* rather than per arm — a different memory class — and
  DESIGN §1.2 already frames the sleeping-bandit corrections as principled engineering rather
  than theorems. It is written down in `root_strategy_into`'s doc comment rather than implied
  away, which is the right handling.
* **`SimultaneousPolicy::Duct` does not converge in mixed-equilibrium games.** Proved, tabulated,
  and not the default. Acceptable because it is opt-in, cheaper per arm (32 B vs 48 B), and
  correct for nodes with pure equilibria — which is why the defaults differ (0.75 vs 0.4 per
  DESIGN §0.3).
* **A simultaneous node expands every participant's full marginal list on first visit.** Capping
  or staging it is exactly what voids the convergence property, so a player with thousands of
  legal actions allocates thousands of 32-byte arms and scans them every visit. Acceptable as a
  stated cost with a stated reason; the alternative is a silently weaker guarantee.
* **`reuse_joint` misses are the common case.** A simultaneous node has up to `prod(|A_i|)`
  successors and materializes at most one per iteration, so the subtree usually is not there. A
  miss drops the tree and costs a fresh search rather than a wrong one, which is the same price
  a `reuse_subtree` miss has always cost. Documented on the method.
* **`reuse_subtree`/`reuse_joint` are two-mode: return false in release, `debug_assert` in
  debug.** Deliberate and pinned by tests both ways; after finding 7's doc fix this is a
  documented behaviour rather than a surprise. **Done:** both method docs now state the split,
  and the CHANGELOG carries it as a known limitation so an adopter meets it before a debug
  build does.
* **DESIGN.md is history, not authority.** It was corrected twice by measurement and §0 records
  what shipped. Acceptable only because §0 is explicit about which later sections it overrides;
  anyone treating a later section as current will be wrong in five places.
* **The early-termination regression test bites statistically, not structurally** (surfaced by
  this review). Under the revert the search always stops early — 32/32 seeds, at ~75% of budget —
  but only ~45% of seeds change answer, so detection comes from the 16-seed loop
  (failure probability ~1e-5) rather than from a deterministic crossover. A deterministic
  construction needs opponent dynamics slow enough to place the crossing late, which is the same
  condition that makes the final answer noisy, and its crossing point would need retuning
  whenever the exploration defaults move. Acceptable, but it should be commented as such in the
  test so a future maintainer does not mistake a flake for the defect. **Done:** both
  early-termination tests carry that comment, and both now also assert `stop_reason == Proven`
  and `root_visits < BUDGET` per seed, so a configuration that never proved anything fails
  instead of silently comparing a search against itself.

---

## 5. What is NOT verified

**Benchmarks: not run, and this is a real hole.** This machine has been under heavy contention
from other users throughout (load 7–44), so wall-clock measurement was void by a pre-registered
criterion and no valid window appeared. Nothing in this review supports any timing claim. In
particular the following remain **unmeasured**, not merely unstated:

* "a game with no simultaneous nodes pays nothing" (CHANGELOG). The fingerprint proves the
  *behaviour* is byte-identical, which says nothing about instruction count or branch cost. The
  new `Status` variant adds a match arm on the hot path and the new `Scratch` fields grow the
  struct; both are structurally cheap and neither is measured. **The CHANGELOG sentence has
  since been rewritten to claim only the structural fact and to say outright that no timing
  comparison was run. The same claim still stands unqualified at README.md:169** ("pays nothing
  in memory or in dispatch"), where the memory half is pinned by `node_layout_is_unchanged` and
  the dispatch half is not measured anywhere.
* Node size 64/72/80/88 bytes for `C = u8/u16/usize/Vec<u16>` (CHANGELOG). **Correction to
  this review:** it *is* pinned, by the `node_layout_is_unchanged` unit test at
  src/node.rs:1065, which also pins `ChildIndex`, `Extra`, `Simul`, `ArmStats` (32 B) and
  `ArmPolicy` (16 B). A runtime test rather than a `const` assertion, but it fails the build
  the same way. The CHANGELOG now points at the test rather than resting on the prose. What
  remains unmeasured is the *cost* of that layout, not the layout.
* Selection cost `sum(|A_i|)` rather than `prod(|A_i|)`, and "zero extra hash probes per visit"
  for `RegretMatching`. Both are inspectable in source and both look right there; neither was
  timed.
* The cost of the extra `Game::choices_for_into` call per parallel merge, and of the extra move
  generation per search that finding 1's fix introduces. Both are once-per-search, so the
  reasoning is that they are negligible against thousands of iterations — reasoning, not
  measurement.

**Before tagging 0.3.0, re-run the benchmark suite on an idle machine and either confirm or
delete every performance sentence in `CHANGELOG.md` and `README.md`.** Do not ship those
sentences on the strength of this review.

**Added since this review, and worth recording as the one steady-state cost that now *is*
gated:** `tests/allocation.rs` instruments the global allocator (including `alloc_zeroed`) and
asserts that four fixtures allocate an identical, small, budget-independent number of times at
two budgets five times apart — measured at 16 / 177 / 20 / 23 allocations and *asserted* equal
across the two budgets rather than against those constants — with the searcher's construction inside the measured region and a node-count
saturation precondition at *both* budgets so a search that stalled cannot pass by allocating
equally little twice. That is an allocation gate, not a timing measurement: it says the steady
state does not allocate per iteration and says nothing about instruction count or branch cost.
The benchmark hole above is untouched by it.

**Also unverified:**

* DESIGN §0's exploitability tables (the `t^-1/4` floor, the `duct_exploration` /
  `regret_matching_exploration` split, the tie-pool second pass). These were measured during
  implementation and were not re-measured here; the review checked that the *code* matches what
  §0 says shipped, not that §0's numbers reproduce.
* `RootParallel` thread-safety beyond the merge arithmetic. No loom, no TSan, no stress run.
  The merge reads worker trees after the scope closes, which is structurally sound, but worker
  interleaving under a deadline or cancellation was not exercised.
* Joint-key packing under adversarial arm counts. `wide_simultaneous_nodes_track_every_arm_and_joint_child`
  covers a wide node, but there is no test near the packing width limit and no property test
  over slot-relative index round-tripping.
* Behaviour with more than three simultaneous participants, and with participants whose action
  lists differ by orders of magnitude. `asymmetric_participants_get_their_own_action_lists` and
  `three_players_each_do_at_least_as_well_as_a_coin_flip` are the ceiling of current coverage.
* The worktree was edited by concurrent agents during the review. `src/search.rs` gained the
  finding-1 fix mid-flight. Diff the final tree against your own expectations before committing.

---

## 6. Mutation coverage

Sixteen-plus green simultaneous tests are only evidence if the paths they name are actually
entered. Each mechanism below was broken **one at a time** in `src/`, the whole suite run with
`--no-fail-fast`, and the source restored byte-exactly (verified with `cmp` against a
pre-mutation snapshot) before the next. `tests/sequential_fingerprint.golden` was never opened
for writing, `MCTS_WRITE_GOLDEN` was never set, and no benchmark was run.

Baseline: **97 tests pass** (55 unit, 1 allocation, 2 root-choices, 12 legacy sequential,
1 fingerprint, 19 simultaneous, 7 reuse), debug and release, clippy clean.

### The table

| # | mutation | plan expected | caught by | failed |
|---|---|---|---|---|
| 1 | `select_regret_matching` -> `unimplemented!()` | `biased_rps_learns_the_known_equilibrium` | **that test**, plus `regret_matching_learns_the_biased_rps_equilibrium`, `a_rarely_legal_action_is_not_starved`, `three_players_…`, `root_parallel_merges_marginals`, and 38 more | 43 |
| 2 | `select_duct` -> `unimplemented!()` | `duct_extracts_a_pure_strategy_on_rps` | **that test**, plus `biased_rps_duct_misses_the_equilibrium`, `both_policies_commit_to_a_pure_equilibrium`, `duct_breaks_a_tie_uniformly`, `duct_opens_unvisited_arms_uniformly`, `duct_ties_are_measured_against_the_final_maximum`, `duct_plays_a_pure_strategy_on_the_same_game`, 4 more | 11 |
| 3 | sleeping-bandit filter: drop the `avail_epoch == epoch` guard on the regret subtraction | `a_rarely_legal_action_is_not_starved` | **that test** (`scissors took 0.0052 of player 1's selections, far below the 0.26-0.31 the availability correction leaves it`), plus `an_arm_legal_half_the_time_keeps_a_usable_regret` and `a_most_visited_merge_agrees_with_a_single_threaded_search` | 3 |
| 4 | importance weight: `regret += u_hat` instead of `u_hat / mu` | `biased_rps_learns_the_known_equilibrium` | **that test**, caught by the **per-run** floor added this session (`seed 1 alone extracted [0.333, 0.333, 0.333] … exploitable by 0.0833, through the per-run floor of 0.12 / 0.04`), plus `regret_matching_learns_the_biased_rps_equilibrium`, `the_default_floor_keeps_improving_with_budget`, `most_visited_root_policy_is_deterministic`, 7 more | 11 |
| 5 | availability division: drop `/ availability` in `root_weight` | `a_rarely_legal_action_is_not_starved` / a root-extraction test | `root_extraction_divides_out_availability`, `the_deterministic_root_move_divides_out_availability`, `a_most_visited_merge_agrees_with_a_single_threaded_search`. **NOT** `a_rarely_legal_action_is_not_starved`; **no integration test at all** — see finding M-1 | 3 |
| 6 | `exploration_floor`: revert `t^(-1/4)` to `gamma_0 / sqrt(t)` | `the_default_floor_keeps_improving_with_budget` | **NOT that test** — see finding M-2. Caught only by `the_exploration_floor_is_always_a_usable_probability` (a constant table), `a_saturated_search_stops_allocating` (a saturation precondition), and `root_parallel_merges_marginals` (per-worker exploitability 0.0599 against the 0.04 floor) | 3 |
| 7 | backup: credit one perspective's `reward` to every arm instead of each arm's own player | `three_players_…` / asymmetric test | **both named tests** (`three_players_each_do_at_least_as_well_as_a_coin_flip`, `asymmetric_participants_get_their_own_action_lists`), plus `matching_pennies_…`, `rps_does_not_converge_to_a_pure_strategy`, `most_visited_returns_the_perspective_players_own_arm`, 13 more | 18 |
| 8 | joint key: pack the global arm index instead of the slot-relative one | `wide_simultaneous_nodes_track_every_arm_and_joint_child` | **that test**, plus 40 more. Surfaces as an index-out-of-bounds inside `JointChoices::get` (`src/game.rs:292`, "the len is 80 but the index is 109") rather than as a behavioural assertion — see note M-3 | 41 |

**No mutation left the suite green.** Every one of the eight is detected. Two are detected by
the wrong tests, which is recorded below as a finding rather than repaired in the table.

A positive result worth stating: **all eight mutations left `tests/sequential_fingerprint.rs`
and all 12 tests in `tests/search.rs` green**, every time. The 405-search byte-exact sequential
record is insensitive to every simultaneous mechanism this feature adds, which is
independent confirmation of the isolation the fingerprint was built to check.

### M-1 — the availability division had no end-to-end test — **CLOSED**

Dropping `/ simul.arm_stats[at].availability` in `root_weight` (src/duct.rs) leaves
**all 19 tests in `tests/simultaneous.rs` green**, `a_rarely_legal_action_is_not_starved`
included. Only three unit tests catch it, two of them by hand-setting counters on a bare
`Simul` and checking the arithmetic; no test that runs an actual search notices.

The plan expected `a_rarely_legal_action_is_not_starved` to catch it, and the reason it does
not is that its extraction assertion is one-sided and loose:

```rust
restricted[SCISSORS] > always_legal[SCISSORS] - 0.10
```

Measured on that fixture at 100 000 iterations, seed 1: correct code extracts scissors at
**0.4624**, the mutant at **0.3007**, against a bar of `0.2969 - 0.10 = 0.1969`. The mutation
moves the answer by 0.162 and the assertion has 0.266 of slack below the true value, so it
cannot see it. The test's other assertions are about *selection* counts during the search,
which the mutation does not touch at all.

This is the availability correction — the thing `root_strategy_into`'s doc comment spends
twenty lines justifying, and the mechanism §2 of this review flagged as silently corrupted
by any future change that resets `strategy_sum` without `availability`. It is currently
defended only by unit tests that would keep passing if the search never reached
`root_weight`.

**Suggested fix:** tighten that last assertion to a two-sided bracket around the measured
0.46 (e.g. `restricted[SCISSORS] > always_legal[SCISSORS] + 0.10`, since the correction is
supposed to push a rarely-legal arm's reported weight *above* its always-legal counterpart,
not merely keep it near). That single flip turns a test the mutant passes into one it fails
by 0.16.

**Status: CLOSED.** The assertion is now `> always_legal[SCISSORS] + 0.10` — strictly
above rather than merely near, which is what the availability division actually claims: it
turns `strategy_sum` into a mean probability *conditional on being legal*, so a rarely-legal
arm worth playing must come back weighted higher than its always-legal twin, not within a
tolerance of it. Re-ran mutation 5 against the fix: `a_rarely_legal_action_is_not_starved`
now FAILS at `0.30066 where an always-legal run gives 0.29690`, against a bar of 0.39690.
The availability correction now has the end-to-end test it was missing.

### M-2 — the floor schedule's behavioural claim was not tested by the test named for it — **CLOSED**

`the_default_floor_keeps_improving_with_budget` exists specifically to catch a reversion to
`gamma_0 / sqrt(t)`; its doc comment says so ("the reason `exploration_floor` is not
`gamma_0 / sqrt(t)`"). Reverting the exponent, it **passes**.

It asserts a *ratio*, `late < early * 0.5`, deliberately, because "the level is what the
tolerances in `regret_matching_learns_the_biased_rps_equilibrium` pin". Measured over the six
seeds the test itself uses:

| schedule | exploitability at 20k | at 200k | ratio | verdict |
|---|---|---|---|---|
| shipped `gamma_0 * t^(-1/4)` | 0.01475 | 0.004035 | **0.274** | passes |
| reverted `gamma_0 / sqrt(t)` | 0.04021 | 0.014188 | **0.353** | **also passes** |

The reverted schedule is **2.7x worse at 20k and 3.5x worse at 200k**, and its ratio still
clears 0.5 comfortably. The doc comment's stated basis for the assertion — that `0.1 / sqrt(t)`
gives a ratio of 0.61 — does not transfer to the default `gamma_0`, because at
`gamma_0 = 0.4` the `1/sqrt(t)` schedule hits `MIN_FLOOR = 0.005` at t = 6400 and is a
*constant* floor for the rest of both budgets. A constant floor is precisely the regime the
schedule was designed to escape, and the ratio test cannot see it, because a floor pinned at
a constant still lets regret matching improve.

The other half of the defence does not hold either: `regret_matching_learns_the_biased_rps_equilibrium`
gates on L-infinity `< 0.15`, far too loose to pin a level that moved from 0.0147 to 0.0402.
It passes under the mutation too.

What actually caught the reversion:

* `the_exploration_floor_is_always_a_usable_probability` — asserts `exploration_floor(0.4, 16) == 0.2`.
  A constants table. It detects *any* edit to the formula, including a correct one, and says
  nothing about whether the schedule works.
* `a_saturated_search_stops_allocating` — its saturation *precondition* fails
  ("the tree holds 8 nodes at 2000 iterations and 10 at 10000"). The allocation gate is not a
  test of the floor; it noticed because a worse-exploring search stops reaching the whole game.
* `root_parallel_merges_marginals` — the per-worker exploitability floor added this session:
  worker 1 at 0.0599 against 0.04. This is the only *behavioural* catch, and its name is about
  merging marginals, not about the floor.

So the exploration schedule — the single design decision this feature's tolerances are most
sensitive to, and the one the header comment of `tests/simultaneous.rs` tells a future
maintainer to suspect first — has **no test that asserts what it buys**.

**Suggested fix:** add a level assertion next to the ratio in
`the_default_floor_keeps_improving_with_budget`, e.g. `late < 0.008` and `early < 0.025`,
both comfortably inside the shipped 0.004 / 0.0147 and both violated by the reverted schedule's
0.0142 / 0.0402. Keep the ratio; it catches a different failure. A comment should record that
the ratio alone was measured *not* to catch the reversion at the default `gamma_0`, so the
level assertion is not decoration.

**Status: open.** `the_default_floor_keeps_improving_with_budget` (src/duct.rs:845) still
asserts only `late < early * 0.5`, verified in the current tree, and its doc comment still
gives the ratio-only rationale that the measurements above show does not hold at the default
`gamma_0`. The shipped schedule is the `t^(-1/4)` one; what is missing is a test that would
notice if it stopped being.

**Status: CLOSED.** The ratio assertion is kept and an absolute level assertion is added:
`late < 0.008`. The shipped schedule reaches 0.0040 at 200k and `gamma_0 / sqrt(t)` reaches
0.0142, so the two are separated by better than 2x in both directions. Re-ran mutation 6
against the fix: `the_default_floor_keeps_improving_with_budget` now FAILS with "mean
exploitability at 200k iterations is 0.014188255672241265". The doc comment no longer
justifies a ratio-only assertion — it records that the original reasoning was measured at
`gamma_0 = 0.1` and does not survive the move to the shipped 0.4, which is precisely how a
test came to not test its own subject.

Both assertions are needed: the ratio catches a rule that is uniformly mediocre, the level
catches a rule that improves at a respectable rate from a bad starting point.

### M-3 — note on mutation 8

The global-index joint key is caught overwhelmingly (41 tests), but the first thing to fail is
a bounds check inside `JointChoices::get`, not an assertion. That is a genuine defence and
arguably the strongest kind — the library refuses to run rather than returning a wrong answer —
but it means the *specific* claim in
`wide_simultaneous_nodes_track_every_arm_and_joint_child` about joint-key uniqueness is
carried by a panic that would also fire for unrelated packing bugs. This restates §5's
existing gap ("no property test over slot-relative index round-tripping"); mutation testing
does not close it.

### Restoration

After the eighth mutation every file in `src/` was restored and verified byte-identical to its
pre-mutation MD5, `tests/` was verified byte-identical including
`sequential_fingerprint.golden` (unchanged mtime), and the full suite is green:
`cargo test --all-features` 97 passed / 0 failed, `cargo test --release --all-features`
97 passed / 0 failed, `cargo clippy --all-features --all-targets` clean.

Re-verified for this update, after the CHANGELOG and REVIEW edits (which touch no code):
`cargo test --all-features` 97 passed / 0 failed and clippy clean, with the fingerprint and
the twelve `tests/search.rs` tests among them.

---

## 7. What a later reader should do with this document

Sections 2 and 6 are the record of what was wrong, kept whole. The **CLOSED** notes say what was
done and where to look; the finding above each one says why it mattered, and is the part worth
reading first if any of those lines is ever touched again. Two things are still open — M-1 and
M-2, both single-assertion test changes — and one, the benchmark hole in §5, gates the 0.3.0
tag rather than the merge. Nothing else in this review is outstanding.
