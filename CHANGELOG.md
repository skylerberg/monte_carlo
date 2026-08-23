# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

* **`StopReason::Proven` now means what it says.** The early-termination proof
  is the conserved-visit bound and nothing else: one iteration adds one visit to
  one candidate, so a candidate behind by more visits than there are iterations
  left cannot draw level. The adversarial UCB1 replay that used to run at a
  sequential root every 1024 visits is gone. It was not an upper bound — it paid
  every challenger the maximum reward in a single replay, so challengers
  competed with each other for a budget the question asks about one at a time —
  and it modelled only the children that existed when it ran, so a choice a
  later determinization first offered could be proven away before it appeared.
  Both returned a move the same search with `early_termination: false` does not.
  Deleting it also removes the replay's cost, which measured tens to hundreds of
  times the search's own selection work.
* **A search that spends its budget reports `StopReason::Budget`.** `settled`
  answered "nothing can change" whenever the root had reached `target`, and it
  is asked at the end of the iteration that gets there, so the loop broke
  `Proven` before its own budget test could run. Under the default config every
  full-budget sequential search reported `Proven`, and `Budget` was unreachable.
* **A `RootParallel` merge over more than one worker no longer reports
  `StopReason::Proven`.** A worker proves that the argmax of its own tree cannot
  be overtaken; the merged answer is an argmax over pooled statistics that proof
  never saw, and a worker that stopped early contributed fewer visits to the
  pool. `early_termination` still stops individual workers — at a pooled root it
  trades merged-answer stability for wall clock.

* **A forced move no longer leaves last turn's tree armed.** `Searcher::search`
  now consumes `tree_is_current` with `mem::take` as its first statement,
  instead of clearing it on its last line. A `StopReason::SingleChoice` return,
  a panic out of the preamble, or a panic thrown by the game can no longer leave
  a retained tree armed for a search that never asked to reuse one — which could
  return a move that is illegal in the current position, sometimes stamped
  `StopReason::Proven`. The same change closes two related variants: a
  non-reused search always rebuilds the root, resetting
  `scratch.root_fully_expanded`, and each `RootParallel` worker disarms itself,
  so an aborted worker cannot pool a stale position into the merge.
* **A zero-iteration or forced-move search at a simultaneous root now leaves a
  simultaneous root.** The marginal block is installed in the search preamble,
  so `Node::simultaneous_players()` is truthful even when no iteration ran, and
  `reuse_joint` takes its documented miss path instead of aborting the process
  in debug and blaming the caller. `Searcher::root_policy_into` returns false
  for such a root rather than reporting an empty, non-normalized policy.

### Added

* `Game::advance` and `Game::determinize_into` document the root contract: an
  `advance` at the root may resolve decisions the tree does not model, but not
  the one being searched, and a determinization may change what is legal but not
  who acts. Debug builds enforce both on every determinization. Release
  behaviour is unchanged.

## [0.3.0]

Simultaneous moves: a state can now report `Status::Simultaneous { players }`,
where every listed player chooses independently and concurrently and the tuple
of their actions determines the successor. See the "Simultaneous moves" section
of `README.md` for what converges to what, and for what determinization does to
that claim.

**Read "Silently changed behaviour" below even if the crate still compiles.**
One change in this release alters what a public accessor reports, in a purely
sequential game, with no compile error to warn you.

### Breaking

* **`Status` gained a variant.** `Status::Simultaneous { players: PlayerSet }`
  joins `Active` and `Terminal`. Any exhaustive `match` on `Status` — typically
  in a rollout policy or an evaluation function — now fails to compile. Games
  with no simultaneous nodes should add an `unreachable!()` arm; the search never
  produces the variant unless `Game::status` does.
* **`Config` gained fields.** `duct_exploration`, `regret_matching_exploration`
  and `root_policy` are new, so a struct literal that names every field no longer
  compiles. Add `..Default::default()`, or name the three. Every existing field
  keeps its meaning and its default, and none of the new ones is read at a
  sequential node.
* **`Config`'s `serde` representation gained those same three fields, with no
  `#[serde(default)]`.** Under the `serde` feature, a `Config` serialized by
  0.2.0 fails to deserialize into 0.3.0. Re-serialize stored configs, or add the
  three keys.

### Silently changed behaviour

* **`Node`'s reward player is redefined, and a *re-rooted* root's accumulated
  reward is reset.** The field that used to mean "the player who moved here" now
  means "the player whose reward this node accumulates", exposed as the new
  `Node::reward_player()`. For a node reached by one player's choice those are
  the same player, so nothing changes there. They differ at the **root**: a root
  produced by `Searcher::reuse_subtree` carried the *previous* mover's index, so
  what its cumulative reward meant depended on whether the tree had been reused.
  `Searcher::search` now stamps the root with the player to act and zeroes its
  `cumulative_reward` at the start of every search, reused tree or not.

  The visible consequences, for a caller that inspects `Searcher::tree()`:

  * `root.mean_reward()` and `root.cumulative_reward()` now report the reward
    of the **player to act**, where after a `reuse_subtree` they used to report
    the previous mover's. At a simultaneous root that is the perspective player,
    whose membership `search` requires; at a sequential root it is whichever
    player the position says is on turn, which is a *different* player from the
    perspective one whenever you are running an opponent model.
    `Node::reward_player()` reports it either way.
  * `root.cumulative_reward()` no longer carries reward inherited from the
    retained subtree; it counts only this search's iterations. `root.visits()`
    is unchanged and still includes the reused visits, so
    `root.mean_reward()` is now `this search's reward / all visits` and is not a
    meaningful average on a reused tree. Read `SearchResult::best_mean_reward`,
    or a child's `mean_reward()`, instead.

  Nothing in this crate reads either value, no child node is affected, and the
  chosen move is unchanged — the fix removes an inconsistency rather than
  changing a search. There is no compile error, which is why it is here.
* **`Node::choice()` returns `None` at a joint successor as well as at the
  root.** Code that walks `children()` and reads `None` as "this is the root"
  will silently skip joint successors. Check `Node::kind()`. Only reachable in a
  game with simultaneous nodes.

### Added

* `Status::Simultaneous { players }`, and `PlayerSet` / `PlayerSetIter` — a
  64-player bit set with a stable slot order, capped at
  `PlayerSet::MAX_SIMULTANEOUS` (8) participants at any one node.
* `Game::choices_for_into`, enumerating one participant's own options. Defaults
  to `choices_into`, which is right for a symmetric simultaneous phase.
* `Game::apply_joint`, resolving one simultaneous round through
  `JointChoices<'_, Choice>` — a borrowing view over the participants' chosen
  actions, so resolving a round clones nothing. There is deliberately no working
  default: applying the actions one after another is serialization, which hands
  the second mover a best response they do not have.
* `Game::SIMULTANEOUS_POLICY`, an associated const selecting
  `SimultaneousPolicy::RegretMatching` (the default) or `SimultaneousPolicy::Duct`.
  Being a const rather than a `Config` field is what folds the unchosen branch
  away.
* `Config::duct_exploration` (default 0.75), `Config::regret_matching_exploration`
  (default 0.4), and `Config::root_policy` with the new `RootPolicy` enum
  (`Sampled`, the default, or `MostVisited`). A simultaneous root returns the
  perspective player's own action, sampled from their mixed strategy under
  `Sampled` — which consumes one draw from the `rng` passed to
  `Searcher::search`, after the search loop.
* `Searcher::root_policy_into`, writing the perspective player's whole mixed
  strategy at a simultaneous root as `(choice, probability)` pairs, renormalized
  over the actions legal in the real position.
* `Searcher::reuse_joint` and `RootParallel::reuse_joint`, the tree-reuse entry
  point once every participant's action is known. Expect it to miss often: joint
  successors materialize one per iteration out of up to `prod(|A_i|)`, so
  simultaneous games effectively lose tree reuse and `reused_iterations` reads 0
  far more often than the sequential experience suggests.
* `Node::kind()` and the `NodeKind` enum (`Root`, `Choice`, `Joint`),
  `Node::reward_player()`, `Node::simultaneous_players()`, `Node::marginals()`
  and `Node::joint_arm()`.
* `Marginals<'_, C>`, a view of one player's bandit over their own actions at a
  simultaneous node: `len`, `is_empty`, `choice`, `visits`, `availability`,
  `mean_reward`, `most_visited` and `policy_into`. Arms are statistics, not
  nodes, so they are not counted by `Node::node_count` or `Node::max_depth`.

### Changed

* `RootParallel::search` merges the perspective player's **marginals** at a
  simultaneous root, keyed by `Choice`, and never merges joint successors: joint
  edges are keyed by per-tree marginal indices, and worker 1's arm 2 is not
  worker 2's arm 2. Under `RootPolicy::Sampled` with a mixing policy the merge
  draws from worker 0's rng, after the thread scope closes, so the result stays
  deterministic for a fixed thread count and seed set. The public signature is
  unchanged. Three things about that merge are worth stating outright, because
  each is a way a parallel root could otherwise disagree with a single-threaded
  one:

  * **It applies the same legality filter the single-threaded root does.** The
    merge is seeded with the perspective player's actions in the *real*
    position (`Game::choices_for_into`) and adds no arm outside that seed, so a
    merged simultaneous root cannot return an action the player does not hold,
    however the determinizations disagreed. Arms accumulate the union of every
    determinization's legal set; the filter is what keeps that union out of the
    answer.
  * **Each worker contributes exactly its own root visits.** A worker's
    normalized policy is renormalized over the arms that survive the filter
    before it is weighted, so a worker whose policy happened to sit largely on
    actions that are unplayable here still counts for its full visit count
    rather than the fraction that survived. A worker with no surviving mass
    contributes no strategy while its visits and rewards still pool.
  * **`RootPolicy::MostVisited` under a mixing policy argmaxes the merged
    strategy, not pooled raw arm visits.** Raw visits are neither
    exploration-free nor divided by availability, so ranking them would undo
    the correction `Searcher::search` applies at a single-threaded root and
    return a different action for the same position and policy. `Duct` (whose
    weight *is* the visit argmax) and every sequential root keep the pooled-visit
    argmax. Pooled visits still supply `best_visits` and `best_mean_reward`
    in every case.
* `Config::early_termination` at a simultaneous root under
  `SimultaneousPolicy::Duct` proves against the arms that are legal in the real
  position, on both sides of the comparison, so it can only stop early on a
  candidate it would actually play. It refuses to prove at all when its legality
  mask is out of step with the arm count, and the mask is rebuilt from scratch at
  the start of every search — a `Searcher` reused across positions cannot carry
  the previous position's legal set into the proof.
* `Game::ROOT_CHOICES_INVARIANT`'s root fast path (opt-in, default `false`) now
  arms only once a full expansion pass has *proved* that the tree's root arms —
  or root children, at a sequential root — are exactly the ones this position
  offers, rather than as soon as one pass has run. A root promoted by
  `reuse_subtree` or `reuse_joint` can carry arms that only ever existed in
  determinizations reached at depth; such a root now keeps paying for full
  expansion, which is correct, instead of restamping those leftovers as
  available and selecting them. A root that was never re-rooted still arms on
  the first iteration exactly as before, so no sequential search that met the
  invariant changes.
* `Config::early_termination` has no effect at a simultaneous root under
  `SimultaneousPolicy::RegretMatching`, and `StopReason::Proven` cannot occur
  there. The proof shows that the most-visited candidate cannot be overtaken,
  which says nothing about an answer drawn from a distribution. A time-budgeted
  search at such a root always runs to its deadline. No new `StopReason` variant
  was added.
* `Config::progressive_bias_weight` has no effect at a simultaneous node, where
  the prior is never evaluated: `Game::heuristic_bias` describes a state, so at a
  simultaneous node it describes a joint successor, while selection there scores
  one player's own arms. A game that tunes progressive bias and then adds a
  simultaneous node loses the term at exactly those nodes.
* `Config::max_reward` / `Config::min_reward` became load-bearing outside the
  early-termination proof. Regret matching clamps payoffs into `[0, 1]` against
  that declared range before touching a regret; it rescales nothing else, since
  its strategy is invariant under any positive rescaling of every regret. Left at
  the default `[0, 1]`, a zero-sum game paying `-1` for a loss and `0` for a draw
  gives regret matching the same number for both, at every budget, with no other
  symptom. Debug builds now assert the range at simultaneous nodes.
* `Node`'s doc comments for `mean_reward` and `cumulative_reward` now name
  `Node::reward_player()` instead of "this node's own player".

### Unchanged

* **Sequential search is byte-identical.** No node grew: `Node<C>` measures
  64 / 72 / 80 / 88 bytes for `C = u8 / u16 / usize / Vec<u16>`, exactly as in
  0.2.0, because a simultaneous node's marginals reuse the same lazily-allocated
  slot the child hash index lives in — a node is never both. Those sizes are
  pinned by a `size_of` unit test rather than by this sentence. Dispatch is
  behind `Game::SIMULTANEOUS_POLICY` and `Status`, and every simultaneous path
  is unreachable for a game that never reports the variant.
  `tests/sequential_fingerprint.rs` pins the sequential search's exact output —
  405 searches, to 12 significant figures — against values captured before this
  work began. That is a statement about *behaviour*: no timing comparison
  against 0.2.0 is claimed here, and none was run for this release.

### Known limitations

Recorded here because none of them produces an error at the point of use.

* Nothing is proved deeper than one simultaneous ply, and determinization voids
  the convergence theorems literally. `README.md` states exactly what survives.
* The availability correction that makes the extracted strategy sound under
  determinization has a residual: two arms conditioned on *different* legality
  events are not two entries of one distribution. Where arm A is always legal and
  arm B is legal half the time and strictly better whenever it is legal, the
  exact answer is all of the mass on B and `Searcher::root_policy_into` reports
  0.33 on A. Fixing it needs a statistic per legal *set* rather than per arm.
* `SimultaneousPolicy::Duct` provably does not converge at a node with a mixed
  equilibrium — decoupled UCB1 cycles rather than settling — which is why it
  is not the default. It is correct at nodes with pure equilibria and cheaper
  per arm, and `Game::SIMULTANEOUS_POLICY`'s doc carries the measurements.
* A simultaneous node expands every participant's full marginal action list on
  first visit, by design: capping or staging that expansion is exactly what
  voids the convergence property. A player with thousands of legal actions
  allocates thousands of arms there and scans them every visit — 32 bytes each,
  plus a further 16 under the default `RegretMatching`.
* `Searcher::reuse_subtree` and `Searcher::reuse_joint` behave differently in
  the two build profiles when called on the wrong kind of root: release returns
  false and drops the tree, debug panics on a `debug_assert!`. Calling the wrong
  one is a caller error rather than a miss, and the return value cannot say
  which it was. In a game with both kinds of node, branch on
  `Node::simultaneous_players()` rather than on the return value.

## [0.2.0]

No changelog was kept before this release.
