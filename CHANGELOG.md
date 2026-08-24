# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

* **A root's answer is its best-valued candidate, not its most-visited one.** A
  choice is selectable only on the iterations in which it was legal, so
  `visits <= availability` is a hard cap and an argmax over raw visits ranks by
  legality rate as much as by value: a move offered by a third of the
  determinizations and played on nearly every iteration that offered it lost to
  a move offered by all of them and played on two thirds, at any budget. Every
  argmax-family root now ranks by the same rule — the sequential root,
  `SimultaneousPolicy::Duct`'s marginal, `Marginals::policy_into`, the new
  `Marginals::leader`, and `RootParallel`'s pooled merge — and the rule is the
  **mean reward**, with the selection rate `visits / availability` breaking ties
  and a candidate the search selected at least 32 times outranking one it did
  not. Not the selection rate itself, which was the obvious correction and does
  not work: a rate saturates at 1.0, and selection pushes any rarely-offered
  candidate against that ceiling whatever it is worth, so a decoy offered one
  determinization in a hundred scores 97/97 = 1.000 and beats a genuinely best
  move at 9873/9999 = 0.987. The mean has no ceiling to hide behind, and the
  32-selection bar is what stops one lucky rollout from winning the root. See
  `src/rank.rs` for the derivation and for what the rule gives up.
  `Marginals::most_visited` still reports the raw count.
* **`Config::early_termination` proves much less, and it is now a bound rather
  than an estimate.** The answer is a mean reward and a mean moves with the
  *rewards*, so no arithmetic over the visit counts bounds it — and
  `Config::min_reward` / `Config::max_reward` are a declaration the crate never
  clamps to, so they cannot supply the missing interval either. What is still
  proved is the evidence bar: a rival that cannot reach 32 selections with the
  iterations left can never leave the tier below the leader's, whatever any
  reward turns out to be. That caps the saving at 31 iterations however large
  the budget, and it is a no-op at a root whose rivals are all well sampled;
  where a 5 000-iteration search used to stop at 2 506 it now stops at 4 974.
  Because the saving is capped, so is the price: the proof is skipped outright
  until the last 31 iterations rather than scanning every root candidate on
  every one of them, which is worth 24% of the wall clock of a 20 000-iteration
  search at a 5 000-child root.
  `StopReason::Proven` still means what it says. Two further refusals: a leader
  that has not cleared the bar proves nothing about anyone, and a position that
  offers a legal move the tree has no candidate for is never settled, because a
  candidate created with `n` iterations left can spend all of them, clear the
  bar, and be ranked on a mean nothing in hand constrains.
* **`Node::availability` is public.** How many iterations offered a choice is
  half of the rate that breaks the root ranking's ties, and it was not reachable
  from `Searcher::tree`, so a caller could neither reproduce the ranking nor tell
  whether their `determinize_into` varies the legal set at all.
* **Breaking: `Rewards::zero()` is now `Rewards::uniform(f64)`.** The crate
  fabricates a payoff for a node it cannot descend from — one nobody's game
  produced — and credits it to every node on the path, so it has to be a value
  the caller's declared range contains. A zero is not that value for a game
  paying in `[1, 2]`: the fabricated reward dragged those nodes' means below the
  range they are read against, and at a simultaneous ancestor it tripped the
  reward-range assertion in a message blaming the game for a payoff the search
  had invented. All three degenerate sites now score `Config::min_reward`, and
  the sequential one gained a debug assertion of its own rather than being
  reported, if at all, by that misdirected message. Every external
  `impl Rewards` needs the one-line migration
  `fn uniform(value: f64) -> Self { [value; N] }` — or whatever builds your
  reward vector from one number; the blanket `[f64; N]` impl is already updated.
  A game whose declared floor is zero sees no behaviour change; one declaring
  `[-1, 1]` now scores a degenerate node at `-1` rather than `0`, which is the
  point — a fabricated payoff has to sit inside the range the game declared,
  and a zero was only ever right for a game whose floor happened to be zero.
* **An empty or inverted reward range is refused.** `Searcher::search` and
  `RootParallel::search` now panic on `min_reward >= max_reward`, which is a
  config a caller who never set the range on a game paying a constant may be
  holding today. It was never honoured: `normalize_reward` divides by the span,
  so regret matching was fed a constant `0.5` for every payoff — a driftless
  random walk over the regrets, measured at roughly *twice* the exploitability
  of playing uniformly at random — and the reward-range assertion was
  short-circuited in exactly that case, so nothing said so. `Duct` fares no
  better: its tie tolerance is a fraction of the same span. The pool refuses on
  its own thread rather than as N workers panicking at once, and disarms every
  worker's retained tree before it does, so a refused search leaves the pool as
  a completed one does.

### Fixed

* **The progressive-bias prior is read in the mover's currency, not the
  searching player's.** `Game::heuristic_bias` was evaluated for `perspective`
  and then added to a child whose mean the crate keeps in the currency of the
  player who moved into it, so at every opponent node the prior was pointed at
  the reply that pays *you* — the optimistic-opponent error the crate refuses to
  make at a simultaneous root, made silently at a sequential one. Measured on a
  two-ply game where player 1's reply is worth 1.0 to whoever it is chosen for:
  at weight 100 and 500 iterations, player 1's node spent 102 of its 246 visits
  helping player 0 and the search returned the losing move on all 16 seeds. Read
  for the mover, the same search spends 1 visit there and answers correctly on
  all 16. The hook's second parameter is renamed `mover` and documented as being
  on the same scale as `Rewards::reward(mover)`. A game that leaves
  `progressive_bias_weight` at its default of `0.0` never evaluates the prior and
  sees no change, and neither does one that ignored the argument and derived the
  player from its own state — an escape hatch that fails at a terminal successor,
  which names nobody, and in a game where one player moves twice.
* **A non-finite `Config` knob is refused instead of silently switching the
  search off.** A NaN or infinite `exploration_constant`, `progressive_bias_weight`,
  reward bound or simultaneous exploration constant makes every UCB value NaN,
  and NaN loses every comparison `select` makes — so `select` answered `None` at
  the first fully-opened node and every remaining iteration bumped the root's
  visit count and grew nothing. A 2 000-iteration search on the trap fixture
  returned the losing choice off a three-node tree. `Config` derives `serde`, and
  TOML, YAML and JSON-with-a-divide all spell `nan` and `inf`, so this arrives
  without anyone typing `f64::NAN`. It now takes the same path as an empty
  reward range: refused by `Searcher::search` and by `RootParallel::search`,
  which disarms every worker's retained tree before it refuses.
* **The sequential root's answer is filtered against the real position.** The
  tree accumulates every choice any determinization offered, and `search` used
  to return that union's leader — so under determinization, or after
  `reuse_subtree` on a game with a chance event, it could return a move the
  player does not hold, stamped `StopReason::Proven`. `SearchResult::choice` is
  now always legal in the `state` the search was called on, and the
  early-termination proof ranks only the legal candidates. The simultaneous root
  already did this; `RootParallel`'s sequential merge now does it too.
* **`Searcher::root_policy_into` no longer returns `true` with an empty
  policy.** Where the tree's arms for that slot and the position's legal actions
  are disjoint — an ordinary state at low budgets when a player's legal set is
  hidden — it wrote nothing and still promised pairs summing to 1. It now falls
  back to a uniform distribution over the player's legal actions, which is the
  distribution `Searcher::search` already drew its answer from, and returns
  `false` when the player has no legal action at all.
* **A root child's availability is counted on the `ROOT_CHOICES_INVARIANT` fast
  path.** That path skips expansion entirely, so availability froze at whatever
  the first pass wrote while visits went on accumulating — after a
  `reuse_subtree`, at values progressive expansion left at a node that was not
  the root. Nothing read the pair before; the root ranking now divides one by
  the other to break its ties, and `Marginals::availability` reports it. The
  cached `ln(availability)` moves with the count on that path too, so no future
  reader of the cache can find it frozen.
* **`StopReason::Proven` now means what it says.** The early-termination proof
  is a bound and never an estimate of how the remaining iterations would
  probably be spent — see the entry on `Config::early_termination` above for the
  bound that survives the root ranking change. The adversarial UCB1 replay that
  used to run at a sequential root every 1024 visits is gone. It was not an upper bound — it paid
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
* **`RootPolicy::MostVisited` answers with an arm where the root policy puts
  mass on none of them.** Regret matching's `sigma` is exactly zero for an arm
  dominated wherever it is legal, so its `strategy_sum` never leaves zero while
  the exploration floor goes on handing it visits — a legal set carrying no
  strategy mass is a state the search reaches and stays in, not a startup
  transient. `duct::best_arm` returned `None` there and the caller fell through
  to its own uniform draw over the position's choice list, which reported
  `best_visits = 0` and `best_mean_reward = 0.0` for a move the tree held real
  statistics for, and consumed an rng draw inside a root policy documented
  deterministic. It now falls back to the crate's root ranking over the legal
  arms, as `root_strategy_into` and `sample_root_arm` already fell back to
  uniform-over-legal, so the reported statistics belong to the action returned.
* **A `RootParallel` merge breaks a tie the way the search it pools does.** Both
  sides rank by the same rule and both keep the incumbent on a tie, so a tied
  root is settled by the order the candidates are scanned in — and the merge
  scanned the position's own choice list while a `Searcher` scans the tree,
  which under determinization is a different order. `RootParallel::new(1, ..)`
  therefore disagreed with a directly-seeded `Searcher` on an identical tree, at
  any budget: exact ties survive one. The merge is now seeded in the first
  worker's discovery order, then in the position's for anything that worker
  never met. Both root kinds moved — a `Duct` simultaneous root and a sequential
  one — so a pooled search over a tied root can return a different (equally
  ranked) action than it did before.
* **The reward-range assertion covers both simultaneous policies, and no longer
  accuses a game paying its own declared maximum.** It sat inside the
  `RegretMatching` backup on the reading that the range reaches the search
  through that policy's clamp and nowhere else. It does not: `Duct` measures its
  tie tolerance against the width of the range, which is the one input that
  turns decoupled UCB1 into a uniform random move picker, and it was the policy
  nothing checked. The bound is now `Config::max_reward` itself with the `1e-9`
  slack `DESIGN.md` §6 specifies, not a `min_reward + span` reconstruction of
  it: for `[-10.0, -3.9]` that sum is `-3.9000000000000004`, so a game paying
  exactly its declared maximum was accused of the defect the assertion exists to
  report — roughly one arbitrary range in twenty, plain two-decimal ones
  included.
* **The example runs in a debug build.** `cargo run --example tic_tac_toe` died
  on its first `status()` call with "attempt to subtract with overflow", before
  printing a line. `winner` closed each line with `then_some(first - 1)`, and
  `bool::then_some` takes its argument by value — evaluated before the flag is
  looked at — so an empty board computed `0u8 - 1`. A release build
  wrapped it to 255 and threw the value away, which is the build the example's
  own instructions and the README named — the crate shipped a sample nothing
  could run as written. It is `then(|| first - 1)` now, the example carries a
  unit test that fails in any build with overflow checks on, and CI runs the
  binary itself in the dev profile.
* **The example prints a value rather than a "win rate".** It labelled
  `best_mean_reward` that way, and on a game paying 1.0 for a win and 0.5 for a
  draw it is an expected score: `win rate 0.500` at a dead-drawn position, and
  `win rate 0.000` on the last forced move, where the zero is the placeholder
  `StopReason::SingleChoice` returns without running an iteration rather than
  anything measured. Forced moves now say `forced` and the rest say `value`.
* **The one code sample the crate ships compiles and runs.** It called
  `state.apply(&result.choice)`; there is no `apply`, the method is
  `apply_choice(ctx, choice, rng)`. The rustdoc copy carried an `ignore` fence —
  the only one in `src/` — and the README is never `include_str!`d, so nothing
  compiled either copy and the sample rotted through two releases. The
  crate-docs copy is a real doctest now, over a small hidden `Game`, so
  `cargo test` fails if those four lines stop compiling; the README carries the
  same four lines.
* **Three doc sites over-claimed the ISMCTS exploration rule.** The crate docs,
  the README and `Marginals::availability` stated the availability denominator
  unqualified, while `select` scores a *root's* children against the root's own
  visit count and only the levels below it against their availability. The code
  is right and deliberate — nothing maintains `ln_availability` on a root child,
  and the root's own ranking is where availability divides — so the three claims
  are qualified and `select` records the coupling rather than inviting a fix.
* **`Marginals::policy_into`'s masking recipe no longer divides by zero.** It
  told a caller holding a narrower legal set to zero the illegal entries and
  renormalize, and said `Searcher::root_policy_into` "does that for you". Under
  `SimultaneousPolicy::Duct` the vector is one-hot at the leader over every arm
  the tree holds, so an illegal leader leaves `[0, 0, 0]` to be normalized by
  0.0 — and `root_policy_into` is not that rescaling: it recomputes the leader
  over the legal arms and can name a different action. Both docs now say so, and
  `Marginals::policy_masked_into` is the extraction to reach for.
* **The budget list no longer promises a cancel-flag-only budget.** The crate
  docs and the README listed a cancellation flag as one of three budgets, and
  `Searcher::search` refuses a config carrying neither an iteration count nor a
  time limit: the flag is polled inside a loop that a budget has to start, so it
  cuts a budget short rather than being one. Both lists say that now, and
  `Config::iterations` documents `u32::MAX` as the run-until-cancelled spelling
  together with what it costs — `early_termination` compares the iterations left
  against the evidence bar, so against that budget it never fires.
* **`Game::Choice`'s `Eq` is documented as full value identity.** The tree
  stores the first `Choice` it ever saw for an edge and thereafter replays *that
  value* into `apply_choice` / `apply_joint` and hands it back as
  `SearchResult::choice`. A lawful but coarser `Eq` — one ignoring data the
  choice carries, such as the determinization it was built for — therefore feeds
  one world's data into another world's state and into the caller's answer, with
  nothing to catch it. It is now a stated precondition rather than an
  implementation detail.

### Performance

Behaviour is unchanged by every item here: the sequential fingerprint is
byte-identical and node counts are bit-identical in every configuration.

* **A simultaneous node's arm index survives a newly discovered action**, which
  turns expansion from `O(A^2)` into `O(A)` in the arm count. The index's
  payloads were global positions into the flat arm array, so the insert that
  `grow_slot` performs moved every later one and the whole index was discarded —
  on a first visit, once per action, with a linear scan over the arms in
  between, and again for every action a later determinization reveals. Payloads
  are now `(slot, slot-relative position)`, which an insert cannot invalidate,
  and `grow_slot` adds the one new arm — and builds the index in the first
  place, when the arm it adds is the one that crosses
  `Game::CHILD_INDEX_THRESHOLD`, since a node holds no arms before its first
  expansion pass and so cannot already be over the threshold when that pass
  begins. Measured on an Information Set MCTS shape (two players, 50 of a pool
  of 2,000 actions legal per determinization, 2,000 iterations): **5,477,504
  `Choice` comparisons before, 296,186 after**, and the cost is now flat in the
  pool size rather than quadratic in it. On a node holding a fixed 512 actions
  per player, whose whole arm set is therefore discovered on one visit: **2,597
  comparisons over two visits, against 133,376** if the index is built only on
  entry to an expansion pass rather than during the growth that creates the
  arms. `tests/arm_index.rs` holds both there.
* **Only `SimultaneousPolicy::Duct` pays for `ln(availability)` on arms.** It is
  the sole reader of the cached logarithm, and the default policy is
  `RegretMatching`, which was paying a libm call per arm per visit — one for
  every action of every participant at every simultaneous node on the descent —
  to maintain a number nothing read. Measured on two players x 20 actions x 3
  plies, 5,000 iterations: **414,200 `ln` calls before, 0 after** under
  `RegretMatching`, unchanged under `Duct`.
* **A root's children no longer cache a denominator `select` does not read.**
  `select` scores a root's children against the root's own visit count and reads
  a child's `ln_availability` only at a non-root parent, so the root paid one
  libm call per legal choice per iteration — on both the enumerating path and
  the `ROOT_CHOICES_INVARIANT` fast path. The availability *count* is still
  bumped, because the root ranking divides by it. Measured on a 30-choice root,
  5,000 iterations: **149,970 `ln` calls before, 0 after**, on either path.
* **Root-parallel workers are cache-line aligned.** `RootParallel` kept its
  per-worker `Searcher` and rng in one packed `Vec`, at a 296-byte stride, so
  three of four worker boundaries fell inside a 128-byte line: two threads
  writing one line every iteration, in the one structure whose premise is that
  workers share nothing. Each worker is now a `#[repr(align(128))]` struct, at a
  cost of 88 bytes of padding per worker.

### Added

* **`Marginals::policy_masked_into`.** This player's strategy restricted to the
  arms legal in the real position and normalized over them, with the mask
  supplied by the caller. `Searcher::root_policy_into` is the same extraction at
  a root, where the searcher can enumerate the mask itself; below a root this is
  the only sound one, because rescaling `Marginals::policy_into` afterwards
  normalizes a vector of zeros whenever `Duct`'s leading arm is one the position
  does not offer. "The same extraction" is literal: the root form, the sampled
  form, the deterministic form and this one all run `duct::strategy_into` over
  the same arm slices, and `Marginals::leader` is `duct`'s root ranking rather
  than a second spelling of it, so the guarantee `root_weight` was written to
  give — that these forms cannot drift apart — now covers the marginal view too.
* `Game::advance` and `Game::determinize_into` document the root contract: an
  `advance` at the root may resolve decisions the tree does not model, but not
  the one being searched, and a determinization may change what is legal but not
  who acts. Debug builds enforce both on every determinization. Release
  behaviour is unchanged.

### Known limitations

* **An over-declared reward range still degrades `Duct` selection, and nothing
  detects it.** `Duct` admits arms within 1% of the range's *width* of the
  leading arm into a uniform tie draw, so a range declared much wider than the
  payoffs really are widens that pool until every visited arm is in it:
  measured on a two-arm game with a strictly dominant row and payoffs in
  `[0, 1]`, 0.999 of the arm visits land on the dominant row at
  `max_reward = 1`, 0.998 at `10`, and at `200` the search answers the *wrong*
  arm with a 0.506 share — chance. This release documents it on
  `Config::max_reward` and asserts, under both policies, that observed payoffs
  fall inside the declared range — which catches a range that is too *narrow*
  and cannot catch one that is too wide, because an over-declared range is one
  every payoff satisfies. Deriving the tolerance from the spread of the values
  observed at the node is not the fix: it lets one arm with a terrible estimate
  widen the band the good arms are compared inside. Declare the range your game
  actually pays in.

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
