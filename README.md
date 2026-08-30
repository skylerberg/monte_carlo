# mcts

Monte Carlo Tree Search for board games, including imperfect-information games
via Information Set MCTS, and games where players move at the same time.

Built to be dropped into a real game engine without giving up the performance of
a hand-written search: no trait objects on the hot path, no allocation per
iteration beyond growing the tree, and no policy-inheritance layer to indirect
through.

## Using it

Implement `Game`, then drive a `Searcher`:

```rust
let mut searcher = Searcher::new(&state);
let result = searcher.search(&state, &ctx, player, &config, None, &mut rng);

state.apply_choice(&ctx, &result.choice, &mut rng);
searcher.reuse_subtree(&result.choice);   // next search inherits this subtree
```

The crate docs run those four lines as a doctest against a small game.
`examples/tic_tac_toe.rs` is a complete implementation of a perfect-information
game; run it with `cargo run --example tic_tac_toe`.

## What it does

* **Information Set MCTS** — a fresh determinization per iteration, and, below
  the root, an exploration term over how often a choice was *available* rather
  than how often its parent was visited, so a rarely-legal move is not mistaken
  for an under-explored one. The root is the textbook `ln(parent visits)`,
  deliberately; its own ranking is what divides by availability there.
* **max^n backup** over per-player reward vectors. Any number of players, any
  reward scale; nothing assumes two players or zero sum.
* **Simultaneous moves** — one bandit per participant over that player's own
  actions, with regret matching as the default selection rule so that a node
  with a mixed equilibrium is played mixed rather than exploitably. What is
  proved and what is not is spelled out below.
* **Budgets** by iterations or wall clock — one of the two is required — either
  of which an external `AtomicBool` can cut short. Plus early termination once
  the remaining iterations provably cannot change the answer, which an
  `iterations: u32::MAX` run-until-cancelled budget switches off in practice.
* **Tree reuse** across moves, and root parallelism behind the `parallel`
  feature.
* **Progressive bias** from a game-supplied prior, evaluated once per child.

## Simultaneous moves

A state can report `Status::Simultaneous { players }`: every player in the set
chooses independently and concurrently, and the tuple of their actions
determines the successor. Implement `Game::choices_for_into` to enumerate one
player's own options and `Game::apply_joint` to resolve the tuple — not
`apply_choice` in a loop, which is serialization and hands the second mover a
best response they do not have. Pick the selection rule with the
`Game::SIMULTANEOUS_POLICY` associated const:

| policy | per-arm state | converges? |
|---|---|---|
| `RegretMatching` (default) | 48 B | ε-Hannan-consistent; see the claim below. |
| `Duct` | 32 B | **No — proved.** Converges to a pure profile; cycles where the equilibrium is mixed. |

The node keeps one bandit per participating player over that player's own
marginal action list — decoupled UCT. Selection costs `sum(|A_i|)` statistics
instead of `prod(|A_i|)`, and joint successors materialize lazily, at most one
per iteration. `RegretMatching` is bandit-feedback regret matching [5], and it
needs exactly the information DUCT already has: no counterfactual row, no
joint-payoff matrix, no imputation heuristic, and therefore zero extra hash
probes per visit. The price is variance, because the regret estimate is
importance-weighted by the probability the arm was actually sampled with —
the outcome-sampling estimator of [4] — rather than bootstrapped from the joint
children.

The sampling distribution mixes in a uniform exploration floor, which is what
bounds that importance weight. It decays as `gamma_0 * t^(-1/4)` in the node's
own visit count, clamped to `[0.005, 0.5]`, with `gamma_0` set by
`Config::simultaneous.regret_matching_exploration` (default 0.4). The exponent is load
bearing: for a floor decaying as `t^-a` the average-regret bound goes as
`T^(a - 1/2) / gamma_0`, so at `a = 1/2` the bound — and the search — stops
depending on the budget at all. `Config::simultaneous.duct_exploration` (default 0.75) is the
separate UCB1 constant `Duct` reads; a sampling probability and a rate on the
reward scale are not interchangeable numbers.

At a simultaneous root the search returns the perspective player's **own**
action, never a joint tuple, sampled by default from the exploration-free,
availability-normalized average strategy (`Config::simultaneous.root_policy`). A caller who
does nothing therefore plays a mixed strategy. `Searcher::root_policy_into`
hands back that whole distribution, renormalized over the actions legal in the
real position, and `Searcher::reuse_joint` is the tree-reuse entry point once
every participant's action is known — though reuse mostly misses here, since
joint successors materialize one per iteration out of up to `prod(|A_i|)`.

### The claim, stated exactly

**Under `SimultaneousPolicy::RegretMatching`, in a perfect-information game with
a fixed action set per node, at a depth-1 simultaneous node:** each player's
time-averaged strategy converges almost surely to the set of coarse correlated
equilibria of that node's matrix game. In the two-player zero-sum case the pair
of averages converges to a Nash equilibrium. The exploration floor decays, so
the residual exploitability from exploration shrinks with the budget rather than
sitting at a floor proportional to a constant `gamma`.

**Deeper than one simultaneous ply, nothing is proved.** Kovařík & Lisý [3],
Theorem 4.1 exhibits a zero-sum perfect-information simultaneous-move game and a
Hannan-consistent algorithm *with* guaranteed exploration for which plain
SM-MCTS almost surely converges outside the set of 1/5-Nash equilibria. The two
known repairs — averaged backup (SM-MCTS-A) and ε-unbiased payoff observations —
are **not** implemented here: SM-MCTS-A needs a per-node reward *vector*, which
would add `size_of::<Rewards>()` to every node in every tree including purely
sequential ones, and UPO is proved for no concrete algorithm. Deeper convergence
is therefore empirical, not proved.

**With more than two players or non-zero-sum rewards**, independent no-regret
learners converge in time-average to the **coarse correlated equilibrium** set,
not to Nash. The marginals of a CCE are not in general a Nash equilibrium. Since
this crate advertises N players and non-zero-sum rewards, CCE is the honest
headline claim and Nash is the two-player zero-sum special case.

**Under determinization (ISMCTS), every theorem above stops applying literally,
for two independent reasons, and both are worth stating:**

1. **Varying action availability.** Every SM-MCTS convergence result fixes
   `A_i(h)` across all visits to `h`. Determinization varies it. That is the
   sleeping-bandits setting, where no algorithm can be no-regret against the best
   fixed *action*; the information-theoretically correct benchmark is the best
   fixed *ordering*, restricted to rounds where the action was legal. The
   corrections this crate applies — regret updated only for arms legal this
   iteration, the average strategy normalized by availability, extraction
   renormalized over the position's real legal set — target that benchmark and
   have a deterministic, severe failure mode when omitted. They are principled
   engineering, not theorems, and they leave a residual: two arms conditioned on
   *different* legality events are still not two entries of one distribution.
   Where arm A is legal always and arm B is legal half the time and strictly
   better whenever it is legal, the exact answer is all of the mass on B and the
   extraction reports 0.33 on A. Getting that right needs a statistic per legal
   *set* rather than per arm.
2. **Strategy fusion and non-locality.** A perfect-information Monte Carlo
   search converges at best to an equilibrium of the *determinization-averaged*
   game, never to an equilibrium of the true imperfect-information game. No
   simultaneous-node policy can recover the latter through determinization.

**Under `SimultaneousPolicy::Duct`, no convergence claim is made at any depth.**
Its failure is proved, not merely unproven: on biased rock-paper-scissors
decoupled UCT enters a stable cycle whose resting point is a function of the
exploration constants rather than of the game, and an opponent who observes the
skew exploits it. Randomized tie-breaking (which this crate applies at
simultaneous nodes, and only there) takes measured exploitability on that game
from 0.5–0.8 down to 0.01–0.05, but it does not make the policy converge. `Duct`
exists because it is the cheapest option and the strongest raw performer in most
published game-playing comparisons; it is the right choice when you know your
simultaneous nodes have pure equilibria.

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
crossover and why one strategy does not fit every branching factor. A
simultaneous node's marginal bandits live in that same lazily-allocated slot — a
node is never both kinds — so a purely sequential game pays nothing in memory or
in dispatch for simultaneous support.

## Features

| feature | default | effect |
|---|---|---|
| `time` | yes | wall-clock budgets. Turn it off for `wasm32-unknown-unknown`, where `Instant` panics. |
| `parallel` | no | `RootParallel`: independent trees per thread, merged at the root. |
| `serde` | no | `Config` is serialisable. |

## Tuning: the `mcts-tune` crate

A game's leaf evaluation is a vector of hand-set weights, and the honest way to
judge a change to them is to play the new numbers against the old and count.
`mcts-tune/` automates that: it proposes candidate parameter vectors, measures
each in threaded self-play against a fixed baseline, and feeds the ranking to a
black-box optimizer. It lives beside the search crate rather than inside it —
it needs threads and whole games, it is offline tooling that never ships, and a
wasm consumer should not pay for it.

```rust
let mut optimizer = CmaEs::new(&base_params, CmaParams::default());
let report = mcts_tune::run(&arena, &mut optimizer, &config, resume, |generation| {
    // persist generation.checkpoint however you like; the crate writes no files
})?;
```

Implement [`Tunable`] to map a parameter struct to a gene vector and [`Match`]
to say how a game starts and how genes become a search context. [`Ga`] and
[`CmaEs`] both implement [`Optimizer`]; CMA-ES is the better fit for a vector of
correlated continuous weights, because it learns the shape of the region worth
sampling rather than recombining coordinates independently.

**Who the candidates play decides what fitness means.** Against a fixed baseline
(`Opponents::Baseline`) a score is absolute and comparable across the whole run,
which is what you read progress from — but it has a ceiling, and runs reach it.
Once the population beats the baseline nine times in ten, every candidate scores
between 0.88 and 0.95, the gaps between them are smaller than the standard error
on each measurement, and selection is ranking noise. Underneath that sits a
second problem: candidates are rewarded for punishing the *particular* opponent
they all face, which is not the same as being strong.

`Opponents::RoundRobin` plays every candidate against every other. The field
improves with the population, so there is no ceiling and no fixed opponent to
specialise against. It is also cheaper than it looks — each game scores two
candidates, so at equal cost every candidate is measured on twice as many games.
What it costs is the absolute scale: scores are zero sum within the field, the
population mean is 0.5 in every generation by construction, and a 0.62 late in a
run was earned against much stronger opposition than a 0.62 early on. Progress
has to be measured separately, by playing a generation's output against fixed
weights. It also brings the co-evolutionary risk of cycling — three strategies
each beating one of the others can chase each other in circles — which shows up
as lively fitness with parameters that keep returning to values they already
held.

That difference reaches further than the fitness column: with a moving field the
highest score of the run is not the run's answer, because an early generation
won its score against weaker opposition. `run` reports the latest generation
under `RoundRobin` and the historical maximum under `Baseline`.

**Fitness is a win rate, so it is noisy, and that governs everything else.** The
standard error over `n` games is at worst `sqrt(0.25 / n)` — 2.5 percentage
points even at 400 games. With `sigma_f` the spread of true strength across a
generation, what the run is really ranking correlates with the truth at
`sigma_f / sqrt(sigma_f^2 + 0.25/n)`. When that is low the ranking is noise, the
bottom half gets bred, and generations move away from the optimum as readily as
towards it. Two things follow, and both are built in:

- Games per candidate matters more than candidates per generation. Below about
  200 games, a run is mostly measuring luck.
- Variance reduction is free strength. Every candidate in a generation plays the
  *same* seeds, and every matchup is played from both seats, so the comparison
  carries neither each candidate's own luck nor the first-move advantage.

**Runs are resumable.** Every [`GenerationReport`] carries a [`Checkpoint`]
holding the optimizer's own state — for CMA-ES the adapted covariance and step
size, which is the part an interruption would otherwise cost and the whole
reason to use the strategy. Persist it and pass it back as `resume`. Re-centring
a fresh optimizer on the best parameters found so far is *not* the same thing:
the covariance resets to diagonal, the step size resets, and the search starts
over from a good point rather than continuing. A resumed run reproduces the
uninterrupted one exactly, generator state included, which is what the suite
checks.

Resuming is refused rather than approximated when the checkpoint does not belong
to the run: a different strategy, a different number of genes, or — the one that
actually catches people — a different baseline. Fitness is a win rate *against
the baseline*, so resuming with the previous run's output as the seed parameters
would restart the scale at even money and make every number after the resume
incomparable with every number before it, while looking in the log exactly like
a run that collapsed.

Two failure modes worth knowing before trusting a result. Parameters that are
symmetric in the real game but come back differentiated are the run fitting
noise, not signal. And an evaluation of the form `tanh(weighted_sum / scale)` is
unchanged when every weight *and* `scale` are multiplied together, so leaving
both in the gene vector spends games wandering a direction that cannot change a
single game — pin one before starting.

## References

Most of the algorithm, and the vocabulary, comes from Browne et al.'s survey [1].
The Information Set treatment follows Cowling, Powley and Whitehouse [2]. The
simultaneous-move claim, and its limits, are Kovařík and Lisý's [3]; the
selection rule is Hart and Mas-Colell's [5] on the importance-weighted feedback
of Lanctot et al. [4].

[1] Browne, C. et al. (2012). *A Survey of Monte Carlo Tree Search Methods.*
IEEE Transactions on Computational Intelligence and AI in Games, 4(1), 1–43.

[2] Cowling, P., Powley, E. & Whitehouse, D. (2012). *Information Set Monte Carlo
Tree Search.* IEEE Transactions on Computational Intelligence and AI in Games,
4(2), 120–143.

[3] Kovařík, V. & Lisý, V. (2019). *Analysis of Hannan Consistent Selection for
Monte Carlo Tree Search in Simultaneous Move Games.* Machine Learning, 109(1),
1–50. doi:10.1007/s10994-019-05832-z

[4] Lanctot, M., Waugh, K., Zinkevich, M. & Bowling, M. (2009). *Monte Carlo
Sampling for Regret Minimization in Extensive Games.* Advances in Neural
Information Processing Systems 22, 1078–1086.

[5] Hart, S. & Mas-Colell, A. (2001). *A Reinforcement Procedure Leading to
Correlated Equilibrium.* In Debreu, G., Neuefeind, W. & Trockel, W. (eds),
Economics Essays: A Festschrift for Werner Hildenbrand, 181–200. Springer.
