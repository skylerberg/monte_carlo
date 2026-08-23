# Simultaneous-move support (DUCT + regret matching) — locked implementation spec

Status: **locked, with the corrections in §0 applied.** Every decision in this
document was final at the time it was written; five of them turned out to be wrong
and were corrected during implementation. Where §0 and a later section disagree, §0
is what shipped.

Target version: `0.2.0` → **`0.3.0`**.

---

## 0. Corrections to this specification

These are changes to *decisions*, not to wording. Each was measured before it was
made, and each replaced something below that is now marked in place.

**0.1 — The exploration floor decays as `gamma_0 * t^(-1/4)`, not `gamma_0 / sqrt(t)`
(§4.3, §1.2).** The specified schedule does not converge and does not improve with
budget. Backup divides the observed payoff by the sampling probability, and the floor
is what bounds that importance weight; for a floor decaying as `t^-a` the
average-regret bound carries `T^(a - 1/2) / gamma_0`, so at `a = 1/2` the bound stops
depending on the budget entirely. Measured on `BiasedRps` through `select_marginal`
and `credit_marginals`, 64 seeds times both slots, mean exploitability of the
extracted strategy:

| floor | 10k | 50k | 200k | 800k |
|---|---|---|---|---|
| `0.1 * t^-1/2` (this spec) | 0.070 | 0.050 | 0.038 | 0.028 |
| constant 0.05 | 0.021 | 0.010 | 0.0082 | 0.0077 |
| `0.4 * t^-1/4` (shipped) | 0.019 | 0.013 | 0.0071 | 0.0045 |

Uniform play is exploitable for 0.083 on that game. The first row is the spec's own
schedule at its own default, and it is a search that has learned very little.

**0.2 — §4.3's justification for decaying was wrong in both halves, and the
conclusion survives anyway.** The spec argued that a *constant* floor buys a
`gamma * depth`-Nash floor, and (in the same section) that accumulating the pre-floor
`sigma` into `strategy_sum` removes the floor from the extracted strategy exactly, so
the objection is answered. Both cannot be true, and the second is the one that is
wrong as an argument: removing the floor from the *average* is arithmetic, and it
does nothing about each player learning a best reply to the opponent's *played*
mixture, floor included. The fixed point of the self-play dynamic is an equilibrium
of the perturbed game, at depth 1, after exploration removal. That is why the
constant-0.05 row above flattens out around 0.008 while the decaying row keeps
falling. `strategy_sum += sigma` is still right and is unchanged; only the claim made
for it is corrected.

**0.3 — `Config::simultaneous_exploration` is split into `Config::duct_exploration`
and `Config::regret_matching_exploration` (§2.3, §4.3).** One field cannot be both a
UCB1 constant on the reward scale and a sampling probability. Measured on `BiasedRps`,
16 seeds times both slots, mean and worst exploitability of `Duct`'s *visit*
distribution at 50k: 0.209 / 0.500 at `c = 0.05`, 0.042 / 0.235 at `c = 0.1`,
0.032 / 0.043 at `c = 0.75`, 0.014 / 0.019 at `c = 2.0`. Regret matching's optimum is
a floor starting at 0.4; `Duct`'s is anything from about 0.5 up. The shared default of
0.1 was on the wrong side of the cliff for **both** policies. Defaults are now 0.75
(matching `Config::exploration_constant`, and deliberately not the `BiasedRps`
optimum, since `Duct` is for nodes with *pure* equilibria) and 0.4.

**0.4 — `RootPolicy::MostVisited` at a simultaneous root is an argmax over the same
weights §4.8 extracts, not over raw arm visits (§4.7 step 4).** Under regret matching
raw visits are neither exploration-free nor divided by availability, and §4.8 says
both corrections are mandatory. Without them an arm that is legal more often is
returned over an arm that strictly dominates it wherever both are legal. Under `Duct`
the weight *is* the visit argmax, so nothing changes there.

**0.5 — `Duct`'s tie pool is built in a second pass, against the final maximum
(§4.3).** The specified single pass compares each arm against the maximum *so far*
and never prunes the pool when that maximum rises, so any ascending chain of arms
spaced under `tol` all land in one pool: twenty such arms are drawn uniformly and the
true argmax is picked one time in twenty. The error that admits is
`(arms - 1) * TIE_TOLERANCE * span`, unbounded in the arm count.

**Two things deliberately not changed.** §4.8's `strategy_sum / availability` has a
residual — arms conditioned on different legality events are still not two entries of
one distribution, and where arm A is always legal and arm B is legal half the time and
strictly better whenever it is, the extraction reports 0.33 on A instead of 0. Fixing
that exactly needs a statistic per legal *set* rather than per arm, and §1.2 already
states that the sleeping-bandit corrections are principled engineering rather than
theorems. The residual is now written down in `root_strategy_into`'s doc comment
instead of being implied away. Separately, `RootParallel::merge` (§5.2) applies no
legality filter to the merged strategy because it has no state to filter against,
while the single-threaded path (§4.7 step 2) does; that is a real defect and it is
outside the files this correction covers.

---

## 1. Overview and the game-theoretic claim

### 1.1 What is being added

A third node kind. Today every non-terminal node has exactly one player to act
(`Status::Active { player }`). This adds `Status::Simultaneous { players }`, where
every player in the set chooses independently and concurrently and the joint tuple
of their actions determines the successor.

The node keeps **one bandit per participating player over that player's own
marginal action list** — decoupled UCT (DUCT). Selection costs `sum(|A_i|)`
statistics instead of `prod(|A_i|)`; successor (joint) children materialize lazily,
at most one per iteration, exactly as the SM-MCTS template specifies.

Two selection policies ship:

| policy | per-arm state | joint lookups per visit | converges? |
|---|---|---|---|
| `SimultaneousPolicy::Duct` | 32 B (`ArmStats`) | 0 | **No — proved.** Converges to a pure profile; cycles in mixed-equilibrium games. |
| `SimultaneousPolicy::RegretMatching` (default) | 48 B (`ArmStats` + `ArmPolicy`) | 0 | ε-Hannan-consistent; see §1.2. |

`RegretMatching` is bandit-feedback regret matching (Hart & Mas-Colell 2001) with a
decaying exploration floor and the sleeping-bandit availability correction. It needs
exactly the information DUCT already has — no counterfactual row, no joint-payoff
matrix, no imputation heuristic, and therefore **zero extra hash probes per visit**.

The root returns the perspective player's own action, **sampled** from the
exploration-free, availability-normalized average strategy by default. A caller who
does nothing therefore plays a mixed strategy.

### 1.2 The claim, stated exactly

Ship these words (or a faithful compression of them) in `lib.rs`, `README.md`, and
the `SimultaneousPolicy` doc comment. Do not soften them.

**Under `SimultaneousPolicy::RegretMatching`, in a perfect-information game with a
fixed action set per node, at a depth-1 simultaneous node:** each player's
time-averaged strategy converges almost surely to the set of coarse correlated
equilibria of that node's matrix game. In the two-player zero-sum case the pair of
averages converges to a Nash equilibrium; the exploration floor decays (see §0.1
for the schedule and §0.2 for why it decays), so the residual exploitability from
exploration shrinks with the budget rather than sitting at a floor proportional to a
constant `gamma`.

**Deeper than one simultaneous ply, nothing is proved.** Kovařík & Lisý (2019),
Theorem 4.1 exhibits a zero-sum perfect-information simultaneous-move game and a
Hannan-consistent algorithm *with* guaranteed exploration for which plain SM-MCTS
almost surely converges outside the set of 1/5-Nash equilibria. The two known
repairs — averaged backup (SM-MCTS-A) and ε-unbiased payoff observations — are
**not** implemented here (SM-MCTS-A needs a per-node reward *vector*, which would
add `size_of::<Rewards>()` to every node in every tree including purely sequential
ones; UPO is proved for no concrete algorithm). Deeper convergence is therefore
empirical, not proved.

**With more than two players or non-zero-sum rewards**, independent no-regret
learners converge in time-average to the **coarse correlated equilibrium** set, not
to Nash. The marginals of a CCE are not in general a Nash equilibrium. Since this
crate advertises N players and non-zero-sum rewards, CCE is the honest headline
claim and Nash is the two-player zero-sum special case.

**Under determinization (ISMCTS), every theorem above stops applying literally, for
two independent reasons, and both must be stated:**

1. **Varying action availability.** Every SM-MCTS convergence result fixes `A_i(h)`
   across all visits to `h`. Determinization varies it. That is the sleeping-bandits
   setting, where no algorithm can be no-regret against the best fixed *action*; the
   information-theoretically correct benchmark is the best fixed *ordering*,
   restricted to rounds where the action was legal. The corrections this design
   applies — regret updated only for arms legal this iteration, the average strategy
   normalized by availability, extraction renormalized over the position's real
   legal set — target that benchmark and have a deterministic, severe failure mode
   when omitted (§4.6). They are principled engineering, not theorems.
2. **Strategy fusion and non-locality.** A perfect-information Monte Carlo search
   converges at best to an equilibrium of the *determinization-averaged* game, never
   to an equilibrium of the true imperfect-information game. No simultaneous-node
   policy can recover the latter through determinization.

**Under `SimultaneousPolicy::Duct`, no convergence claim is made at any depth.** Its
failure is proved, not merely unproven: on biased rock-paper-scissors decoupled UCT
enters a stable cycle whose resting point is a function of the exploration constants
rather than of the game, and an opponent who observes the skew exploits it.
Randomized tie-breaking (which this crate applies at simultaneous nodes, and only
there) takes measured exploitability on that game from 0.5–0.8 down to 0.01–0.05,
but it does not make the policy converge. `Duct` exists because it is the cheapest
option and the strongest raw performer in most published game-playing comparisons;
it is the right choice when you know your simultaneous nodes have pure equilibria.

### 1.3 Design decisions where the reviewing judges disagreed

Each of these is decided here, with the one-line reason.

- **Default policy: `RegretMatching`, not `Duct`.** The stated goal is optimal play;
  a default that provably cycles does not meet it, and bandit-feedback RM costs zero
  joint lookups so the throughput objection to a converging default does not apply.
- **Bandit-feedback RM, not full-matrix / counterfactual RM.** Full-matrix RM needs
  `sum(|A_i|)` joint-child probes per node per visit against a doubly-sparse row
  (lazy joint expansion × determinization) and an unprincipled imputation knob;
  bandit RM needs neither and keeps the allocation and probe budget at zero.
- **No `Exp3Ix` variant.** Its availability correction is an invented rescaling with
  no published analysis; shipping an unvalidated heuristic inside a policy sold on a
  regret bound is dishonest in this crate's voice. RM's correction is exact.
- **No `Game::HAS_SIMULTANEOUS_NODES` master const.** It is a second source of truth
  for a fact `status()` already carries, and two things that can disagree about the
  same fact is what `select.rs:9-10` exists to forbid. The zero-cost guarantee comes
  from the dispatch location (§4) plus the `tiny/100k` regression gate (§7.6).
- **`JointKey(u64)` packed at `64/k` bits, not a `Vec<u32>` arena.** 8 keys per cache
  line at any arity, one-multiply hash, and a 2-instruction arm decode; the per-arity
  cap is asserted at arm creation with a message that names the player (§6).
- **`enum Extra<C>` fused into the existing `index` slot, not a new `NodeAux` struct
  or a second `Option<Box<_>>` field.** Measured at exactly 32 bytes for every
  `Choice` — byte-identical to today's bare `ChildIndex` box — so a wide *sequential*
  node's heap footprint does not grow at all.
- **`avail_epoch` on the arm, not a per-level legality bitmap.** It deletes a buffer
  and a memory stream; the ordering constraint that makes it sound at backup is
  pinned by a debug assertion (§4.6, §6).
- **Keep `Node::is_root()`, backed by a new `kind` byte, rather than threading an
  `is_root` bool through `select` and `expand`.** The byte is free at every measured
  `Node` size, and it leaves `select` and `expand` unchanged line for line.
- **`reuse_joint` is a new method; `reuse_subtree` is untouched.** Zero call sites
  break, and re-rooting at a joint action genuinely has a different failure rate that
  deserves its own doc comment.
- **Early termination is off at simultaneous roots except under `Duct`, and even
  there only the cheap proof runs.** The expensive proof replays UCB1, which under RM
  is a different algorithm, and for a sampled answer the question is ill-posed.

---

## 2. Public API

### 2.1 `src/game.rs`

```rust
/// A set of players, as a bitmask over player indices `0..=63`.
///
/// A bitmask rather than a slice because [`Status`] is `Copy` and returned by value
/// from the hottest call in the crate: a `&[u8]` would put a lifetime on every
/// `fn status` signature in every game, and a `Vec<u8>` would allocate once per node
/// visit. The second reason matters more. A bitmask cannot express order, so
/// participant order is defined once, here, ascending — a game physically cannot
/// hand back `[1, 0]` on one determinization and `[0, 1]` on the next and silently
/// permute a node's marginal statistics.
///
/// The 64 is a cap on player *indices* that may appear here, not on how many players
/// a game has: [`Rewards::reward`] still takes any `u8`. A single simultaneous node
/// is separately capped at [`PlayerSet::MAX_SIMULTANEOUS`] participants.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct PlayerSet(u64);

impl PlayerSet {
    /// Highest player index representable in a set.
    pub const MAX_PLAYER: u8 = 63;

    /// Most players that may act at one simultaneous node.
    ///
    /// Eight, because a joint successor is keyed by packing every participant's
    /// marginal index into one `u64`. Past eight the joint space is `prod(|A_i|)`
    /// with at least eight factors and the decoupled layer is the wrong data
    /// structure rather than merely a slow one.
    pub const MAX_SIMULTANEOUS: usize = 8;

    pub const EMPTY: Self = PlayerSet(0);

    /// Panics if any player exceeds [`PlayerSet::MAX_PLAYER`].
    pub fn of(players: impl IntoIterator<Item = u8>) -> Self;
    /// The set `{0, 1, .., n - 1}`. `PlayerSet::first_n(2)` is the two-player case.
    pub const fn first_n(n: u8) -> Self;
    pub const fn from_bits(bits: u64) -> Self;
    pub const fn bits(self) -> u64;
    /// Panics if `player` exceeds [`PlayerSet::MAX_PLAYER`].
    pub const fn with(self, player: u8) -> Self;
    pub const fn contains(self, player: u8) -> bool;
    /// Number of members.
    pub const fn len(self) -> usize;
    pub const fn is_empty(self) -> bool;
    /// Position of `player` among the members in ascending order, or `None`.
    ///
    /// This is the **slot** index, and it is the one order everything else uses:
    /// marginal storage, joint keys, [`JointChoices`] and
    /// [`Searcher::reuse_joint`]. Nothing anywhere depends on any other ordering,
    /// so the four never have to be kept in sync by hand.
    pub const fn slot_of(self, player: u8) -> Option<usize>;
    /// The player occupying `slot`, or `None`.
    pub const fn player_at(self, slot: usize) -> Option<u8>;
    /// Members in ascending player index. Allocation-free.
    pub fn iter(self) -> PlayerSetIter;
}

/// Ascending iterator over a [`PlayerSet`].
#[derive(Clone, Copy, Debug)]
pub struct PlayerSetIter(u64);
impl Iterator for PlayerSetIter { type Item = u8; /* trailing_zeros; bits &= bits - 1 */ }
```

```rust
/// Whether a state needs a decision, or is finished and worth `Rw`.
///
/// Games fold their own cutoffs into `Terminal` — a round horizon with a heuristic
/// evaluation is reported here exactly like a real game end.
#[derive(Clone, Copy, Debug)]
pub enum Status<Rw> {
    /// `player` must choose.
    Active {
        /// The player to act.
        player: u8,
    },
    /// Every player in `players` chooses at the same time, none of them seeing the
    /// others' choice, and the tuple of their actions determines the successor.
    ///
    /// Enumerate each participant's own options with [`Game::choices_for_into`] and
    /// resolve the tuple with [`Game::apply_joint`]. The set must be non-empty and
    /// hold at most [`PlayerSet::MAX_SIMULTANEOUS`] players.
    ///
    /// A node's *kind* and its exact participant set must not vary across
    /// determinizations — determinization may change which actions are legal, never
    /// who acts. An information set whose kind depends on hidden state is not one
    /// information set. Debug builds check this on every visit; in release the
    /// search reads one node's statistics as the other kind's and returns a choice
    /// belonging to the wrong player, which is a wrong move rather than a slow
    /// search.
    Simultaneous {
        /// The players choosing concurrently.
        players: PlayerSet,
    },
    /// Finished, or cut off by the game's own horizon.
    Terminal(Rw),
}
```

```rust
/// One action per participant at a simultaneous node, borrowed from the tree.
///
/// A borrowing view rather than a slice of owned choices: resolving a joint action
/// must not clone one `Choice` per participant per iteration, and the marginal that
/// owns each choice already lives in the node. The three slices it holds are
/// disjoint from the `&mut self` the game applies them to, so it costs no scratch
/// buffer and no clone.
#[derive(Clone, Copy)]
pub struct JointChoices<'a, C> { /* players, arm_choices, starts, picks */ }

impl<'a, C> JointChoices<'a, C> {
    pub fn players(&self) -> PlayerSet;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    /// The player occupying `slot`, in ascending player order.
    pub fn player_at(&self, slot: usize) -> u8;
    /// The action chosen in `slot`. Panics if `slot >= len()`.
    pub fn get(&self, slot: usize) -> &'a C;
    /// This player's action, or `None` if they are not a participant.
    pub fn for_player(&self, player: u8) -> Option<&'a C>;
    /// `(player, action)` in ascending player order.
    pub fn iter(&self) -> impl Iterator<Item = (u8, &'a C)> + 'a;
}
impl<'a, C> core::ops::Index<usize> for JointChoices<'a, C> { type Output = C; }
```

```rust
/// Which bandit each player runs over their own actions at a simultaneous node.
///
/// Sequential nodes are always UCB1; this never touches them. It is an associated
/// const on [`Game`] rather than a [`Config`] field because which equilibria a game
/// has is a property of the game, like [`Game::ROOT_CHOICES_INVARIANT`] — and
/// because a const folds the whole unchosen branch away.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SimultaneousPolicy {
    /// Regret matching on bandit feedback, with a decaying exploration floor and the
    /// availability correction determinization requires.
    ///
    /// The default. Each player's time-averaged strategy converges to the coarse
    /// correlated equilibrium set — to a Nash equilibrium only in two-player
    /// zero-sum games, and only at a single simultaneous ply. It needs exactly the
    /// information [`SimultaneousPolicy::Duct`] needs: no joint-payoff matrix, no
    /// counterfactual row, no extra hash probes. The price is variance, because the
    /// regret estimate is importance-weighted rather than bootstrapped from the
    /// joint children.
    ///
    /// See the crate docs for what determinization does to that claim; the short
    /// version is that the equilibrium reached is one of the determinization-averaged
    /// game, never of the true imperfect-information game.
    #[default]
    RegretMatching,
    /// Decoupled UCB1 over each player's own actions, on the availability count, with
    /// ties broken at random inside a tolerance.
    ///
    /// The cheapest option and the strongest raw performer in most published
    /// comparisons — and it **provably does not converge** in games whose
    /// equilibrium is mixed. On biased rock-paper-scissors it settles into a stable
    /// cycle whose resting point is a function of the exploration constants rather
    /// than of the game, and an opponent who notices the skew exploits it.
    /// Randomized tie-breaking, which this crate applies at simultaneous nodes and
    /// nowhere else, takes measured exploitability there from 0.5-0.8 down to
    /// 0.01-0.05; it does not make the policy converge. Pick this only when you know
    /// your simultaneous nodes have pure equilibria, or when you have measured it
    /// winning on your game.
    Duct,
}

impl SimultaneousPolicy {
    /// Whether this policy plays a mixed strategy.
    ///
    /// A mixed root cannot be proven settled: the answer is a draw from a
    /// distribution, so "the leader cannot be overtaken" does not imply "the returned
    /// action cannot change".
    pub const fn mixes(self) -> bool { matches!(self, Self::RegretMatching) }
}
```

Changes to the `Game` trait — **every one of them defaulted, so all eight existing
impls compile untouched**:

```rust
pub trait Game: Sized {
    // ... existing associated types and consts unchanged ...

    /// The bandit each player runs at a [`Status::Simultaneous`] node. Ignored
    /// entirely by a game that never reports one.
    ///
    /// Leave it alone unless you have measured. The default converges to a mixed
    /// strategy; [`SimultaneousPolicy::Duct`] is cheaper per iteration but converges
    /// to a pure one, which a game like rock-paper-scissors punishes by exactly as
    /// much as its equilibrium is mixed.
    const SIMULTANEOUS_POLICY: SimultaneousPolicy = SimultaneousPolicy::RegretMatching;

    /// Fill `out` with `player`'s own legal choices at a [`Status::Simultaneous`]
    /// state. Same contract as [`Game::choices_into`]: `out` is cleared by the caller
    /// and reused across calls and iterations, so this must not allocate in steady
    /// state.
    ///
    /// The default assumes every participant chooses from the same list, which is
    /// true of most simultaneous games — bidding, simultaneous card selection,
    /// rock-paper-scissors — and false of an asymmetric phase. Getting it wrong hands
    /// a player an action they do not own, and [`Game::apply_joint`] is where you
    /// find out.
    fn choices_for_into(&self, ctx: &Self::Context, player: u8, out: &mut Vec<Self::Choice>) {
        let _ = player;
        self.choices_into(ctx, out)
    }

    /// Resolve one simultaneous round: every participant's action at once. Resolve
    /// any chance events with `rng`.
    ///
    /// This is not [`Game::apply_choice`] in a loop. Applying the actions one after
    /// another is *serialization*, which hands the second mover a best response they
    /// do not have — it is a bound on the true value of the node, not an
    /// approximation of it, and a searcher built on it plays defensively as first
    /// mover and optimistically as second. There is deliberately no working default
    /// for that reason.
    fn apply_joint<R: Rng + ?Sized>(
        &mut self,
        _ctx: &Self::Context,
        _joint: JointChoices<'_, Self::Choice>,
        _rng: &mut R,
    ) {
        unimplemented!(
            "mcts: this game returns Status::Simultaneous but does not implement \
             Game::apply_joint"
        )
    }

    /// Prior for progressive bias, evaluated once when a child is first visited.
    /// Only called when `Config::progressive_bias_weight` is non-zero.
    ///
    /// Never called at a simultaneous node. The prior is evaluated at the *successor*
    /// state, so it can only ever describe a joint child — and selection at a
    /// simultaneous node scores each player's own marginals, not the joint children.
    /// `Config::progressive_bias_weight` therefore has no effect there.
    fn heuristic_bias(&self, _ctx: &Self::Context, _perspective: u8) -> f32 { 0.0 }
}
```

`Game::ROOT_CHOICES_INVARIANT`'s doc gains one paragraph:

```
    /// At a simultaneous root this is restated per participant: for every player
    /// acting there, that player's own marginal set does not vary across
    /// determinizations. It says nothing about joint successors, which are created
    /// only as they are visited and are never "fully expanded".
```

### 2.2 `src/node.rs`

```rust
/// Which of the three things a node is.
///
/// Stored rather than inferred, because `choice.is_none()` used to mean "the root"
/// and a joint successor also has no single choice. Overloading one field to answer
/// two questions is what would make those two cases indistinguishable. It costs
/// nothing: measured, `Node<C>` is 64 / 72 / 80 / 88 bytes for
/// `C = u8 / u16 / usize / Vec<u16>` both with this byte and without it — it lands in
/// tail padding at every size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum NodeKind {
    /// The root. `choice` is `None` and there is no incoming edge.
    Root = 0,
    /// Reached by one player's single `Choice`, which `choice` holds.
    Choice = 1,
    /// Reached by one action per participant of a simultaneous parent. `choice` is
    /// `None`; the marginal indices live in the parent — see [`Node::joint_arm`].
    Joint = 2,
}

impl<C> Node<C> {
    // ---- unchanged public surface, unchanged signatures ----
    pub fn visits(&self) -> u32;
    /// Mean reward for the player this node accumulates for, or 0.0 if never
    /// visited. See [`Node::reward_player`].
    pub fn mean_reward(&self) -> f64;
    /// Total reward accumulated for [`Node::reward_player`].
    pub fn cumulative_reward(&self) -> f64;
    /// This node's expanded children. At a simultaneous node these are the joint
    /// successors that have actually been visited, not every possible one.
    pub fn children(&self) -> &[Node<C>];
    /// Nodes in this subtree, including this one. Marginal bandit arms are
    /// statistics, not nodes, so they are not counted here or by
    /// [`Node::max_depth`].
    pub fn node_count(&self) -> usize;
    pub fn max_depth(&self) -> usize;

    /// The choice leading to this node.
    ///
    /// `None` at the root **and at a joint successor**, which is reached by one
    /// action per participant rather than by any one choice. Code that walks
    /// `children()` and skips `None` will silently skip every joint successor — check
    /// [`Node::kind`] rather than reading `None` as "root".
    pub fn choice(&self) -> Option<&C>;

    // ---- new ----

    /// Which of the three things this node is.
    pub fn kind(&self) -> NodeKind;

    /// The player whose reward [`Node::cumulative_reward`] accumulates.
    ///
    /// For a node reached by one player's choice that is the player who moved here,
    /// which is the max^n rule. A joint successor has no single mover and the root
    /// has no incoming edge at all, so both accumulate the reward of the player the
    /// search is running on behalf of.
    pub fn reward_player(&self) -> u8;

    /// The participants, if this is a simultaneous decision node.
    pub fn simultaneous_players(&self) -> Option<PlayerSet>;

    /// `player`'s bandit over their own actions here, or `None` if this is not a
    /// simultaneous node or `player` does not act at it.
    pub fn marginals(&self, player: u8) -> Option<Marginals<'_, C>>;

    /// The index, into `self.marginals(player)`, of the action `player` took to reach
    /// `self.children()[child]`.
    ///
    /// `None` if this is not a simultaneous node, if `player` does not act here, or
    /// if `child` is out of range.
    pub fn joint_arm(&self, child: usize, player: u8) -> Option<usize>;
}

/// One player's bandit over their own actions at a simultaneous node.
///
/// Arms are statistics, not nodes: an arm owns no children and no successor of its
/// own, and storing them as `Node`s would silently change what
/// [`Node::node_count`] and [`Node::max_depth`] report.
#[derive(Clone, Copy)]
pub struct Marginals<'a, C> { /* stats, choices, policy, mixes */ }

impl<'a, C> Marginals<'a, C> {
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    /// The action this arm plays.
    pub fn choice(&self, arm: usize) -> &'a C;
    /// Iterations in which this arm was selected.
    pub fn visits(&self, arm: usize) -> u32;
    /// Iterations at this node in which this action was legal for this player.
    ///
    /// The exploration term uses this rather than the node's visit count, so an
    /// action that is rarely legal is not mistaken for an under-explored one — the
    /// same subset-armed-bandit correction the rest of the crate applies to children.
    pub fn availability(&self, arm: usize) -> u32;
    /// Mean reward of this arm for its own player, or 0.0 if never selected.
    pub fn mean_reward(&self, arm: usize) -> f64;
    /// Index of the most-selected arm, first maximum so ties break deterministically.
    ///
    /// The right answer only where the equilibrium is pure. See
    /// [`Marginals::policy_into`].
    pub fn most_visited(&self) -> Option<usize>;

    /// Write this player's mixed strategy into `out`, parallel to the arms and
    /// summing to 1. `out` is cleared first, and is a caller buffer so polling this
    /// every turn costs nothing after the first.
    ///
    /// Under [`SimultaneousPolicy::RegretMatching`] this is the time-averaged
    /// strategy with the exploration floor already excluded and availability divided
    /// out — the only extraction with a convergence result behind it, and the one to
    /// sample. Under [`SimultaneousPolicy::Duct`] it is one-hot at the most-selected
    /// arm, deliberately: decoupled UCB1 converges to a pure policy, and its visit
    /// distribution is precisely the object an opponent exploits, so this crate will
    /// not hand it back dressed as a mixed strategy.
    ///
    /// It is averaged over every determinization in which each action was legal, so
    /// a caller holding a narrower legal set should zero the illegal entries and
    /// renormalize. [`Searcher::root_policy_into`] does that for you.
    pub fn policy_into(&self, out: &mut Vec<f64>);
}
```

### 2.3 `src/search.rs`

```rust
/// How the returned move is read off a simultaneous root. Ignored at a sequential
/// root, which is always the most-visited child.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RootPolicy {
    /// Sample the perspective player's mixed strategy, renormalized over the actions
    /// legal in the real position.
    ///
    /// The default, because a game with a mixed equilibrium has no single best move
    /// and returning one makes you exploitable by exactly as much as the equilibrium
    /// mixes. This consumes one draw from the `rng` passed to
    /// [`Searcher::search`], after the search loop — so a fixed seed still
    /// reproduces the whole run, but the returned move is not the argmax of
    /// anything. Under [`SimultaneousPolicy::Duct`] the strategy is one-hot, so this
    /// is identical to [`RootPolicy::MostVisited`].
    #[default]
    Sampled,
    /// Return the perspective player's most-selected action.
    ///
    /// Deterministic and reproducible, and exploitable wherever the equilibrium is
    /// mixed. Worth choosing when you need pinned output, or when you are
    /// deliberately exploiting an opponent you know does not best-respond. Never an
    /// argmax over joint successors: your action appears in one joint child per
    /// opponent action, so the best *pair* is not the best action.
    MostVisited,
}

pub struct Config {
    // ... seven existing fields unchanged ...

    // CORRECTED, §0.3: this single field became two. Neither is a reuse of
    // `Config::exploration_constant`: at a sequential node the constant sets how
    // fast the bandit concentrates and the limit does not depend on it, while at a
    // simultaneous node the players' bandits are coupled and the cycle two decoupled
    // UCB1 players settle into is a function of their constants.

    /// UCB1's exploration constant at a simultaneous node, under
    /// [`SimultaneousPolicy::Duct`]. Same units and scale as
    /// [`Config::exploration_constant`].
    pub duct_exploration: f64,               // default 0.75

    /// The uniform-exploration probability a [`SimultaneousPolicy::RegretMatching`]
    /// node mixes in on its first visit, decaying from there as `gamma_0 * t^(-1/4)`
    /// in that node's own visit count and clamped into `[0.005, 0.5]`.
    pub regret_matching_exploration: f64,    // default 0.4

    /// Whether a simultaneous root returns a sampled move or the most-selected one.
    pub root_policy: RootPolicy,         // default RootPolicy::Sampled
}
```

`Config::max_reward` / `min_reward` gain one paragraph, because they become
load-bearing for correctness rather than only for the early-stop proof:

```
    /// Bounds of the reward scale.
    ///
    /// Used by the early-termination proof, and by regret matching, which *clamps*
    /// payoffs into `[0, 1]` before touching a regret. CORRECTED: the range does not
    /// rescale the floor or the importance weights — both are dimensionless, and
    /// regret matching's strategy is invariant under any positive rescaling of every
    /// regret. It reaches the search through the clamp and nowhere else, which is
    /// worse rather than better: a game paying in `[-1, 1]` left at the default
    /// `[0, 1]` hands regret matching the same number for a loss and for a draw, at
    /// every budget. Debug builds assert that observed rewards fall inside the
    /// declared range at simultaneous nodes.
```

`SearchResult` and `StopReason` are **unchanged in shape**. Two doc comments change:

```rust
pub struct SearchResult<C> {
    /// The chosen root move.
    ///
    /// At a sequential root, the most visited root choice. At a simultaneous root,
    /// the *perspective* player's own action — drawn from their mixed strategy under
    /// [`RootPolicy::Sampled`] — never a joint tuple and never read off a joint
    /// successor.
    pub choice: C,
    // ...
    /// Visits to the chosen child, or to the chosen marginal at a simultaneous root.
    pub best_visits: u32,
    /// Mean reward of the chosen child, for the player who moved there. At a
    /// simultaneous root, of the chosen marginal — whose player is unambiguously the
    /// perspective player.
    pub best_mean_reward: f64,
}
```

```rust
impl<G: Game> Searcher<G> {
    // new, take_tree, clear_tree, tree, search: signatures unchanged

    /// Re-root the retained tree at `choice`. Unchanged.
    ///
    /// Returns false, and drops the tree, at a simultaneous root: the successor there
    /// is determined by every participant's action, not by one player's. Use
    /// [`Searcher::reuse_joint`].
    pub fn reuse_subtree(&mut self, choice: &G::Choice) -> bool;

    /// Re-root the retained tree at the joint action a simultaneous round actually
    /// resolved to.
    ///
    /// `played` must name every participant exactly once; order does not matter, and
    /// the entries are matched to slots by player index.
    ///
    /// **Expect this to miss, and often.** A simultaneous node has up to
    /// `prod(|A_i|)` successors and materializes at most one per iteration, so unless
    /// the search concentrated its visits on the tuple that was actually played, the
    /// subtree simply is not there. A miss drops the tree and returns false, which
    /// costs a fresh search rather than a wrong one — the same price a
    /// [`Searcher::reuse_subtree`] miss has always cost.
    ///
    /// The promoted node keeps its own marginal statistics, availability included. A
    /// node's `availability` counts iterations at the parent that was just discarded,
    /// which is why re-rooting resets it; an arm's counts iterations at *this* node,
    /// which are still valid and are still the denominator of its exploration term.
    /// Zeroing them would leave a carried-forward arm with hundreds of visits and no
    /// exploration bonus, frozen at whatever its mean happened to be.
    pub fn reuse_joint(&mut self, played: &[(u8, G::Choice)]) -> bool;

    /// Write `player`'s mixed strategy at a simultaneous root into `out`, as
    /// `(choice, probability)` pairs summing to 1 over the actions legal in `state`.
    ///
    /// Returns false if there is no retained tree, if the root is not simultaneous,
    /// or if `player` does not act there. `out` is cleared first and is a caller
    /// buffer, so repeated calls do not allocate.
    ///
    /// This is [`Marginals::policy_into`] renormalized against the real position
    /// rather than against the union of every determinization, which is the only
    /// sound extraction when a player's legal set varies between worlds.
    pub fn root_policy_into(
        &self,
        state: &G,
        ctx: &G::Context,
        player: u8,
        out: &mut Vec<(G::Choice, f64)>,
    ) -> bool;
}
```

`Searcher::search`'s doc gains:

```
    /// At a simultaneous root, `result.choice` is the perspective player's own
    /// action, drawn from their mixed strategy under [`RootPolicy::Sampled`], and
    /// this call panics if `perspective` is not one of the acting players.
    /// [`Config::early_termination`] has no effect at a simultaneous root under
    /// [`SimultaneousPolicy::RegretMatching`] and [`StopReason::Proven`] cannot occur
    /// there: the proof replays UCB1, which is not the rule being run, and for a
    /// sampled answer proving the leader cannot be overtaken proves nothing about
    /// what comes back.
```

### 2.4 `src/parallel.rs`

```rust
impl<G, R> RootParallel<G, R> where /* unchanged bounds */ {
    // new, threads, trees, clear_trees, reuse_subtree, search: signatures unchanged

    /// Re-root every worker's tree at the joint action actually played.
    /// See [`Searcher::reuse_joint`], including its warning that misses are the
    /// common case.
    pub fn reuse_joint(&mut self, played: &[(u8, G::Choice)]);
}
```

### 2.5 `src/lib.rs`

```rust
pub use game::{Game, JointChoices, PlayerSet, Rewards, SimultaneousPolicy, Status};
pub use node::{Marginals, Node, NodeKind};
pub use search::{Config, RootPolicy, SearchResult, Searcher, StopReason};
```

Add to the crate docs, under "What it does":

```
//! * **Simultaneous moves** via decoupled UCT, with regret matching as the default
//!   selection rule so that a game with a mixed equilibrium is played mixed rather
//!   than exploitably. See [`SimultaneousPolicy`] for exactly what converges to what,
//!   and what determinization does to that claim.
```

---

## 3. Internal data structures

All sizes below are **measured** on `aarch64-apple-darwin` with `hashbrown 0.17`,
not estimated. Reproduce with a `size_of` harness before trusting them elsewhere.

### 3.1 `Node<C>` — unchanged in size at every `Choice` type

```rust
pub struct Node<C> {
    pub(crate) cumulative_reward: f64,
    /// Cached `ln(availability)`, refreshed only when `availability` changes.
    pub(crate) ln_availability: f64,
    pub(crate) children: Vec<Node<C>>,
    /// Was `index: Option<Box<ChildIndex>>`. Same slot, same eight bytes: the hash
    /// index and the simultaneous block share one lazily-allocated side table, so a
    /// node that has neither pays nothing and a node that has one pays exactly what
    /// it paid before.
    pub(crate) extra: Option<Box<Extra<C>>>,
    pub(crate) visits: u32,
    /// Iterations in which this choice was legal at its parent.
    pub(crate) availability: u32,
    pub(crate) heuristic_bias: f32,
    /// The player whose reward `cumulative_reward` accumulates.
    pub(crate) player: u8,
    pub(crate) kind: NodeKind,
    /// `Some` only when `kind == NodeKind::Choice`.
    pub(crate) choice: Option<C>,
}
```

| `C` | today | after |
|---|---:|---:|
| `u8` | 64 | **64** |
| `u16` | 72 | **72** |
| `u32` | 72 | **72** |
| `usize` | 80 | **80** |
| `Vec<u16>` | 88 | **88** |

The `kind` byte lands in tail padding at every one of those sizes.

### 3.2 `Extra<C>` — the fused side table

```rust
/// A node's lazily-built side table. A node is either sequential or simultaneous,
/// never both, so the two possibilities share one allocation and one pointer.
pub(crate) enum Extra<C> {
    /// Hash index over `children`, keyed by `Choice`, built once a sequential node
    /// grows past `Game::CHILD_INDEX_THRESHOLD` children.
    Index(ChildIndex),
    /// Present exactly at simultaneous nodes.
    Simul(Box<Simul<C>>),
}

pub(crate) struct ChildIndex { table: HashTable<u32> }   // 32 B
```

`size_of::<Extra<C>>() == 32` for every `C` measured (`u8`, `usize`, `Vec<u16>`):
the discriminant niche-packs into `HashTable`'s non-null pointer. **A wide sequential
node's boxed side table is therefore byte-identical to today's bare `ChildIndex`
box**, and the only cost on the indexed `find_child` path is one enum-tag compare on
a word the load already brought into cache.

### 3.3 `Simul<C>` — the simultaneous block, 208 B plus its arrays

```rust
pub(crate) struct Simul<C> {
    /// Structure-of-arrays, deliberately. Selection reads only `arm_stats` and never
    /// touches a `Choice`; expansion matches `Choice`s and never touches the stats.
    /// At 32 bytes an arm's statistics are exactly half a cache line, so a pass over
    /// a 16-arm bandit touches 8 lines — an array-of-structs layout carrying the
    /// `Choice` inline would touch 16, and would stride unpredictably for a `Choice`
    /// that owns heap data.
    ///
    /// Every participant's arms, concatenated in slot (ascending player) order.
    /// Append-only, which is what makes an arm index stable for the life of the node
    /// — the same argument that makes `path`'s child indices safe.
    pub(crate) arm_stats: Vec<ArmStats>,      // 24
    pub(crate) arm_choices: Vec<C>,           // 24
    /// Parallel to `arm_stats`. **Empty** under `SimultaneousPolicy::Duct`, which
    /// needs no policy state, so `Duct` pays 32 bytes an arm rather than 48.
    pub(crate) arm_policy: Vec<ArmPolicy>,    // 24
    /// Keys of the materialized joint successors, index-parallel to `children`.
    ///
    /// Held here rather than on each child so a linear scan walks a dense `u64`
    /// array — 8 candidates per cache line at any arity — instead of striding 64 to
    /// 88 bytes into `children` to reach a field.
    pub(crate) joint_keys: Vec<JointKey>,     // 24
    /// One index over *all* arms, hashed on `(slot, choice)`, built once the total
    /// arm count passes `Game::CHILD_INDEX_THRESHOLD`. One index rather than one per
    /// participant, because arms live in one flat array and a slot-tagged hash
    /// distinguishes them for free.
    pub(crate) arm_index: Option<ChildIndex>, // 32
    pub(crate) joint_index: Option<ChildIndex>, // 32
    pub(crate) players: PlayerSet,            // 8
    /// `starts[s] .. starts[s + 1]` is slot `s`'s arm range. `starts[arity]` is the
    /// total arm count. A fixed array rather than a `Vec` because arity is capped at
    /// `PlayerSet::MAX_SIMULTANEOUS`.
    pub(crate) starts: [u32; 9],              // 36
    pub(crate) arity: u8,
    /// Bits per participant in a `JointKey`: `64 / arity`.
    pub(crate) key_bits: u8,
}
```

`size_of::<Simul<C>>() == 208` for every `C` measured. It is boxed inside `Extra`, so
a wide *sequential* node never carries it.

### 3.4 `ArmStats` — 32 B, half a cache line

```rust
#[derive(Clone, Copy, Default)]
pub(crate) struct ArmStats {
    /// Cumulative reward for this arm's own player, on the game's raw scale, so
    /// `Marginals::mean_reward` and `Duct` read a real reward.
    pub(crate) cumulative_reward: f64,   // 8
    /// Cached `ln(availability)`, the arm's ISMCTS exploration denominator.
    pub(crate) ln_availability: f64,     // 8
    pub(crate) visits: u32,              // 4
    /// Iterations at this node in which this action was legal for this player.
    pub(crate) availability: u32,        // 4
    /// The node's visit count at the last iteration in which this arm was legal.
    ///
    /// `avail_epoch == node.visits` *is* the legality test, so arms need no parallel
    /// availability buffer at all: the test rides on a word in the same 32-byte line
    /// as the statistics being read, instead of a second memory stream. Initialized
    /// to `u32::MAX`, which no reachable visit count equals.
    ///
    /// The test is only sound while `node.visits` still holds the value stamped
    /// during the descent, so backup **must credit a simultaneous node's arms before
    /// calling `Node::record` on the node itself**. See `back_propagate`.
    pub(crate) avail_epoch: u32,         // 4
}                                        // = 32 (28 + 4 padding)
```

### 3.5 `ArmPolicy` — 16 B, allocated only under `RegretMatching`

```rust
#[derive(Clone, Copy, Default)]
pub(crate) struct ArmPolicy {
    /// Cumulative regret for this arm, on the normalized `[0, 1]` reward scale.
    pub(crate) regret: f64,
    /// Sum, over iterations in which this arm was legal, of the probability the
    /// policy assigned it *before* the exploration floor was mixed in.
    ///
    /// Accumulated at selection time rather than reconstructed from visit counts
    /// afterwards, which makes exploration removal exact instead of an approximate
    /// subtract-and-renormalize. The published recipe subtracts `gamma * T / |A|`,
    /// which is only correct for a constant exploration rate over a constant legal
    /// set — here the rate decays and determinization varies the legal set, so the
    /// subtraction would be wrong in exactly the setting this crate runs in.
    pub(crate) strategy_sum: f64,
}
```

### 3.6 `JointKey` — 8 B per joint edge, zero `Choice` clones

```rust
/// A joint successor edge, as the tuple of marginal indices that produced it, packed
/// at a fixed `64 / arity` bits per participant.
///
/// Fixed rather than derived from the live arm counts: arm lists grow as
/// determinizations reveal new actions, and a radix that grows with them makes every
/// previously-stored key silently alias a different tuple. A fixed radix is stable
/// for the same reason `path`'s child indices are — arms are only ever appended.
///
/// Eight bytes per joint edge and one FxHasher multiply to hash. Making the joint
/// action a `Choice` instead would clone one `Choice` per participant per edge — for
/// a choice that owns heap data, that is `arity` allocations and an order of
/// magnitude more bytes — and would force every game with both node kinds into an
/// `enum Choice`, inflating it at every *sequential* edge in the tree as well.
///
/// Arms per participant are therefore capped by arity, and the cap is asserted where
/// an arm is created rather than where a key is packed, so the failure names the
/// player and the action count:
///
/// | participants | bits each | max arms per player |
/// |---:|---:|---:|
/// | 1 | 64 | 4 294 967 295 |
/// | 2 | 32 | 4 294 967 295 |
/// | 3 | 21 | 2 097 152 |
/// | 4 | 16 | 65 536 |
/// | 5 | 12 | 4 096 |
/// | 6 | 10 | 1 024 |
/// | 7 | 9 | 512 |
/// | 8 | 8 | 256 |
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub(crate) struct JointKey(u64);

impl JointKey {
    /// Arms per participant at this arity. `u32::MAX` where the field is 32 bits or
    /// wider, since an arm index is a `u32` regardless.
    #[inline(always)]
    pub(crate) const fn capacity(key_bits: u8) -> u32 {
        if key_bits >= 32 { u32::MAX } else { (1u32 << key_bits) - 1 }
    }
    /// `picks[s]` is slot `s`'s **slot-relative** arm index, `0 .. slot_len(s)`.
    #[inline(always)]
    pub(crate) fn pack(picks: &[u32], key_bits: u8) -> Self;
    /// The **slot-relative** arm index for `slot`. The global position in
    /// `Simul::arm_stats` is `starts[slot] + arm(slot, key_bits)`; that conversion is
    /// the only place the two numberings meet.
    #[inline(always)]
    pub(crate) fn arm(self, slot: usize, key_bits: u8) -> u32;
}
```

### 3.7 `SimFrame` — 12 B, one per simultaneous level of the current descent

```rust
/// One simultaneous level of the descent that actually selected a joint action.
///
/// Backup needs each participant's sampling probability as an importance weight, and
/// that probability cannot be recovered from the tree: it depends on the whole
/// distribution at the moment of selection and on a decaying floor that has moved by
/// the time backup runs. Recomputing it would rebuild an `O(sum |A_i|)` distribution
/// a second time per level and would silently disagree with the value actually
/// sampled.
///
/// The arm *indices* are not recorded — backup reads them out of the parent's
/// `joint_keys`, which is the whole payoff of keying joint edges by marginal index.
/// `path` therefore stays `Vec<u32>`, one word per level, exactly as it is today.
///
/// Frames are pushed in descent order and consumed in backup order, so backup keeps
/// one cursor and asks `frames[cursor].depth == depth` — a compare against an
/// integer, not a field read on the node. For a purely sequential game `frames` is
/// empty and that is one length compare per level.
#[derive(Clone, Copy)]
struct SimFrame {
    /// Index into `path` of the level this describes.
    depth: u32,
    arity: u32,
    /// Start of this frame's `arity` entries in `sim_probs`.
    prob_start: u32,
}
```

### 3.8 `Scratch<G>` — every reusable buffer, in one place

```rust
/// Every reusable allocation one iteration needs.
///
/// Grouped into a struct because `run_iteration` was already at twelve arguments with
/// an `allow(clippy::too_many_arguments)` and would otherwise reach seventeen. The
/// fields split-borrow, so `&mut s.state` and `&*s.picks` coexist — which is what lets
/// `JointChoices` point into the node while the game mutates the state.
struct Scratch<G: Game> {
    /// The single state every iteration determinizes into.
    state: G,
    /// At a simultaneous node this holds **one participant's** list at a time.
    ///
    /// No two participants' lists are ever live at once — enumerate slot `s`, expand
    /// slot `s`'s arms, clear, next — so no flat-buffer-plus-spans machinery is
    /// needed and the existing single buffer suffices. `JointChoices` borrows the
    /// chosen actions from the node's `arm_choices`, not from here.
    choices: Vec<G::Choice>,
    avail: Vec<bool>,          // sequential nodes only, unchanged
    root_avail: Vec<bool>,     // sequential root fast path only, unchanged
    /// One child index per level, sequential or joint. Unchanged.
    path: Vec<u32>,
    /// One arm index per participant at the current level. At most eight; consumed
    /// before the descent moves on, so one buffer serves every level, and it doubles
    /// as `JointKey::pack`'s input so there is no separate key scratch.
    picks: Vec<u32>,
    sim_frames: Vec<SimFrame>,
    /// `arity` sampling probabilities per frame. `f32` is ample: the floor keeps a
    /// probability well inside f32's precision, and it halves the buffer.
    sim_probs: Vec<f32>,
    /// Which of a simultaneous root's arms are legal in the *real* position. Filled
    /// once per search during root extraction, not per iteration.
    root_legal: Vec<bool>,
    /// Whether the root has had a full expansion pass. At a simultaneous root this
    /// means every participant's arms are fully enumerated, not that any joint
    /// successor exists.
    root_fully_expanded: bool,
}

pub struct Searcher<G: Game> {
    scratch: Scratch<G>,
    side: G::Side,
    root: Option<Node<G::Choice>>,
    tree_is_current: bool,
}
```

Three new hot buffers (`picks`, `sim_frames`, `sim_probs`) and one cold one
(`root_legal`). Nothing else is added, and nothing nested is introduced — a
`Vec<Vec<Choice>>` would lose every inner allocation to an outer `clear()` and start
allocating per node visit with no test that would catch it.

### 3.9 Per-simultaneous-node tree growth, stated out loud

A simultaneous node allocates, once, on first visit: one `Box<Extra<C>>`, one
`Box<Simul<C>>`, and two or three `Vec`s (`arm_stats`, `arm_choices`, and
`arm_policy` only under `RegretMatching`), each `reserve_exact`'d to the enumerated
arm count so they never regrow for arms present on the first visit. Later
determinizations that reveal new arms append and may reallocate. `joint_keys` and
`children` grow by amortized push as joint successors materialize — **a joint
successor costs no allocation of its own**. This is tree growth, which the crate's
allocation claim permits; it is named here rather than hidden behind the letter of it.

---

## 4. Algorithms

Throughout: `k` = arity = `players.len()`; slot `s` runs `0..k` in ascending player
order; `p = players.player_at(s)`; `L_s` = the arms of slot `s` legal in this
determinization; `n_s = |L_s|`; `t = node.visits` **before** this iteration's record;
`span = cfg.max_reward - cfg.min_reward`.

Reward normalization, used by `RegretMatching` and by nothing else:

```
u_hat(p) = if span > 0 { ((rewards.reward(p) - cfg.min_reward) / span).clamp(0.0, 1.0) }
           else        { 0.5 }
```

### 4.1 Node kind resolution (descent, per level)

1. `match state.status(ctx)`:
   - `Terminal(rw)` → `rewards = rw`; break. (unchanged)
   - `Active { player }` → the existing sequential path, unchanged in every detail:
     `reuse_root` fast path, `choices_into`, `Node::expand`, `select`, `path.push`,
     `apply_choice`, `advance`, freshness test, descend.
   - `Simultaneous { players }` → §4.2 onward.
2. Before anything else in the simultaneous arm, run the kind and participant-set
   debug assertions of §6.
3. If `node.extra` is not yet `Extra::Simul`, create it: `arity = players.len()`,
   `key_bits = (64 / arity) as u8`, `starts = [0; 9]`, `players` stored, empty arrays.
   Set `node.kind` stays whatever it was (`Root` / `Choice` / `Joint`) — kind
   describes the *incoming edge*, and simultaneity is a property of the block's
   presence, which is why `NodeKind` has no `Simultaneous` variant.

### 4.2 Marginal expansion

**Full, on every visit, at every simultaneous node, root or not.** Joint successors
stay strictly lazy.

Justification, to be reproduced in the doc comment on `expand_marginals`:
progressive expansion exists because a node with a five-figure *choice list* makes
full expansion quadratic; under DUCT the quantity that explodes is `prod(|A_i|)` — the
joint successors, which stay lazy — and the quantity that is cheap is `sum(|A_i|)`,
which is the entire reason to decouple. Applying progressive expansion to the arms
optimizes the term that was already cheap. It also costs correctness twice: an arm
created at visit `v` has `availability == 1`, so `ln_availability == 0` and no
exploration bonus at all, while its siblings compete on a full history — and every
mixed policy normalizes over a support, so a support that keeps growing makes the
time-averaged strategy an average over incomparable supports, which is not an
approximate equilibrium over anything.

Per visit, for each slot `s` in `0..k`:

1. `scratch.choices.clear(); state.choices_for_into(ctx, p, &mut scratch.choices)`.
2. For each enumerated choice `c`:
   - `find_arm(s, c)`:
     - below the index threshold, linear scan of `arm_choices[starts[s]..starts[s+1]]`;
     - at or above it, `hash_of(&(s as u32, c))` into `arm_index`, with a closure that
       compares both the slot (recovered from the global arm position against
       `starts`) and the choice.
   - **Hit, already stamped this iteration** (`arm_stats[a].avail_epoch == t`):
     a duplicate within the enumeration; do nothing.
   - **Hit, not yet stamped**: `availability += 1`;
     `ln_availability = (availability as f64).ln()`; `avail_epoch = t`.
   - **Miss**: create the arm with `grow_slot(s, c)`, below.
   - After creating: `availability = 1`, `ln_availability = 0.0`, `visits = 0`,
     `cumulative_reward = 0.0`, `avail_epoch = t`, and under `RegretMatching` an
     `ArmPolicy::default()` at the same position.

**`grow_slot(s, c)`, locked.**

```
let at = starts[s + 1] as usize;          // one past slot s's last arm
let new_len = at - starts[s] as usize + 1;
assert!(new_len <= JointKey::capacity(key_bits) as usize, /* message in section 6 */);
arm_stats.insert(at, ArmStats { avail_epoch: t, availability: 1, ..Default::default() });
arm_choices.insert(at, c.clone());
if policy_in_use { arm_policy.insert(at, ArmPolicy::default()); }
for x in &mut starts[s + 1 ..= arity as usize] { *x += 1; }
arm_index = None;                          // payloads are global positions; they moved
```

Three consequences, each of which is why this is cheap enough to do inline:

- **Arm indices inside a `JointKey` are slot-relative** — `0 .. slot_len(s)` — never
  global positions in `arm_stats`. An insert into slot `s` therefore renumbers nothing
  in any other slot, and within slot `s` the new arm takes the next slot-relative
  index, so **no stored `JointKey` ever has to be remapped**. Backup converts a key
  back to a global position with `starts[s] + key.arm(s, key_bits)`; that is the only
  place the two numberings meet.
- **`arm_index` is dropped, not repaired.** Its payload is a global position, and the
  insert moved every later one. It is rebuilt by the next `find_arm` that finds it
  `None` while `starts[arity] > G::CHILD_INDEX_THRESHOLD`. Rebuilding costs one pass
  over the arms.
- **Inserts are rare after the first visit.** The first visit enumerates every slot and
  builds the arrays in one pass, so `grow_slot` only runs when a later determinization
  reveals an action no earlier one offered. The index rebuild is therefore amortized to
  nothing, and `arm_stats` / `arm_choices` / `arm_policy` are `reserve_exact`'d on the
  first visit so the common case never reallocates either.
3. If `slot_len(s) == 0` after enumeration, or no arm of slot `s` is stamped
   (`avail_epoch == t`), the joint action does not exist: set
   `rewards = G::Rewards::uniform(cfg.min_reward)` and break out of the descent, exactly
   as the existing degenerate empty-choice node does. The declared floor rather than a
   zero, because this payoff is credited to every node on the path and is asserted
   against the declared range at every simultaneous one. Emit the debug assertion of §6.

**`ROOT_CHOICES_INVARIANT` fast path at a simultaneous root.** When
`G::ROOT_CHOICES_INVARIANT && node.is_root() && root_fully_expanded`, skip steps 1–2
entirely and instead, for every arm of every slot: `availability += 1`,
`ln_availability = ln(availability)`, `avail_epoch = t`. Move generation is what the
invariant saves; the stamp is `sum(|A_i|)` word writes, the same order as the
sequential path's `root_avail.resize`. Debug verification: §5.4.

### 4.3 Per-player selection

`select_marginal` is a **separate function** from `select`, in a new module
`src/duct.rs`. `select` is unchanged line for line, so sequential codegen is
unchanged and the `tiny/100k` bench is structurally protected.

Common to both policies:
- `select_marginal(s)` returns a **slot-relative** arm index, `0 .. slot_len(s)`; the
  scans below iterate `starts[s] .. starts[s + 1]` and subtract `starts[s]` on the way
  out. That is what lets `grow_slot` insert into one slot without renumbering any
  stored `JointKey`.
- Legality is `arm_stats[a].avail_epoch == t`. There is no availability buffer.
- **There is no root special case.** An arm's `availability` counts iterations at this
  node in which the action was legal for this player, which is the correct
  denominator at the root as well as elsewhere; where the choice set is invariant it
  equals the node's visit count anyway, so the two cannot silently diverge.
- No progressive-bias term. `heuristic_bias` is evaluated at the successor state, so
  it describes a joint child, and selection scores arms.

#### `SimultaneousPolicy::Duct`

One pass over slot `s`'s arms. Maintain a reservoir counter for unvisited arms and a
second for ties.

CORRECTED, §0.5: **two** passes, not one. The tie pool cannot be built against a
running maximum, because a maximum that only ever rises admits arms it later leaves
far behind, and a reservoir cannot evict.

```
score(a) = stats[a].cumulative_reward / stats[a].visits
         + cfg.duct_exploration * (stats[a].ln_availability / stats[a].visits).sqrt()
tol = TIE_TOLERANCE * span                       // TIE_TOLERANCE = 0.01, a private const

// pass 1: the unvisited reservoir, and the maximum score over visited legal arms
unvisited = 0; unvisited_pick = NONE; best_value = -inf
for a in slot_arms(s) where stats[a].avail_epoch == t:
    if stats[a].visits == 0 {
        unvisited += 1
        if below(rng, unvisited) == 0 { unvisited_pick = a }
    } else {
        best_value = max(best_value, score(a))
    }
if unvisited_pick != NONE { return (unvisited_pick, 1.0) }   // pass 2 never runs

// pass 2: a reservoir over the arms within tol of the FINAL maximum
best = NONE; ties = 0
for a in slot_arms(s) where stats[a].avail_epoch == t and stats[a].visits > 0:
    if score(a) >= best_value - tol {
        ties += 1
        if below(rng, ties) == 0 { best = a }
    }
pick = best
prob = 1.0                                        // no importance weighting under Duct
```

Two details that are not optional. **Unvisited arms are chosen uniformly at random**,
not in index order, and **ties are broken at random inside `tol`**, not by first
maximum. Deterministic index-ordered tie-breaking is the correct rule at a sequential
node and the pathological one here: two decoupled bandits breaking ties by index pair
player 0's arm `i` with player 1's arm `i`, so the search opens the diagonal of the
joint matrix first and systematically starves the off-diagonal cells that decide a
mixed equilibrium. Measured exploitability on biased rock-paper-scissors is 0.5–0.8
with deterministic ties and 0.01–0.05 with randomized ones, across the whole
exploration range. The sequential `select` keeps first-strict-maximum: its fixed point
does not depend on the tie rule, and changing it would break
`explores_evenly_when_nothing_distinguishes_choices` for nothing.

#### `SimultaneousPolicy::RegretMatching`

```
// CORRECTED, §0.1 and §0.3.
gamma_t = (cfg.regret_matching_exploration / (max(1, t) as f64).powf(0.25))
              .clamp(0.005, 0.5)
// pass 1: positive-regret mass over the legal arms
sum_r = 0.0; n = 0
for a in slot_arms(s) where legal:
    n += 1
    sum_r += policy[a].regret.max(0.0)
// pass 2: accumulate the exploration-free strategy, and sample inline
target = rng_uniform_01()          // (below(rng, u64::MAX) as f64) / (u64::MAX as f64)
acc = 0.0; pick = last legal arm   // guards against float drift
for a in slot_arms(s) where legal:
    sigma = if sum_r > 0.0 { policy[a].regret.max(0.0) / sum_r } else { 1.0 / n as f64 }
    policy[a].strategy_sum += sigma                       // exploration-free by construction
    mu = (1.0 - gamma_t) * sigma + gamma_t / n as f64
    acc += mu
    if pick_not_yet_set && acc >= target { pick = a; prob = mu }
```

Inverse-CDF in two passes with a scalar accumulator and one draw: **no probability
vector is materialized**, which is why the buffer list in §3.8 has no strategy
buffer. `sigma` is accumulated into `strategy_sum`, never `mu` — that is what makes
exploration removal exact rather than a post-hoc subtraction that would itself have to
be availability-weighted and rate-weighted.

**CORRECTED (§0.1, §0.2).** What this paragraph said: `gamma_t` decays as
`gamma_0 / sqrt(t)` because a *constant* floor cannot converge closer than a
`gamma * depth`-Nash equilibrium, and `1/sqrt(t)` still explores every arm
infinitely often. The premise about the constant floor is right; the schedule
drawn from it is not, and the paragraph two above — that accumulating `sigma`
makes exploration removal exact — contradicts it.

What ships: `gamma_t` decays as `gamma_0 * t^(-1/4)`, clamped into
`[0.005, 0.5]`, and the exponent is the whole point. Backup divides by the
sampling probability, the floor bounds that importance weight at `n / gamma_t`,
and the average-regret bound goes as `T^(a - 1/2) / gamma_0` for a floor
decaying as `t^-a` — so `a = 1/2` is exactly the value at which the bound stops
depending on the budget. Balancing that against a residual exploitability linear
in the floor gives `a = 1/4`, and the measurements in §0.1 agree. The floor is
still what stops regret matching from starving an arm it has driven negative,
and the decay rate matters there too: on a fixture where an arm is legal half
the time and strictly best whenever it is, the specified schedule leaves it with
zero visits in 2 of 40 seeds at 3000 visits (10 of 40 at `gamma_0 = 0.05`) while
the shipped one leaves it with zero in 0 of 40 and lifts its share of the
iterations that offered it from 0.67 to 0.94.

The clamp is not a schedule. `0.4 * t^(-1/4)` only reaches 0.005 after
`(0.4 / 0.005)^4`, about forty million visits to one node; the lower bound exists so that the floor is positive
for every configuration a caller can write, including zero and negative, because
backup divides by a probability it bounds below.

#### After the per-slot loop

```
picks.clear(); prob_start = sim_probs.len()
for s in 0..k:
    picks.push(select_marginal(s)?)        // None => rewards = zero, break descent
    sim_probs.push(prob_s as f32)          // slot-relative index, one probability each
```

The `k` picks are made **independently**; that is the definition of decoupling. The
joint tuple they form need not already exist as a child, and must not be assumed to
(§4.4).

### 4.4 Joint-child lookup and creation

1. `key = JointKey::pack(&picks, key_bits)`.
2. Look up:
   - below `G::CHILD_INDEX_THRESHOLD` joint children, linear scan over
     `joint_keys` — a dense `u64` array, 8 candidates per cache line;
   - at or above it, `hash_of(&key)` into `joint_index`, comparing `joint_keys[i]`.
3. Miss: push a new child with `kind = NodeKind::Joint`, `choice = None`,
   `player = perspective`, `availability = 1`, `ln_availability = 0.0`, and push `key`
   onto `joint_keys`. Build or update `joint_index` exactly as `push_child` does for
   the `Choice`-keyed index.
4. `path.push(i as u32)`.
5. Push `SimFrame { depth: (path.len() - 1) as u32, arity: k as u32, prob_start }`.
6. Build `JointChoices` from `players`, `arm_choices`, `starts`, `picks` and call
   `state.apply_joint(ctx, joint, rng)`. Nothing is cloned.
7. `state.advance(ctx, side, perspective, rng)` — unchanged, and it runs after the
   joint resolution just as it runs after `apply_choice`.
8. **Freshness test on the joint child**, not on any arm:
   `if node.children[i].visits() == 0 { rewards = state.rollout(ctx, rng); break }`.
   The two genuinely disagree in both directions — a never-selected arm can lead to an
   already-visited joint child reached via a different partner action, and a joint
   child can be brand new while every arm in it is well visited — and the joint child
   is the right test because it is the node whose value is being estimated. No
   `heuristic_bias` call here (see §5.5).
9. `node = &mut node.children[i]`.

### 4.5 Descent path recording

`path: Vec<u32>` is **unchanged**: one child index per level, sequential or joint.
The `k` arm indices are recovered at backup from `joint_keys[path[depth]]`, which
costs `k` shift-and-mask operations and no bookkeeping. Only the sampling
probabilities are recorded, in `sim_probs`, indexed through `sim_frames`.

### 4.6 Backup

```
frame_cursor = 0
node = &mut *root
credit_level(node, depth = 0)
for (depth, &i) in path.iter().enumerate():
    node = &mut node.children[i as usize];
    credit_level(node, depth + 1)

fn credit_level(node, depth):
    // 1. Arms first, while node.visits still holds the epoch the descent stamped.
    if frame_cursor < frames.len() && frames[frame_cursor].depth == depth as u32:
        let f = frames[frame_cursor]; frame_cursor += 1
        let key = simul.joint_keys[path[depth] as usize]
        for s in 0..f.arity:
            let p    = simul.players.player_at(s).unwrap()
            let a    = simul.starts[s] + key.arm(s, key_bits)
            let u    = rewards.reward(p)
            let mu   = sim_probs[f.prob_start + s] as f64
            match G::SIMULTANEOUS_POLICY {
                Duct => {}
                RegretMatching => {
                    let u_hat = normalize(u)
                    // Sleeping-bandit correction: only arms that were LEGAL this
                    // iteration accumulate regret. An arm that was illegal could not
                    // have been played, so there is nothing to regret; leaving the
                    // subtraction unrestricted drives a rarely-legal arm's regret
                    // toward negative infinity at rate (1 - f) * E[u] * t, where f is
                    // its legality rate, and it is never selected again.
                    for b in slot_arms(s) where stats[b].avail_epoch == node.visits:
                        policy[b].regret -= u_hat
                    policy[a].regret += u_hat / mu
                }
            }
            stats[a].cumulative_reward += u
            stats[a].visits            += 1
    // 2. Then the node's own record, which bumps visits and invalidates the epoch.
    node.record(rewards.reward(node.player))
```

The ordering in that function is load-bearing and is pinned by the debug assertion in
§6. `Node::record` is unchanged and the loop's inner statement
`node.record(rewards.reward(node.player))` is **literally the line it is today** — no
sentinel player, no branch, no new instruction on either path.

The credit rule is the max^n rule specialized: **each player's own reward goes into
their own arm**, which is exactly what makes the `k` bandits independent.

### 4.7 Root move extraction

In `search()`, after the loop:

- **Sequential root**: unchanged. `most_visited(root)` → clone the child's choice, its
  `visits()`, its `mean_reward()`; on `None`, the existing random fallback.
  *(Superseded by the root extraction correction: the sequential root takes step 2's
  legality mask too — built over children instead of arms, by the one helper both
  extractions and the early-termination proof now call — and ranks the surviving
  children by `§4.8`'s corrected rule, the best mean reward among the children that
  cleared the evidence bar, rather than by raw visits. "Unchanged" was the bug twice
  over: the tree holds the union of every determinization's choices, so the visit
  leader can be a move the player does not hold, and `search` returned it; and a visit
  argmax ranks by legality rate as much as by value. Step 4's "under `Duct` that weight
  is the visit argmax" is superseded there for the same reason.)*
- **Simultaneous root**:
  1. `slot = players.slot_of(perspective).unwrap()` (the preamble already asserted
     membership).
  2. Fill `scratch.root_legal`: enumerate `state.choices_for_into(ctx, perspective, ..)`
     on the **real** state, resize `root_legal` to `slot_len(slot)`, set each arm whose
     choice appears in that list.
  3. Build the mixed strategy over the legal arms (§4.8) into a scalar-accumulator
     pass — no vector is materialized.
  4. `RootPolicy::Sampled` → inverse-CDF draw with `rng`.
     `RootPolicy::MostVisited` → **CORRECTED, §0.4**: argmax by the same weight §4.8
     builds, first maximum, restricted to legal arms — not by `stats[a].visits`.
     Under `Duct` that weight is the visit argmax, so this is unchanged there; under
     regret matching an argmax over raw visits is neither exploration-free nor
     divided by availability, and returns an arm that is strictly dominated wherever
     both it and its rival are legal.
  5. `choice = arm_choices[a].clone()`, `best_visits = stats[a].visits`,
     `best_mean_reward = stats[a].cumulative_reward / stats[a].visits` (0.0 if unvisited).
  6. If no arm is legal, or no iteration completed, fall back to a uniform draw over
     the enumerated legal list — never over `choices_into`, which at a simultaneous
     state may return another player's actions for an asymmetric game.

**Never argmax over joint children.** The perspective player's action `a` appears in
one joint child per opponent action, so an argmax over joint children picks the best
*pair* — the classic optimistic bug that makes a decoupled agent assume the opponent
plays along.

### 4.8 Root policy extraction

For arm `a` of the perspective slot, legal in the real position:

```
RegretMatching:  w[a] = max(0, policy[a].strategy_sum) / max(1, stats[a].availability) as f64
Duct:            w[a] = 1 if a == argmax_legal rank(stats[.]) else 0
```

then normalize over the legal arms; if the total is zero, fall back to uniform over
them.

**Superseded for `Duct` (root extraction correction).** `argmax_legal stats[.].visits`
was the shipped rule and it is the same defect this section spends three paragraphs
describing for regret matching, in its sharpest form: `select_duct` is
availability-corrected and had already identified the dominant arm, and the extraction
handed back the arm that was merely legal more often (measured: arm 0 at 45 180 visits
/ 60 000 availability / mean 0.4 returned over arm 1 at 14 820 / 14 821 / mean 0.9,
and reported `Proven`). `rank` is now `src/rank.rs`, shared by every argmax-family
root in the crate — the sequential root, this one, `Marginals::policy_into` and
`RootParallel`'s merge:

```
rank:  mean reward, ties broken by visits / availability,
       and a candidate with >= MIN_EVIDENCE selections outranks one without
```

**Not** the selection rate `visits / availability`, which was the first correction
attempted here and is wrong for a reason no discount of the rate repairs: *the rate
saturates at 1.0*. Selection takes an unvisited candidate on sight and then keeps
taking it while its exploration bonus covers the gap to the leader —
`(c / gap)^2 * ln(iterations)` selections, a budget that has nothing to do with how
often the candidate was legal — so a candidate whose availability window is shorter
than that budget is taken on essentially every iteration that offered it *whatever it
is worth*, and lands against the ceiling with no room above it. Measured: a decoy
offered one determinization in 100 reaches 97 visits / 97 availability for a rate of
exactly 1.000, well clear of any evidence bar, against the genuinely best move at
9 873 / 9 999 = 0.987 — and a rate-ranked root answered the decoy, worth 0.75 against
0.90, in 16 of 16 seeds and stamped it `Proven`. The mean has no such ceiling: it is
measured on the caller's declared reward scale and it is the quantity the root is
choosing between, so a better candidate can always out-score a worse one on it.

The bar is what keeps this from being the max-child rule robust extraction exists to
avoid — one lucky rollout is a mean of 1.0, and the bar is what stops it winning the
root. It leads under the bar too, because there is nothing better down there: the rate
is manufactured for exactly the candidates that live under it. Where availability is
uniform and the budget is large the mean argmax and the visit argmax coincide on
anything the search has separated, so `§4.7 step 4` and the sequential fingerprint move
only at re-rooted trees, at budgets too small for anything to clear the bar, and where
a wide root never separated its children at all.

The cost lands on `§5.1`, and it is real. A rate is built out of counts, so the
remaining iterations bound it; a mean moves with the rewards, and bounding *those*
needs the rewards reaching a node's accumulator to lie in a known interval.
`Config::min_reward` and `Config::max_reward` are a declaration, not a clamp — `record`
and `credit_marginals` add whatever the game returned — so no such bound is available
and the crate does not pretend otherwise. `src/rank.rs` carries the derivation, the
choice of constant, and what the rule gives up in exchange.

The `/ availability` is not cosmetic. `strategy_sum` accumulates only on iterations
where the arm was legal, so two equally good arms legal in 90% and 20% of
determinizations end up with mass in roughly a 4.5:1 ratio. Dividing by availability
turns each accumulator into a mean probability *conditional on being legal*, which is
the best quantity a per-arm accumulator can offer for comparing arms with different
legality rates. This is sound here precisely because `reuse_joint` keeps arm
availability (§5.3): numerator and denominator always describe the same window of
iterations. Any future change that resets arm availability while keeping
`strategy_sum` would silently corrupt the extracted equilibrium and must reset both
together.

It is a correction and not a cure, and the earlier claim that it makes the arms
"comparable" full stop is too strong. Two arms conditioned on *different* legality
events are still not two entries of one distribution: an arm legal every iteration
absorbs mass on the iterations where its rival is absent, and no per-arm accumulator
can distinguish that from mass it earned. Where arm A is legal always and arm B is
legal half the time and strictly better whenever it is, the exact answer at a
position offering both is all of the mass on B, and this extraction reports 0.33 on A
(measured, 16 seeds at 200k, and unchanged by the floor schedule). Getting it exactly
right needs a statistic per legal *set* rather than per arm. This is left as it is —
see §0's closing paragraph — and is now stated in `root_strategy_into`'s doc comment
rather than implied away.

`Marginals::policy_into` computes the same thing over *all* arms (no legality filter,
since a `Marginals` view has no position to filter against) and says so in its doc.

---

## 5. Every ripple

### 5.1 `early_stop`

- `settled` gains a `perspective: u8` parameter and a policy check. Signature:
  `settled<G: Game>(root: &Node<G::Choice>, perspective: u8, target: u32, cfg: &Config) -> bool`.
- **Sequential root**: unchanged in every detail. *(Superseded by the root
  extraction correction: the sequential branch now takes the same legality mask as
  the simultaneous one, chooses between the two candidate sets on the position rather
  than on the tree, and states its bound against `src/rank.rs`'s ranking — which is a
  mean reward, and nothing in the counts bounds a mean. What is left of this section
  is the evidence bar: a rival that cannot reach `MIN_EVIDENCE` selections with the
  iterations left stays in the tier below the leader's, whatever any reward turns out
  to be. That caps the saving at `MIN_EVIDENCE - 1` iterations no matter how large the
  budget, and it is a no-op at any root whose rivals are all well sampled. The
  alternative was a bound computed from `Config::max_reward`, which the crate does not
  clamp to; a lost optimisation is cheaper than a false `Proven`. Both branches
  additionally refuse to prove while the position offers a legal action the tree has
  no candidate for: a candidate created with `n` iterations left can spend all of them,
  clear the bar, and be ranked on a mean nothing in hand constrains.)*
- **Simultaneous root under `RegretMatching`**: return `false` immediately.
  `StopReason::Proven` is unreachable there.
- **Simultaneous root under `Duct`**: the candidate set becomes the perspective
  player's arms. Invariant I5 survives — one iteration bumps exactly one arm per
  player — so visit counts are still a conserved budget and the cheap
  `gap > remaining` proof carries over verbatim against arm visits.
  `can_challenger_overtake` **does not run**: its replay uses `ln(total)` where the real
  arm selection uses per-arm `ln_availability`, and under determinization those diverge.
  `MAX_SIMULATION_WORK` is therefore never consulted at a simultaneous root; note in
  the doc that when it is consulted at a sequential root it is unchanged.
- `top_two_visits` and `leader_index` are refactored to take a `&[u32]` of candidate
  visit counts, so one implementation serves children and arms. `can_challenger_overtake`
  keeps its `&Node` signature — it is sequential-only.
- **The coupling that makes the `Duct` case sound must be asserted**, because it is
  otherwise an unstated assumption: `settled` proves "the argmax by visits cannot
  change", and that is only the returned answer because under `Duct` both root
  policies reduce to argmax-by-visits (`policy_into` is one-hot). See §6.

### 5.2 `RootParallel::merge`

Rewritten. `merge` becomes `fn merge(&mut self, perspective: u8, cfg: &Config, results: ...)`
— private, so no public signature changes.

- **Sequential root**: as today, keyed by `Choice`, first-seen order, summing `visits`
  and `cumulative_reward`. One change: the `let Some(choice) = child.choice() else { continue }`
  is replaced by a `debug_assert!(child.kind() == NodeKind::Choice, ...)` plus the
  unwrap, so a joint child reaching that path is a loud failure rather than a silent
  skip (§6). This closes the trap that `Node::choice()` gaining a second `None` case
  would otherwise open.
- **Simultaneous root**: merge the perspective player's **arms, keyed by `Choice`**,
  and never touch a joint child. Joint keys are tuples of *slot-relative marginal
  indices*, and worker 1's arm 2 is not worker 2's arm 2 — each appended arms in its
  own determinizations' order — so merging joint children across workers would combine
  statistics for different actions with nothing to catch it. Arms are keyed by
  `Choice`, which means the same thing in every tree, and merging one player's
  per-action statistics is what root parallelism is for anyway.
  - Sum `visits` and `cumulative_reward` per `Choice`.
  - Under `RegretMatching`, **normalize each worker's mixed strategy first**, then
    average the normalized vectors weighted by that worker's `root.visits`. Summing raw
    `strategy_sum` across workers would over-weight whichever tree ran longest, and
    deadline and cancellation make worker iteration counts unequal in practice.
  - Apply `cfg.root_policy` to the merged vector. `Sampled` needs an rng: borrow
    worker 0's, after the thread scope has closed, so the merge stays deterministic for
    a fixed thread count. That is why `merge` takes `&mut self`.
- `merged_stop_reason` is unchanged.
- `RootParallel::reuse_joint` mirrors `reuse_subtree` — a `for` loop over workers.

### 5.3 `reuse_subtree` / `reroot`

- `Node::reroot_at(&C)` — unchanged behaviour, plus it now sets
  `child.kind = NodeKind::Root` alongside clearing `choice`, and it refuses to match a
  joint child for free (a joint child's `choice` is `None`, which `find_child`'s
  `Option<C>` comparison can never equal).
- `Searcher::reuse_subtree` — unchanged signature. At a simultaneous root
  (`root.simultaneous_players().is_some()`) it returns `false`, drops the tree, and
  debug-asserts with the message in §6.
- `Node::reroot_at_joint(&mut self, key: JointKey) -> bool` — new. Find the joint child
  by key; `swap_remove` it; set `kind = Root`, `choice = None`, `availability = 1`,
  `ln_availability = 0.0`; `*self = child`. **Do not touch the promoted node's arms.**
- `Searcher::reuse_joint(&[(u8, G::Choice)])` — new. Validates that the slice names
  every participant exactly once (debug assertion, §6), resolves each choice to its
  slot-relative arm index by scanning that slot's `arm_choices`, packs the key, calls
  `reroot_at_joint`, and on any miss drops the tree and returns false.
- **The root's `player` and `cumulative_reward`.** `search()` stamps
  `root.player = root_player` and `root.cumulative_reward = 0.0` immediately after the
  tree-staleness check, where `root_player` is `Status::Active { player }` at a
  sequential root and `perspective` at a simultaneous one. This is the fix for a
  pre-existing latent bug: today `reroot_at` never re-stamps `player`, so a re-rooted
  root keeps the *previous mover's* index while a fresh root gets the player to act,
  and `root.cumulative_reward` therefore means a different player's payoff depending on
  whether the tree was reused. Nothing in the search reads it, which is why it has gone
  unnoticed; the DUCT change forces the question and this resolves both cases at once.
  The fresh-root value for a sequential game is unchanged, so no sequential behaviour
  moves.

### 5.4 `ROOT_CHOICES_INVARIANT`

- Meaning at a simultaneous root: **for every participant, that player's own marginal
  set does not vary across determinizations.** It says nothing about joint successors.
- `root_fully_expanded` means "every participant's arms are fully enumerated" at a
  simultaneous root, and "a full expansion pass has created a child for every legal
  choice" at a sequential one, as today.
- Debug verification at a simultaneous root replaces the `choices.len() == children.len()`
  assertion entirely, which would otherwise fire on the second iteration of any
  simultaneous root with the flag set (joint-child count is a function of iterations
  run, not of any enumeration). It becomes `k` independent checks: for each slot,
  enumerate `choices_for_into`, assert the count equals `slot_len(s)`, and assert every
  enumerated choice resolves to an arm. Messages in §6.

### 5.5 Progressive bias and `heuristic_bias`

`heuristic_bias` is **never called at a simultaneous node**, and
`Config::progressive_bias_weight` has no effect there. The prior is evaluated at the
successor state, so it can only describe a joint child, and selection scores arms.
This is a silent behavioural hole for a game that tunes progressive bias and then adds
a simultaneous node, so it is documented on both `Game::heuristic_bias` and
`Config::progressive_bias_weight`. `ucb_raw`'s bias term is therefore not used by
`select_marginal`, which is one reason `select_marginal` does not call it.

### 5.6 `Side` and `advance`

Unchanged. `advance` is called after `apply_joint` exactly as it is called after
`apply_choice`. Add one paragraph to `Game::advance`'s doc: a game whose `advance`
resolves opponent moves via a side model is doing something that overlaps
semantically with simultaneous play, and it should pick one — modelling the opponents'
concurrent choice inside `advance` and *also* reporting `Simultaneous` would credit the
same decision twice.

### 5.7 The `SingleChoice` fast path and the search preamble

```rust
let root_player = match state.status(ctx) {
    Status::Active { player } => player,
    Status::Simultaneous { players } => {
        assert!(!players.is_empty(), "mcts: Status::Simultaneous with no players");
        assert!(
            players.len() <= PlayerSet::MAX_SIMULTANEOUS,
            "mcts: Status::Simultaneous names {} players; at most {} may act at once",
            players.len(), PlayerSet::MAX_SIMULTANEOUS
        );
        assert!(
            players.contains(perspective),
            "mcts: search called on behalf of player {perspective}, who is not one of \
             the players acting at this simultaneous state"
        );
        perspective
    }
    Status::Terminal(_) => panic!("mcts: search called on a terminal state"),
};
```

The single-choice fast path and the zero-iteration random fallback **both** enumerate
through `choices_for_into(ctx, perspective, ..)` when the root is simultaneous, and
through `choices_into` when it is sequential. Returning `SingleChoice` when the
perspective player has exactly one action is sound at a simultaneous root: with one
action, nothing the opponents do changes which action you take. The zero-choices panic
names the player:

```
"mcts: search called on a state where player {perspective} has no legal choices"
```

---

## 6. Debug assertions

Every one of these is `debug_assert!` / `debug_assert_eq!`, following the
`ROOT_CHOICES_INVARIANT` precedent: document the invariant, name the wrong-move
failure mode, verify where verification is affordable, and tell the implementor to run
integration tests in debug before trusting release.

**Node kind stability** — in `run_iteration`, in the `Simultaneous` arm, where
`status()` is already in a register:

```
debug_assert!(
    node.simultaneous_players().is_none() || node.simultaneous_players() == Some(players),
    "mcts: this node reported Status::Simultaneous with players {:?} but its \
     statistics were built for {:?}. A node's kind and participant set must not vary \
     across determinizations — determinization may change which actions are legal, \
     never who acts. The search will read one player's arms as another's and return \
     a move belonging to the wrong player.",
    players, node.simultaneous_players().unwrap()
);
```

and in the `Active` arm:

```
debug_assert!(
    node.simultaneous_players().is_none(),
    "mcts: this node reported Status::Active {{ player: {player} }} but its statistics \
     were built as a simultaneous node. A node's kind must not vary across \
     determinizations; the search will read marginal bandits as children."
);
```

**Backup ordering** — inside `credit_level`, before the arm loop:

```
debug_assert_eq!(
    node.visits, epoch_stamped_during_descent,
    "mcts: a simultaneous node's arms must be credited before Node::record bumps its \
     visit count — the legality test is `avail_epoch == node.visits`, and after the \
     record every arm reads as illegal, which silently disables the regret \
     correction that keeps a rarely-legal action playable."
);
```

(Implement by recording the epoch in `SimFrame` as a fourth `u32` field only under
`cfg!(debug_assertions)`; in release `SimFrame` stays 12 bytes.)

**Arm-array parallelism** — after every expansion pass, the analogue of
`select.rs:49`:

```
debug_assert_eq!(simul.arm_stats.len(), simul.arm_choices.len(),
    "mcts: arm statistics and arm choices are out of step");
debug_assert!(simul.arm_policy.is_empty()
    || simul.arm_policy.len() == simul.arm_stats.len(),
    "mcts: arm policy state is out of step with arm statistics");
debug_assert_eq!(simul.starts[simul.arity as usize] as usize, simul.arm_stats.len(),
    "mcts: slot boundaries do not cover the arm array");
debug_assert_eq!(simul.joint_keys.len(), node.children.len(),
    "mcts: joint keys and joint children are out of step");
```

**Arm capacity** — at arm creation, before the push, so the message names the player
rather than surfacing later as a corrupted key:

```
assert!(
    new_len <= JointKey::capacity(key_bits) as usize,
    "mcts: player {p} has {new_len} legal actions at a simultaneous node with \
     {arity} participants, where at most {} fit in a joint key. Narrow the action \
     set, or split the decision — with {arity} players the joint successor space is \
     already unreachable.",
    JointKey::capacity(key_bits)
);
```

This one is a hard `assert!`, not a debug assertion: exceeding it silently aliases
joint successors, which is a wrong move in release.

**Reward range** — once per credited arm at a simultaneous node, under *both*
policies. It sat inside the `RegretMatching` backup on the reading that the range
reaches the search through that policy's clamp and nowhere else; it does not,
because `Duct` measures its tie tolerance against the width of the range, so an
over-declared range is the one input that turns decoupled UCB1 into a uniform
random move picker and it was the policy nothing checked. Against
`cfg.max_reward` itself rather than a `cfg.min_reward + span` reconstruction of
it, which is not the same number in f64 — for `[-10.0, -3.9]` the sum is
`-3.9000000000000004` — and would accuse a game paying its own declared maximum:

```
debug_assert!(
    u >= cfg.min_reward - 1e-9 && u <= cfg.max_reward + 1e-9,
    "mcts: observed reward {u} for player {p} falls outside \
     Config::min_reward..=Config::max_reward ({}..={}). Regret matching normalizes \
     payoffs to [0, 1] before touching a regret, so a wrong range rescales the \
     exploration floor and the importance weights together.",
    cfg.min_reward, cfg.max_reward
);
```

**Config refusal** — `Config::validate`, in the preamble of `Searcher::search` and
on `RootParallel::search`'s own thread before any worker is spawned:

```
assert!(
    cfg.iterations != 0 || cfg.time_limit_ms.is_some(),
    "mcts: Config has neither an iteration nor a time budget"
);
assert!(
    cfg.max_reward > cfg.min_reward,
    "mcts: Config declares the reward range [{}, {}], which is empty. Set both bounds \
     to your game's actual payoff range: regret matching cannot tell one payoff from \
     another inside an empty range, and decoupled UCB1 measures its tie tolerance \
     against the width of it.",
    cfg.min_reward, cfg.max_reward
);
```

Hard `assert!`s, not debug assertions: an empty or inverted range is a
configuration no search can honour under either policy, and it is silent in
release. Both callers raise the same refusal — one `Config::refusal()` returning
the message, so the pool can act before it panics. `normalize_reward` divides by the span, so regret matching is handed a
constant `0.5` for every payoff — a driftless, high-variance random walk over the
regrets, measured at roughly twice the exploitability of playing uniformly at
random — and `Duct`'s tie tolerance is a fraction of the same span, so a zero one
leaves the tie pool decided by nothing. Both bounds are checked, because an
inverted range takes the same branch as an empty one.

`RootParallel::search` raises it on its own thread rather than as N workers
panicking at once, and **disarms every worker's retained tree first**.
`Searcher::search` consumes `tree_is_current` as its first statement, so every
exit from it — the forced-move return, a preamble panic — leaves the tree stale;
a config refused before the workers are spawned means no worker's `search` runs
to do that, and the refusal has to stand in for it or the next search reuses
trees built for the previous position.

**Degenerate simultaneous node**:

```
debug_assert!(
    false,
    "mcts: player {p} has no legal action at a simultaneous node, so no joint action \
     exists. The iteration scores the declared minimum reward, which will drag this \
     node's parent's mean down indistinguishably from a bad evaluation."
);
```

**Root reuse mismatch** — in `Searcher::reuse_subtree`:

```
debug_assert!(
    root.simultaneous_players().is_none(),
    "mcts: reuse_subtree was called on a simultaneous root, whose successor is \
     determined by every participant's action rather than by one player's. Use \
     Searcher::reuse_joint. The tree has been dropped."
);
```

**`reuse_joint` argument validation**:

```
assert_eq!(played.len(), players.len(),
    "mcts: reuse_joint was given {} actions for {} participants",
    played.len(), players.len());
for (p, _) in played {
    assert!(players.contains(*p),
        "mcts: reuse_joint names player {p}, who does not act at this node");
}
// plus a duplicate check via a u64 seen-mask
```

**`ROOT_CHOICES_INVARIANT` at a simultaneous root**:

```
assert_eq!(enumerated.len(), slot_len(s),
    "ROOT_CHOICES_INVARIANT is set, but this determinization offers player {p} {} \
     choices where the tree holds {} marginals",
    enumerated.len(), slot_len(s));
assert!(find_arm(s, choice).is_some(),
    "ROOT_CHOICES_INVARIANT is set, but this determinization offers player {p} a \
     choice the tree has no marginal for");
```

**`build_index` / `push_child` on a simultaneous node** — replacing the two bare
`.unwrap()`s at `node.rs:151,153` and the one at `:142`:

```
debug_assert!(
    self.simultaneous_players().is_none(),
    "mcts: the Choice-keyed child index is only ever built on a sequential node; a \
     joint successor has no single choice to key on"
);
// then `.expect("a sequential node's children all carry a choice")`
```

**`RootParallel::merge` joint-child trap**:

```
debug_assert_eq!(
    child.kind(), NodeKind::Choice,
    "mcts: the sequential root merge reached a joint successor. Joint edges are keyed \
     by per-tree marginal indices and must never be merged across workers; a \
     simultaneous root merges the perspective player's marginals instead."
);
```

**`settled` policy coupling** — at the top of the simultaneous branch:

```
debug_assert!(
    !G::SIMULTANEOUS_POLICY.mixes(),
    "mcts: the early-termination proof shows that the most-visited candidate cannot \
     be overtaken, which is only the returned answer while the root policy is an \
     argmax. Under a mixing policy the answer is a draw from a distribution and the \
     proof says nothing about it."
);
```

---

## 7. Test plan

New fixture module `tests/common/simul.rs`, re-exported from `tests/common/mod.rs`.
Every fixture is perfect-information (`determinize_into` is `clone_from`) unless it is
explicitly testing determinization.

### 7.1 Fixtures to add

- **`Rps`** — one simultaneous ply, `PlayerSet::first_n(2)`, `Choice = u8` in `0..3`,
  standard rock-paper-scissors payoffs on `[0, 1]` (win 1.0, draw 0.5, loss 0.0).
- **`BiasedRps`** — the Shafiei/Sturtevant/Schaeffer matrix, rescaled to `[0, 1]`:

  | P0 \ P1 | rock | paper | scissors |
  |---|---|---|---|
  | rock | 0.50 | 0.25 | 1.00 |
  | paper | 0.75 | 0.50 | 0.45 |
  | scissors | 0.00 | 0.55 | 0.50 |

  Zero-sum (`p1 = 1 - p0`). Unique Nash equilibrium `(1/16, 10/16, 5/16)` for both
  players.
- **`Pennies`** — matching pennies as one simultaneous ply. Equilibrium `(1/2, 1/2)`,
  value 0.5.
- **`SequentialPennies`** — the *same* payoff matrix modelled as two sequential plies:
  player 0 commits, then player 1 chooses knowing it. Used only to demonstrate that
  serialization gets the value wrong.
- **`ThreeWayParity`** — three players each pick 0 or 1 simultaneously
  (`PlayerSet::first_n(3)`); each player scores 1.0 if their bit differs from the XOR
  of the other two, else 0.0. Symmetric, unique equilibrium uniform for every player.
- **`SeqThenSimul`** — player 0 picks one of two branches sequentially; branch `A`
  leads to a simultaneous `Pennies` node, branch `B` leads to a terminal worth 0.25 to
  player 0. The equilibrium value of `A` is 0.5, so a correct search prefers `A`.
- **`AsymmetricBid`** — two players, player 0 has three actions and player 1 has two;
  `choices_for_into` is overridden. Exercises the non-default enumerator and asymmetric
  slot lengths.
- **`VariableRps`** — `BiasedRps` where determinization removes `scissors` from
  player 1's legal set with probability 1/2. Exercises the availability correction.
- **`ManyArm`** — a two-player simultaneous node where each player has 40 actions,
  above `CHILD_INDEX_THRESHOLD`, to exercise `arm_index` and `joint_index`.

### 7.2 Correctness of the simultaneous machinery

| test | proves |
|---|---|
| `rps_does_not_converge_to_a_pure_strategy` | After 50 000 iterations of `Rps` under `RegretMatching`, `root_policy_into` gives every action probability in `[0.2, 0.5]` — i.e. no action is played nearly always and none is abandoned. Asserted for both players by running two searches. |
| `duct_does_converge_to_a_pure_strategy_on_rps` | The same run under `Duct` produces a `policy_into` that is one-hot, and its most-visited arm takes more than 45% of arm visits. The contrast is the point: it documents *why* `Duct` is not the default, and it fails loudly if `Duct` ever silently starts mixing. |
| `biased_rps_learns_the_known_equilibrium` | 200 000 iterations of `BiasedRps` under `RegretMatching`; the extracted `mu_bar` is within 0.06 (L-infinity) of `(0.0625, 0.625, 0.3125)` for the perspective player. This is the load-bearing correctness test for the whole feature. |
| `biased_rps_duct_misses_the_equilibrium` | The same budget under `Duct` is **not** within 0.15 of the equilibrium. Documents the proved failure and would catch someone accidentally making `Duct` mix. |
| `matching_pennies_is_worth_half_simultaneously` | `Pennies` under `RegretMatching`: `result.best_mean_reward` for the perspective player is within 0.05 of 0.5, and `root_policy_into` is within 0.06 of `(0.5, 0.5)`. |
| `sequentialized_matching_pennies_gets_the_value_wrong` | The same game as `SequentialPennies`, searched sequentially, gives the first mover a root mean reward below 0.15 — a bound, not an approximation. Run back to back with the previous test in one `#[test]` so the pair is a single, unmissable comparison. **This is the test that justifies the whole feature.** |
| `three_players_each_learn_a_uniform_mixture` | `ThreeWayParity`, 100 000 iterations, three searches (one per perspective): each player's `mu_bar` is within 0.08 of `(0.5, 0.5)`, and the node's arity is 3. |
| `a_mixed_tree_searches_both_kinds_of_node` | `SeqThenSimul`: the root is sequential and picks branch `A`; the tree contains a node with `simultaneous_players() == Some(first_n(2))`; `node_count()` counts only real nodes (arms excluded); a `Choice` is a plain `u8` throughout — no enum wrapper anywhere. |
| `asymmetric_participants_get_their_own_action_lists` | `AsymmetricBid`: `marginals(0).len() == 3`, `marginals(1).len() == 2`, and every arm's choice is one the corresponding player actually owns. |
| `a_rarely_legal_action_is_not_starved` | `VariableRps` under `RegretMatching`: player 1's `scissors` arm ends with `availability` roughly half the node's visits **and** non-zero `visits`, and its extracted probability is within 0.10 of what an always-legal run gives. Without the sleeping-bandit correction this test fails hard — the arm's regret runs to negative infinity and its visits stall. |
| `wide_simultaneous_nodes_track_every_arm_and_joint_child` | `ManyArm`: every player has exactly 40 arms with distinct choices, `arm_index` is in use, joint children are distinct by key, and `Node::joint_arm(child, player)` round-trips through `Marginals::choice`. |
| `a_joint_child_reports_no_single_choice` | For every joint child: `kind() == NodeKind::Joint`, `choice().is_none()`, `reward_player() == perspective`. |
| `a_degenerate_simultaneous_node_scores_zero` | A fixture where one player has no legal action: the search completes, returns a legal move for the perspective player, and does not panic in release. |

### 7.3 Tree reuse

| test | proves |
|---|---|
| `reuse_joint_carries_arm_statistics_forward` | Search `SeqThenSimul` past the simultaneous node, `reuse_joint` at a tuple that was visited, and check the new root's `reused_iterations` equals the promoted child's visits and that every arm's `availability` and `visits` survived. |
| `reuse_joint_misses_are_survivable` | `reuse_joint` at a tuple with no materialized child returns false, leaves `tree().is_none()`, and the next search runs from scratch and returns a legal move. |
| `reuse_subtree_refuses_a_simultaneous_root` | Returns false, drops the tree, and (in debug) panics with the named message under `#[should_panic]`. |
| `a_reused_simultaneous_root_keeps_arm_availability` | After `reuse_joint`, every arm's `availability` is unchanged and the node's own `availability` is 1 — the asymmetry that keeps the exploration term calibrated. |
| `reuse_joint_rejects_a_malformed_tuple` | `#[should_panic]` on a tuple that names a non-participant, and on one that names a participant twice. |

### 7.4 Regression: every existing sequential behaviour is byte-identical

- All twelve tests in `tests/search.rs` **must pass unmodified**. In particular
  `explores_evenly_when_nothing_distinguishes_choices`'s exact 27/27/27 pins
  `select`'s tie-break, float evaluation order and child creation order, and
  `finds_the_only_winning_line` / `defeats_a_shallow_trap` pin the RNG draw sequence.
  Every new RNG draw lives in `select_marginal` and root sampling, neither of which a
  sequential game can reach.
- **New file `tests/sequential_fingerprint.rs`.** Before touching any source, run the
  existing build and capture, for each of `{AlwaysWin, CountToThree, GameTree::minimal_trap,
  GameTree::wide(200), GameTree::wide_two_ply(40)}` × seeds `{1, 7, 21}` × budgets
  `{50, 500, 5000}`: `result.choice`, `iterations_used`, `root_visits`, `best_visits`,
  `best_mean_reward` (to 12 significant figures), `stop_reason`, `tree().node_count()`,
  `tree().max_depth()`, and the sorted per-child `(choice, visits, cumulative_reward)`
  vector. Hardcode all of it into the test as expected values. It must pass byte-for-byte
  after the change. If it does not, the change touched the sequential path and the
  change is wrong, not the test.
- **New test `node_layout_is_unchanged`** in `src/node.rs` (a `#[cfg(test)]` module):
  `assert_eq!(size_of::<Node<u8>>(), 64)` and the same for `u16`→72, `u32`→72,
  `usize`→80, `Vec<u16>`→88, plus `size_of::<Extra<u8>>() == 32` and
  `size_of::<ArmStats>() == 32`. These are the numbers the doc comments cite; a test
  keeps them honest.

### 7.5 Determinism

| test | proves |
|---|---|
| `same_seed_gives_the_same_simultaneous_search` | Two identical runs of `BiasedRps` under `RegretMatching` with the same seed give identical `result.choice` and identical arm statistics. Sampling is seeded, so it reproduces. |
| `most_visited_root_policy_is_deterministic` | `RootPolicy::MostVisited` on `BiasedRps` returns the same choice across ten runs with *different* seeds once the budget is large — the policy consumes no rng draw. |
| `root_parallel_merges_marginals` (feature `parallel`) | Four workers on `BiasedRps`; the merged answer is legal, `root_visits` is the sum, and — under `MostVisited` — matches a single-threaded run of the same total budget within one action. |

### 7.6 Benches

- **New `benches/simultaneous.rs`** with three groups, all measured against
  `benchmarks/BASELINE.md`'s protocol (interleave A/B, discard the first repetition,
  medians, re-run the first configuration at the end, void the session on >3% drift):
  - `rps/{10k,100k}` — the simultaneous core with the game removed, one `Duct` point and
    one `RegretMatching` point, so the cost of the converging default is a number rather
    than an argument.
  - `simul_wide/{8,32,128}` — arms per player at arity 2, measuring the arm scan and the
    joint-child lookup crossover.
  - `simul_arity/{2,3,4}` — fixed 8 arms per player, measuring what arity costs.
- **`benches/search.rs` and `benches/child_storage.rs`** gain one `Status::Simultaneous { .. } => unreachable!()`
  arm each where they match exhaustively (`benches/search.rs:302-305` is the only
  exhaustive match; `benches/child_storage.rs` only constructs). No other change.
- **Regression gate, and it is a blocker not a cost.** `tiny/100k` must stay within 3%
  of 22.4 ms (224 ns/iteration), and `narrow/{1000,10000}` and `wide/{100,400,1600}`
  within 3% of their `BASELINE.md` medians. If `tiny` moves, the third `Status` arm or
  the `Extra` tag compare landed somewhere hot; find it before shipping. Record the new
  simultaneous numbers into `BASELINE.md` in the same PR.
- **Exploitability harness.** `biased_rps_learns_the_known_equilibrium` and
  `biased_rps_duct_misses_the_equilibrium` together are the gate on the default policy.
  Bandit-feedback regret matching has never been measured inside SM-MCTS — every
  published RM number is the full-matrix variant this design rejects — so the default
  is not defensible in this crate's citation-carrying voice until those two tests pass
  and their numbers are recorded in `BASELINE.md`. **They ship in the same PR as the
  feature.** If `RegretMatching` cannot beat `Duct` on exploitability there, one of the
  availability corrections is wrong; fix it rather than changing the default.

---

## 8. File-by-file work plan

Steps within a group are independent; groups are ordered.

**Group 0 — capture, before touching anything**
1. Write `tests/sequential_fingerprint.rs` against the *current* build and commit the
   captured values. Nothing after this point is trustworthy without it.
2. Re-run `benches/search.rs` on the current build and confirm the `BASELINE.md`
   medians still hold on this machine. Record the sha and `rustc -vV`.

**Group 1 — leaves, independently editable**
3. `src/util.rs` — add `uniform_01<R: Rng + ?Sized>(rng) -> f64`. Nothing else changes.
4. `src/game.rs` — `PlayerSet`, `PlayerSetIter`, `JointChoices`, `SimultaneousPolicy`,
   `Status::Simultaneous`, `Game::{SIMULTANEOUS_POLICY, choices_for_into, apply_joint}`,
   the doc paragraphs on `ROOT_CHOICES_INVARIANT`, `heuristic_bias` and `advance`.
5. `src/node.rs` — `NodeKind`, `ArmStats`, `ArmPolicy`, `JointKey`, `Simul<C>`,
   `Extra<C>`; rename `index` → `extra` and route `find_child`/`push_child`/`build_index`
   through it; add `kind` and back `is_root()` with it; add the public accessors and
   `Marginals<'_, C>`; add `expand_marginals`, `find_arm`, `push_arm`, `find_joint`,
   `push_joint`, `reroot_at_joint`; add the `#[cfg(test)]` layout test. **`expand`,
   `record`, `new_root` and `reroot_at` keep their existing bodies**, with `reroot_at`
   gaining one `kind` assignment.

**Group 2 — the new module**
6. `src/duct.rs` (new) — `TIE_TOLERANCE`, `select_marginal`, `credit_marginals`,
   `root_strategy_into`, and the reward normalization helper. Nothing here is reachable
   from a sequential game. Declared `mod duct;` in `lib.rs`.

**Group 3 — the search, one file, one commit**
7. `src/search.rs` — `RootPolicy`; the two `Config` fields and their docs; `Scratch<G>`
   and the `Searcher` regrouping; the preamble; the `choices_for_into`-routed
   `SingleChoice` path and random fallback; the root `player`/`cumulative_reward` stamp;
   the `Simultaneous` arm of `run_iteration`; the rewritten backup with the frame
   cursor; `reuse_joint`; `root_policy_into`; the simultaneous root extraction.
8. `src/early_stop.rs` — `settled` gains `perspective` and the policy gate;
   `top_two_visits` / `leader_index` become slice-based.

**Group 4 — dependents**
9. `src/parallel.rs` — the `merge` rewrite, `reuse_joint`, the `debug_assert` on the
   sequential merge path.
10. `src/lib.rs` — `mod duct;`, the re-exports, the crate-doc bullet.

**Group 5 — consumers, each independently editable**
11. `benches/search.rs` — one `unreachable!()` arm at line 302.
12. `benches/child_storage.rs` — no change expected; verify it compiles.
13. `examples/tic_tac_toe.rs` — one `unreachable!()` arm in the rollout match at line 89.
14. `tests/common/mod.rs` — one `unreachable!()` arm in `GameTree::rollout` at line 236.

**Group 6 — the new tests and benches, independently editable**
15. `tests/common/simul.rs` — the nine fixtures of §7.1.
16. `tests/simultaneous.rs` — §7.2 and §7.5.
17. `tests/simultaneous_reuse.rs` — §7.3.
18. `benches/simultaneous.rs` — §7.6, plus the `Cargo.toml` `[[bench]]` stanza.

**Group 7 — documentation and release**
19. `README.md` — a "Simultaneous moves" section carrying the §1.2 claim verbatim,
    and a line in the feature list.
20. `benchmarks/BASELINE.md` — the re-run sequential medians, the new simultaneous
    numbers, and the two exploitability figures.
21. `Cargo.toml` — `version = "0.3.0"`.
22. `CHANGELOG.md` (new) — the two source breaks, and the **silent** semantic change:
    `Node::player` is redefined from "the player who moved here" to "the player whose
    reward this node accumulates", which changes what a *re-rooted* root's
    `mean_reward()` reports and resets its `cumulative_reward` to zero. Nothing in-repo
    reads either, and the change fixes a pre-existing inconsistency, but a downstream
    inspector will see it and there is no compile error to warn them.

---

## 9. Open questions for the user

**These are for after the implementation, not now.** The spec proceeds on the stated
default in every case.

1. **Default policy.** The default is `SimultaneousPolicy::RegretMatching`, on the
   grounds that a default which provably cycles does not meet "play optimally". But
   bandit-feedback RM has never been measured inside SM-MCTS — every published RM
   result is the full-matrix variant this design rejects on cost — while decoupled
   UCB1 is the best-attested raw performer. Would you rather default to `Duct` (with
   the randomized tie-breaking, which alone is worth an order of magnitude of
   exploitability on biased RPS) and make the converging policy opt-in until this
   crate has its own numbers? *Default: `RegretMatching`, gated on the two
   exploitability tests passing in the same PR.*
2. **Full-matrix regret matching.** It is the variant every paper benchmarks, and it
   would need a per-participant reward row on each joint edge (8 bytes × arity, arena'd
   on the parent) plus `sum(|A_i|)` hash probes per node per visit — plausibly a 2–4×
   per-iteration cost — and an imputation rule for the missing row entries that no
   theorem covers. It is deliberately excluded. Do you want it behind a third enum
   variant, or left out until a game demands it? *Default: left out; the storage hook
   is noted here so adding it later is additive.*
3. **Arm-count cliff.** Full marginal expansion has no escape hatch by design: a
   simultaneous node where a player has thousands of legal actions allocates thousands
   of 32-byte arms on first visit and scans them every visit. Every knob that would
   soften it (cap the arms, expand progressively) is exactly what voids the
   convergence property. Is documenting the cliff enough, or do you want a
   `debug_assert` warning threshold at, say, 1024 arms? *Default: documented, no
   assert.*
4. **`PlayerSet::MAX_SIMULTANEOUS = 8`.** This is a hard `assert!`, driven by
   `JointKey`'s packing and `starts: [u32; 9]`. Nine concurrent movers is a
   `prod(|A_i|)` with nine factors, so the cap should be unreachable — but it is a hard
   cap in a public type and raising it later means a wider key. Comfortable? *Default:
   8.*
5. **`RootPolicy::Sampled` consumes an rng draw.** At a simultaneous root the returned
   move is non-deterministic for a fixed position under the default, and the draw
   shifts the rng stream for whatever the caller does next with the same generator.
   `MostVisited` opts out. Is the default right, or should determinism win? *Default:
   `Sampled`, because a pure root policy in a mixed-equilibrium game is exploitable by
   construction.*
6. **Tree reuse at simultaneous nodes will usually miss**, because joint successors
   materialize one per iteration out of up to `prod(|A_i|)`. Simultaneous games
   effectively lose tree reuse, and `reused_iterations` will read 0 far more often than
   the sequential experience suggests. There is no fix inherent to lazy joint
   expansion. Should `SearchResult` or `StopReason` signal it, or is the doc comment
   enough? *Default: doc comment only; neither type grows.*
7. **Early termination is dead at mixed simultaneous roots**, so a time-budgeted search
   there always runs to the deadline with no `StopReason` explaining why. Adding a
   variant would break `parallel.rs`'s exhaustive `rank` closure. Worth the break?
   *Default: no new variant.*
8. **`Node::player` redefinition** is a semantic change with no compile error (item 22
   above). Do you want a deprecated `Node::player()` accessor alongside
   `reward_player()` to make the transition visible, or is the changelog enough?
   *Default: changelog and the new accessor; no deprecated alias.*
