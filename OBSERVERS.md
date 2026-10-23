# One tree per observer

How `ObservedSearcher` handles moves that conceal something, what it claims, and
what it does not. `TRANSPOSITIONS.md` is the same document for the other
separate searcher.

## 1. The failure it exists for

`Searcher` identifies a tree edge by `Game::Choice`, and the `Choice` contract
requires full value identity. That is right for everything the search does — an
edge's statistics belong to exactly that move — but it has a consequence that
does not show up until a move hides something.

Consider a player who can play either of two cards **face down**, so the
opponent learns that a card was played and not which. Those are two `Choice`
values, so `Searcher` grows two subtrees. The simulated opponent has a node in
each, and each of those nodes has only ever been reached in worlds where that
particular card was behind the move. So the opponent's statistics in each are
conditioned on the secret: it passes against the strong card and calls against
the weak one, exactly as though it had seen them.

The searcher then prices every bluff against a response that already knows. The
bluff is worth what disclosure is worth, so it is never played, and no amount of
budget changes that — the leak is in the identity of the node, not in its
statistics.

This is the strategy-fusion failure Cowling, Powley and Whitehouse describe for
single-observer ISMCTS, and `Searcher` is deliberately a single-observer
searcher: one tree, one observer's information sets, every player in it
simulated inside those.

## 2. The fix, and why both halves are needed

A **key** is a move as one particular player perceives it. Two moves that player
cannot tell apart share a key, so they share a node, and that node's statistics
are the mixture over what the move might have been — the observer's information
set, and the belief a deception is played against.

A **tree per player** is what makes the key useful. With one tree, the
opponent's decisions would still be nodes in the searcher's tree, and a node
there is one the searcher can distinguish. Each player needs its own tree, keyed
by its own perceptions, and each player's moves chosen from its own.

One consequence is worth stating plainly, because it is the mechanism rather than
an implementation detail: the mixture at a merged node is weighted by how often
the searcher takes each of the concealed branches. That is the belief — a move's
value depends on the strategy that produces it, and the search reaches the
fixed point the same way any UCT search reaches one, by revisiting and
re-estimating.

## 3. Public API

```rust
pub trait Observed: Game {
    type Key: Clone + Eq + Hash;
    fn observed_key(&self, ctx: &Self::Context, choice: &Self::Choice,
                    actor: u8, observer: u8) -> Self::Key;
}

impl<G: Game + Observed> ObservedSearcher<G> {
    pub fn new(template: &G, players: u8) -> Self;
    pub fn players(&self) -> u8;
    pub fn tree(&self, player: u8) -> Option<&ObservedNode<G::Key>>;
    pub fn clear_tree(&mut self);
    pub fn reuse_subtree(&mut self, state: &G, ctx: &G::Context, choice: &G::Choice) -> bool;
    pub fn search<R: Rng + ?Sized>(&mut self, state: &G, ctx: &G::Context,
        perspective: u8, cfg: &Config, cancel: Option<&AtomicBool>, rng: &mut R)
        -> SearchResult<G::Choice>;
}
```

`search` takes the same `Config`, returns the same `SearchResult`, and reports
the same `StopReason`s as `Searcher`. `players` is the number of seats, so every
tree is on the same path from the first iteration; the `search` perspective must
be the player to act, because the answer is read off that player's tree.

`reuse_subtree` takes the state the retained trees were built from, because a
perception is a fact about the position the move was made in and has to be read
from it.

## 4. Why a separate searcher

The same reason `GraphSearcher` is separate, and it is a hard requirement rather
than a preference:

* `Game` gains no method and no associated type. `Observed` is a separate trait,
  so a game that conceals nothing never names a key type.
* `Searcher`, `Node`, `select`, and the sequential descent are untouched. An
  edge's identity changes, and that changes the node — so the observed searcher
  has its own node in `src/observed/node.rs`, exactly as the graph has
  `Vertex`/`Link` in `src/graph/node.rs`. Only leaf helpers are shared:
  `select::ucb_raw`, `rank`, `early_stop`'s absence, `util`.
* Nothing is gated behind a cargo feature, matching `GraphSearcher`. The crate's
  features are platform gates (`time`, `parallel`), not size gates; a generic
  searcher a consumer never names emits no code.
* The obvious cheaper designs are refused for the same reason. `observed_key` on
  `Game` with a default would hand every game a key type and a decision, and
  would make node identity a mode of `Searcher`; genericizing `Node<C>` over an
  edge payload would put the observed searcher's needs into the struct the
  sequential hot path owns.

`tests/sequential_fingerprint.rs` is unchanged and green, which is the crate's
standing evidence that this claim holds rather than being asserted.

## 5. The algorithm

One iteration:

1. Determinize **once**, from the search perspective, and `advance` — exactly as
   `Searcher` does. Worlds are drawn from one player's knowledge; each observer's
   tree then filters them by observation.
2. Descend. At each step `status` names the actor:
   * enumerate that actor's moves and key each one **in the actor's own tree**
     (`observed_key(choice, actor, actor)`);
   * expand and select in the actor's tree, and apply the selected move;
   * push the same step into **every other tree** under that tree's own key
     (`observed_key(choice, actor, observer)`), creating the child if the tree
     has never taken the edge.
   The action applied is always re-derived from the world being searched. Only
   the perception is stored, because a stored `Choice` would be replayed into a
   later determinization in which the concealed card behind it had been
   resampled — the trap the `Choice` contract exists to prevent.
3. Roll out at a leaf, as `Searcher` decides a leaf: the first visit to the
   actor's child.
4. Back up every tree's path, each node credited with the reward of the player
   who moved into it. max^n backup composes unchanged: a node's owner is the
   actor, not the observer — the observer only decides *which* node.

Only the acting player's tree is read for selection, so a tree grows statistics
at its own decisions and scaffolding elsewhere. A node's actor is a function of
the position, which is what makes that safe.

The availability rule is `Searcher`'s: the exploration denominator below the
root is the child's own `times_available`, and the root's is the parent's visit
count.

## 6. The key contract

A key must depend on **what the observer knows and on the move**, and nothing
else. Four consequences:

* **Never information the observer lacks.** A key that reads it splits one of the
  observer's information sets into many, and the simulated player then chooses at
  each with knowledge it does not have. Debug builds check this directly: the
  searcher re-determinizes from the observer's own perspective — which by
  definition leaves everything the observer knows untouched — and asserts the key
  does not change. That check has its own generator, so a debug build searches
  exactly the tree a release build does.
* **Coarser is safe, finer is not.** Merging moves the observer *could* tell
  apart only makes it decide as though it knew less. Splitting moves it could
  not tell apart is the leak. So the safe rule is identity for public moves and
  collapse exactly what is concealed — and that also keeps the tree close to the
  size a single-observer one would be.
* **The move need not be legal where the key is read.** The debug check reads it
  in a world where whatever the move concealed has been resampled.
* **`actor == observer` must be injective.** The acting player knows what it is
  doing, and the root's answer is mapped back from the key. Debug builds assert
  it.

Prefer a structural key — an enum, an index, a packed integer — over a hash. A
collision merges two moves that play differently, silently; there is no
`Option` in the return type to decline a merge with.

## 7. What is claimed, and what is not

**Claimed.** Merging what an observer cannot distinguish makes that observer's
simulated decisions a function of its own information set, so a concealment is
priced against a belief rather than against the truth. `tests/observed.rs`
measures it on a fixture with a known answer:

| | clairvoyant (`Searcher`) | merged (`ObservedSearcher`) |
|---|---|---|
| seat 0's value | 0.4691 | 0.5466 |
| equilibrium value | 0.4667 | 0.5556 |
| share of budget on the bluff | 0.027 | 0.303 |
| opponent's answers to a concealed move | call 0.95 against the weak card, pass 0.99 against the strong one | call 0.43 / pass 0.57, one node |

The fixture's payoff table is in `tests/common/bluff.rs`; the numbers are stable
across seeds, not a favourable run.

**Not claimed.**

* **Equilibrium play.** This is still MCTS with a sampled opponent model. The
  mixture at a merged node is the searcher's own visit distribution, so the
  belief is only as good as the strategy that produced it. It is not CFR, it has
  no regret bound, and it does not solve poker.
* **Full MO-ISMCTS information sets.** A tree is keyed by the perceptions the
  game names, and a coarse key is allowed — including one that leaves out
  something the observer *does* know, such as its own hand. That is safe and it
  keeps the tree small; it makes the simulated player weaker, not clairvoyant.
* **Concurrent moves.** `Status::Simultaneous` is a panic. DUCT's per-participant
  arms are decoupled *selection* at one node, not an information set per
  observer, and a game with both kinds of node needs a design that says what a
  joint action's perception is.
* **Transpositions on top.** Two notions of node identity in one searcher is how
  a tree stops being explainable. A graph over per-observer keys would be a third
  searcher.
* **Early termination.** The proof in `early_stop` is about one tree's
  candidates; here other trees grow alongside the one the answer is read from.
  `Config::early_termination` is ignored rather than honoured against the wrong
  statistics.

## 8. Rejected alternatives

* **Keep one tree, merge concealed edges with a coarser `Choice::Eq`.** The
  `Choice` contract forbids it for a reason: the tree stores the first `Choice`
  it saw for an edge and replays it into later iterations, so a coarse `Eq` feeds
  one determinization's move into another's world.
* **`Transposition` with a key that ignores the concealed card.** It merges
  nodes the contract says are not interchangeable, replay a stored `Choice`
  across worlds, and has no way to say *which observer* the merge is for — every
  observer would get the same merge.
* **Opponent decisions outside the tree, in `Side`/`advance`.** This is how a
  side model is supposed to work, and it is leak-free if the model's policy
  respects its own player's knowledge. It gives up the opponent's search, though,
  and it has to decide whether an opponent's node is an information set — which
  is the same key this searcher needs. Building that model correctly converges on
  a one-node MCTS per decision.
* **DUCT.** Per-participant arms at a simultaneous node decouple *selection*, so
  the second mover cannot answer the first. They do not key by observer: the arms
  live in one node of the searching player's tree, and the opponent's values are
  still conditioned on what the searcher knows. It fixes the other half of the
  problem, in a searcher that does not have this one.
* **A bluff bonus in the evaluator.** It encodes a policy as a value, cannot be
  derived from the rules, and leaves the opponent's subtree exactly as
  clairvoyant as it was. The leak is in the information set, not the evaluation.
