# Transposition support — design

Status: **implemented**, in the unreleased `0.4.0`.

A [`Searcher`] grows a tree of move sequences. Two orders of play that reach
one position grow two nodes, and each learns part of what one node would have.
In games whose moves commute — pick these three cards in any order, resolve
these abilities in any order — that is most of the tree. This adds a second
searcher, [`GraphSearcher`], that merges every node a game identifies as the
same position into one, so each position is searched once and every route into
it reads the same estimate.

---

## 1. Requirements

1. **Zero cost for games that do not opt in.** Not "cheap": a game that never
   asks for transpositions must run the same instructions over the same memory
   it did before, and the sequential golden fingerprint must not move.
2. **Opt-in by type, not by flag.** A game that forgets to supply keys must not
   compile against the merging searcher, rather than silently never merging.
3. **Feature parity** with [`Searcher`]: ISMCTS availability, simultaneous
   nodes under both policies, progressive bias, the root ranking and its
   early-termination proof, tree reuse, root parallelism, and the debug
   assertions that police each.
4. **Identical behaviour where nothing merges.** A graph search over a game
   whose positions are never reached two ways must *be* the tree search, to the
   last bit. That is both a correctness argument and the strongest test the
   port can have.

## 2. Public API

```rust
pub trait Transposition: Game {
    type Key: Clone + Eq + Hash;
    fn transposition_key(&self, ctx: &Self::Context) -> Option<Self::Key>;
}

pub struct GraphSearcher<G: Transposition> { .. }   // Searcher's interface
pub struct Graph<C, K> { .. }                        // the retained graph
pub struct GraphNode<'a, C, K>; pub struct GraphEdge<'a, C, K>;  // read-only views
pub struct GraphRootParallel<G: Transposition, R>;   // behind `parallel`
```

`GraphSearcher` mirrors `Searcher` method for method: `search`,
`reuse_subtree`, `reuse_joint`, `root_policy_into`, with `graph` / `take_graph`
/ `clear_graph` in place of the tree accessors. `GraphNode` and `GraphEdge`
expose what `Node` does — visits, mean and cumulative reward, the reward
player, a simultaneous node's marginals and joint successors — plus the key a
position was merged under and the link-level counts that a tree folds into its
child nodes.

## 3. Why a separate searcher

The tree's hot structures leave no room. `Node<u8>` is exactly 64 bytes, a
cache line, with no padding to spare; the descent records child indices and
backup re-walks them through owned `Vec<Node>`s. A graph needs three things a
tree node does not have: a link's statistics separate from its successor's,
a shared successor that more than one parent can reach, and a per-iteration
stamp to stop on repetitions. Any of them inside `Node` costs every game; a
const flag on `Game` would fold the *code* away but not the *fields*.

So the tree is untouched, and the graph is its own module, `src/graph/`. What
the two share is everything that does not depend on the storage: `Config`,
`SearchResult`, the root ranking (`rank::leader_of`, `Candidate`), the
early-termination proof (`early_stop::proven`, `settled_arms`), the
simultaneous machinery (`Simul`, `duct::select_marginal`,
`duct::credit_marginals`, the root extractions), and the `RootCheck` debug
guard.

What changed in the tree's own files is **visibility, and copies beside the
originals**. Items the graph calls became `pub(crate)` — `Simul::new`,
`early_stop::proven`, the `RootCheck` debug guard, `DEADLINE_CHECK_MASK` — and
`Node::push_joint` now delegates its key bookkeeping to `Simul::record_joint`,
which the graph shares. Everything else the graph needs from the tree's code is
a **copy** beside the original: `Simul::{expand_slot, mark_every_arm_available,
debug_check_arrays}` beside `Node::{expand_marginals,
mark_every_arm_available, debug_check_arms}`, `early_stop::settled_arms`
beside `settled`'s simultaneous branch, and `search::extract_marginal_from`
beside `extract_marginal`.

The copies are deliberate. The first version extracted each body and had the
original delegate to it, which is the obvious factoring and leaves the
behaviour exactly as it was — and it changed the tree's machine code. Link-time
optimization made different interprocedural choices once a body sat behind a
call: `expand_marginals` grew eleven instructions, the simultaneous search loop
moved its early-termination check, and `#[inline(always)]` did not restore
either. Requirement 1 is about instructions, so the originals are verbatim and
the graph has copies. What stops a copy drifting from its original is §7's
equivalence test: a graph that never merges must search exactly as the tree,
simultaneous fixtures included, so any divergence between a copy and its
original fails the suite.

`RootParallel` is the one place the tree's compiled code does change: its pool
and merge are now generic over the kind of worker (§8). That is once-per-search
work after every worker has finished; each worker's search is the unchanged
`Searcher`.

## 4. The graph

Positions are **vertices** in one arena, `Vec<Vertex>`, referring to each other
by `u32` index. A vertex holds what belongs to the *position*: visits and total
reward over every iteration that entered it by any route, the progressive-bias
prior, the key it was merged under, its outgoing links, and — at a
simultaneous position — its marginal bandits and joint successors.

A **link** holds what belongs to the *edge*: the choice, the successor's index
(or `UNRESOLVED` before the first traversal), how often this parent took it,
and how often it was legal here. The last two are exactly what ISMCTS
measures exploration against, and they are per parent: the same position can
be offered by two parents at different rates.

**Merge identity is `(key, mover)`.** Backup is max^n: a node accumulates the
reward of the player who moved into it. Two routes into one position by
different movers would accumulate two players' rewards in one total, so the
mover is part of the identity. The merge table is a `hashbrown::HashTable<u32>`
over arena indices hashed by `(key, mover)`, the same pattern `ChildIndex`
uses, so a key is stored once, on its vertex.

**The root is never in the table.** A root accumulates the player *to act*,
not a mover, so it answers a different question from any merged vertex. A
fresh search's root carries no key; a root promoted by re-rooting keeps its
key, is removed from the table, and is held to it by the debug key check.

## 5. Algorithm

One iteration is the tree's, with the differences below. Everything not listed
— determinization, `advance`, the root's preamble checks, expansion, the
degenerate-node rule — is the same code path or a line-for-line port of it.

### 5.1 Expansion

`Vertex::expand` is `Node::expand` on links: every legal link gains an
opportunity, the root opens a link per legal choice, any other vertex opens one
sampled uniformly from the choices it has no link for, and the
`links_match_choices` flag feeds `CHOICES_ON_REVISIT` exactly as the tree's
does. The same reservoir draws in the same order are what make an unmerged
graph reproduce the tree's random stream.

### 5.2 Selection: UCT2

At a vertex, a link's value is

```
Q(successor) + c * sqrt(ln(N) / n(link)) + w * prior(successor) / (1 + N(successor))
```

where `Q` and `N(successor)` are the successor vertex's shared mean and visits,
`n(link)` the link's own visits, and `ln(N)` either `ln` of this vertex's
visits (the root, and wherever choices never change) or the link's cached
`ln(times_available)` (ISMCTS everywhere else) — the same rule the tree
applies to child nodes.

This is UCT2 in the terms of Childs, Brodeur and Kocsis [1]: the **value** is
the position's, pooled over every route, which is the point of merging; the
**exploration** is the edge's, because the confidence bound is about how often
*this parent* has tried *this choice*. Using the successor's visits in the
exploration term instead would let a position popular elsewhere starve its
other parents' exploration of it.

Where a successor has only one parent, `N(successor) == n(link)` and the
expression is the tree's `ucb_raw` term for term, in the same operation order —
which is why §1.4 holds to the bit rather than approximately.

### 5.3 Resolving a link

A link's successor is unknown until it is first taken, because the key is read
off the state the choice produces. On that traversal:

* the key is looked up under `(key, mover)`; a hit links to the existing
  vertex, a miss inserts one and registers it;
* a `None` key inserts an unregistered vertex — a tree node, in effect.

The descent **rolls out only at a new position**. Reaching an existing vertex
by a new route continues the descent into what is already known there, as a
tree continues into a visited child. Rolling out instead would spend an
iteration re-estimating a position that already has an estimate, and leave its
subtree no deeper.

Every later traversal of the link lands in the same vertex. A determinization
or chance event may produce a different *state* on the way; the contract (§6)
is that it cannot produce a different *position*.

Nor can it change the vertex's mover, which was fixed when the link resolved.
Where the player choosing the link varies between visits, because chance or
hidden information decides whose turn it is, the successor keeps accumulating
for the player it was resolved for. A tree child keeps the player it was first
stamped for in the same way, so this is not taken as a broken key.

### 5.4 Repetition

A tree cannot repeat; a graph can. Each iteration takes a fresh stamp, and
every vertex entered is stamped with it. A link whose successor already carries
the current stamp leads back onto the path being searched: the descent counts
the link as taken, rolls out from the current state, and stops. The successor
is not credited a second time — it is already on the path, and backup credits
each vertex once.

This guarantees termination whatever the keys do, at the cost of estimating a
repetition by rollout rather than by any rule of the game; a game that scores
repetition specially should put that rule in its own state and key.

Stamps are `u32`. On wrap-around every vertex's stamp is cleared before the
counter restarts at 1 (a new vertex starts at 0), so no stale stamp can equal a
reissued one.

### 5.5 Backup

Every vertex on the path records the reward for its own player; every link on
the path gains a visit. A simultaneous vertex's arms are credited *before* its
record, for the reason the tree's backup gives: the arm legality test is
`avail_epoch == visits`.

In a graph with no repetitions and no re-rooting this conserves counts exactly:
every non-root vertex has been entered precisely as often as its incoming links
were taken. The tests assert it.

### 5.6 Simultaneous positions

A simultaneous vertex holds a `Simul` — the same marginal bandits, expanded,
selected and credited by the same functions — and a vector of joint
successors parallel to `Simul::joint_keys`. A joint successor is resolved like
a link, with the **perspective player as mover**, which is whose reward a
tree's joint child accumulates. Joint successors carry no visit counts of
their own: selection happens on the arms.

### 5.7 The root's answer and early termination

A root link is a ranking candidate with the **link's** selections and
opportunities — the root's own evidence, which is what the evidence bar and the
rate tiebreak are about — and the **successor's** mean, which is the best
estimate of what the choice is worth. Where the successor has no other parent
its total reward is passed through untouched rather than rebuilt from the mean,
so an unmerged graph ranks to the bit as the tree does. The early-termination
proof runs over the same candidates through `early_stop::proven`; a
simultaneous root goes through `settled_arms` and `extract_marginal` unchanged.

### 5.8 Re-rooting

`reuse_subtree` and `reuse_joint` find the successor and compact the arena down
to what it reaches: breadth-first from the new root, survivors renumbered in
visit order, every link remapped, the table rebuilt without the root. Two
consequences differ from a tree.

* **More survives.** A position reached through a sibling of the played choice
  *and* through the played choice is still ahead, and keeps everything learned
  about it by either route.
* **Links into the new root are cut.** They are repetitions through the root,
  and the root's total is now kept in the player-to-act's currency, which a
  link reads in the mover's. Each such link is reset to unresolved and
  re-resolves to an ordinary vertex if taken again.

A survivor keeps the visits it took through parents that were discarded, so
after a re-root a vertex has been entered *at least* as often as its surviving
links were taken, and conservation (§5.5) becomes an inequality.

## 6. The key contract

The key is the whole claim the game makes, and the search trusts it:

1. **Equal keys mean interchangeable positions** for the rest of the game:
   whose turn it is, what is legal, and what any line pays.
2. **A vertex's key is the same on every visit.** A link resolves once; every
   later iteration that takes it lands in its vertex whatever determinization it
   drew and whatever chance produced on the way. So a key may read only what is
   fixed for all of them — never anything a determinization resamples, and never
   a chance outcome. Where the state is not fixed by the moves that reached it,
   the key must be `None`.

`None` is always sound, which is what makes the contract practical: a game
merges only the phases of play where nothing hidden or random moves, and
declines everywhere else.

**Debug builds enforce (2) on every visit**, at the root of every iteration and
at every resolved link and joint successor, and panic with the reason. A key
that reads hidden state is caught on the first iteration that draws a different
world into a merged position. (1) is not checkable by the crate; a mismatch in
kind or participants between merged positions is, and is asserted.

## 7. What is claimed, and what is not

* **Claimed and tested:** where no position is reached two ways — every key
  `None`, or every key unique to its move history — `GraphSearcher` returns what
  `Searcher` returns and builds the same statistics everywhere, bit for bit,
  across the whole sequential fingerprint matrix, re-rooting, every
  simultaneous fixture under both policies and both root policies, and root
  parallelism.
* **Claimed and tested:** every route into a position reads one vertex; counts
  are conserved (§5.5); repetitions terminate; a key that reads resampled state
  is refused in debug; a saturated graph search allocates nothing per
  iteration.
* **Not claimed: convergence.** UCT's regret bound is proved for trees.
  UCT2-style selection on a graph is widely used and well behaved in practice
  [1, 2], but sharing a successor's mean across parents means a parent's own
  mean lags improvements its successors received through other parents, and no
  bound covers that here. Monte-Carlo graph search [3] corrects the lag by
  propagating value differences to every parent; its correction needs a full
  reward vector per node under max^n, which is `size_of::<Rewards>()` per vertex
  and is not implemented (§10).
* **Not claimed: anything stronger under determinization** than the tree
  claims. Merging is only sound for positions identical across the
  determinizations that reach them (§6), so it adds no strategy-fusion error of
  its own, and removes none.

## 8. Root parallelism

`GraphRootParallel` runs one `GraphSearcher` per thread and merges the roots by
the same rule and the same code as `RootParallel`: the pool and the merge are
generic over a crate-private worker trait that both searchers implement. Graphs
are never merged across workers; only root candidates are pooled, keyed by
`Choice`. `RootParallel`'s public API is unchanged.

## 9. Evidence

* **Machine code.** `benchmarks/codegen_diff.py` disassembles two builds,
  normalizes addresses and symbol hashes away, and compares every function's
  instruction stream. Between `main` and this branch, every function in the
  `search` bench binary (1 630) and in the `simultaneous` bench binary (1 620) is
  identical: the tree search a game that does not opt in runs is the same
  instructions it was. Identical instructions settle a cost question that a
  benchmark on a loaded machine could only bound to a few percent.
* `tests/sequential_fingerprint.rs` — the golden file is unchanged.
* `tests/allocation.rs` — unchanged for the tree; graph cases added for a
  merging chain, a wide game merging behind the hash index, and two games that
  never merge.
* `tests/transpositions.rs` — the equivalence matrix of §7, merging, counts,
  repetition, the leaky key, re-rooting, determinism, forced moves, and the
  parallel pool.
* `src/graph/node.rs` unit tests — stamp wrap-around, compaction, and the
  `(key, mover)` table.
* `benches/transposition.rs`, `examples/transposition_quality.rs` and
  `benchmarks/BASELINE.md` — what the graph costs where it cannot merge, and
  what it buys where it can.

## 10. Rejected alternatives

* **Shared statistics on an unchanged tree** (a transposition table beside the
  tree, as in early Go programs). Selection would need each child's position
  key, which means storing it per child — the cost requirement 1 forbids — and
  every duplicate node would still re-expand its children one visit at a time.
* **A storage-generic `Searcher`.** Expressing the tree as one implementation of
  a storage trait would put the graph's indirections behind an abstraction in
  the tree's hot loop and stake requirement 1 on the optimizer. Two searchers
  sharing every storage-independent function is less code at risk.
* **Monte-Carlo graph search's value correction** [3]. Needs a per-vertex
  reward vector for max^n; deferred until a game needs it.

## References

[1] Childs, B. E., Brodeur, J. H. & Kocsis, L. (2008). *Transpositions and Move
Groups in Monte Carlo Tree Search.* IEEE Symposium on Computational
Intelligence and Games, 389–395.

[2] Saffidine, A., Cazenave, T. & Méhat, J. (2012). *UCD: Upper Confidence bound
for rooted Directed acyclic graphs.* Knowledge-Based Systems, 34, 26–33.

[3] Czech, J., Korus, P. & Kersting, K. (2021). *Improving AlphaZero Using
Monte-Carlo Graph Search.* Proceedings of the International Conference on
Automated Planning and Scheduling, 31(1), 103–111.
