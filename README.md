# monte_carlo

A rust library for Monte Carlo Tree Search.

Implements Monte Carlo Tree Search (MCTS) using Upper Confidence Bound for Trees
(UCT).

Many optimizations are from [1].

## Planned

### MCTS-Solver [3]

This approach takes into account proven wins and losses.

* Decisive and Anti-Decisive Moves [2]


## Brain Dump

Built in optimizations:
* 

Configurable variants:
* Nondeterministic MCTS
  * Determinization
  * Information Set UCT (ISUCT) (I think I have this implemented)

Not implemented variants:
* Flat UCB
* Bandit Algorithm for Smooth Trees (BAST) (extends Flat UCB)
* Learning in MCTS (Is this actually a variant?)
* Single-Player MCTS (SP-MCTS)
  * Feature UCT Selection (FUSE)
* Multi-player MCTS
  * Coalition Reduction
* Multi-agent MCTS
  * Ensemble UCT
* Real-time MCTS
* Nondeterministic MCTS
  * Hindsight optimisation (HOP)
  * Sparse UCT
  * Multiple MCTS
  * UCT+
  * Monte Carlo alpha-beta (MC_alpha_beta)
  * Monte Carlo Counterfactual Regret (MCCFR)
  * Inference and Opponent Modelling
  * Simultaneous Moves
* Recursive Approaches
  * Reflexive Monte Carlo Search
  * Nested Monte Carlo Search
  * Nested Rollout Policy Adaptation (NRPA)
  * Meta-MCTS
  * Heuristically Guided Swarm Tree Search
* Sample-Based Planners
  * Forward Search Sparse Sampling (FSSS)
  * Threshold Ascent for Graphs (TAG)
  * RRTs
  * UNLEO
  * UCTSAT
  * _rho UCT
  * Monte Carlo Random Walks (MRW)
  * Mean-based Heuristic Search for Anytime Planning (MHSP)




[1] Browne, Cameron & Powley, Edward & Whitehouse, Daniel & Lucas, Simon & Cowling, Peter & Rohlfshagen, Philipp & Tavener, Stephen & Perez Liebana, Diego & Samothrakis, Spyridon & Colton, Simon. (2012). A Survey of Monte Carlo Tree Search Methods. IEEE Transactions on Computational Intelligence and AI in Games. 4:1. 1-43. 10.1109/TCIAIG.2012.2186810.

[2] Teytaud, Fabien & Teytaud, Olivier. (2010). On the Huge Benefit of Decisive Moves in Monte-Carlo Tree Search Algorithms. Proceedings of the 2010 IEEE Conference on Computational Intelligence and Games, CIG2010. 359 - 364. 10.1109/ITW.2010.5593334. 

[3] Winands, Mark & Björnsson, Yngvi & Saito, Jahn-Takeshi. (2008). Monte-Carlo Tree Search Solver. 25-36. 10.1007/978-3-540-87608-3_3.
