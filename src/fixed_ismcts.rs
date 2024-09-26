use crate::Game;
use crate::monte_carlo::{TreePolicy, MonteCarloTreeNode, upper_confidence_bound};
use std::hash::Hash;
use std::fmt::Debug;

#[derive(Clone, Copy, Default)]
pub struct FixedIsmcts;

impl TreePolicy for FixedIsmcts {
    type Stats<Choice: Clone + Hash + Eq + Debug> = ();
    type SimulationData<Choice, PlayerId> = ();

    fn get_selection_value<G: Game>(&self, parent: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>, child: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>) -> f64 {
        let c = 0.4;
        let cumulative_reward = child.cumulative_reward;
        let games = child.games;
        let total_game_count = if parent.is_root() {
            // The root is always fully expanded and the availability of nodes does not change
            parent.games
        }
        else {
            *parent.choice_availability_count.get(child.choice.as_ref().unwrap()).unwrap() as f64
        };
        upper_confidence_bound(cumulative_reward, games, total_game_count, c)
    }
}
