use crate::Game;
use crate::monte_carlo::{TreePolicy, MonteCarloTreeNode, upper_confidence_bound};
use std::hash::Hash;
use std::fmt::Debug;

#[derive(Clone, Copy, Default)]
pub struct SufficiencyTheshold {
    pub threshold: f64,
}

impl TreePolicy for SufficiencyTheshold {
    type Stats<Choice: Clone + Hash + Eq + Debug> = ();
    type SimulationData<Choice, PlayerId> = ();

    fn get_selection_value<G: Game>(&self, parent: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>, child: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>) -> f64 {
        let cumulative_reward = child.cumulative_reward;
        let games = child.games;
        let total_game_count = parent.games;
        let c = 0.4;
        if cumulative_reward / games >= self.threshold {
            let win_rate = cumulative_reward / games;
            win_rate + c * f64::sqrt(f64::ln(total_game_count) / games)
        }
        else {
            upper_confidence_bound(cumulative_reward, games, total_game_count, c)
        }
    }
}
