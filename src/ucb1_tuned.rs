use crate::Game;
use crate::monte_carlo::{TreePolicy, MonteCarloTreeNode};
use std::hash::Hash;
use std::fmt::Debug;

#[derive(Clone, Copy, Default)]
pub struct Ucb1TunedPolicy;

impl TreePolicy for Ucb1TunedPolicy {
    type Stats<Choice: Clone + Hash + Eq + Debug> = ();
    type SimulationData<Choice, PlayerId> = ();

    fn get_selection_value<G: Game>(&self, parent: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>, child: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>) -> f64
    {
        // TODO what happens when there are ties that add a non zero reward? Do I need to recalculate variance?
        // Bind math operations to more concise names to make equations easier to read
        let ln = f64::ln;
        let sqrt = f64::sqrt;
        let min = f64::min;

        let cumulative_reward = child.cumulative_reward;
        let games = child.games;

        let total_game_count = if parent.is_root() {
            // The root is always fully expanded and the availability of nodes does not change
            parent.games
        }
        else {
            *parent.choice_availability_count.get(child.choice.as_ref().unwrap()).unwrap() as f64
        };

        let max_bernoulli_random_variable_variance = 0.25;
        let average_reward = cumulative_reward / games;
        let sample_variance = average_reward * (1.0 - average_reward);
        let max_variance_for_arm = sample_variance + sqrt((2.0 * ln(total_game_count)) / games);

        average_reward + sqrt((ln(total_game_count) / games) * min(max_bernoulli_random_variable_variance, max_variance_for_arm))
    }
}
