use weighted_rand::builder::{WalkerTableBuilder, NewBuilder};

use crate::Game;
use crate::monte_carlo::{TreePolicy, MonteCarloTreeNode};
use std::hash::Hash;
use std::fmt::Debug;

#[derive(Clone, Copy, Default)]
pub struct Exp3 {
    pub gamma: f64,
    pub eta: f64,
}

impl TreePolicy for Exp3 {
    type Stats<Choice: Clone + Hash + Eq + Debug> = f64;  // Selection probability
    type SimulationData<Choice, PlayerId> = ();

    fn select<'a, G: Game>(&'a self, node: &'a mut MonteCarloTreeNode<G, Self::Stats<G::Choice>>, game: &'_ G, _choices: Option<Vec<<G as Game>::Choice>>) -> &'a mut MonteCarloTreeNode<G, Self::Stats<G::Choice>> {
        let mut unexpored_choice = None;
        for (choice, child) in node.children.iter() {
            if child.games == 0.0 {
                unexpored_choice = Some(choice);
                break
            }
        }
        if let Some(choice) = unexpored_choice {
            let unexplored_node = node.children.get_mut(&choice.clone()).unwrap();
            unexplored_node.node_statistics = 1.0;
            return unexplored_node;
        }

        let (children, expected_rewards): (Vec<_>, Vec<_>) = node.children
            .iter()
            .filter(|(choice, _)| {
                game.choice_is_available(choice)
            })
            .map(|(_, child)| {
                (
                    child,
                    std::f64::consts::E.powf(-1.0 * self.eta * child.cumulative_reward)
                )
            }).unzip();

        let total_expected_reward: f64 = expected_rewards.iter().sum();

        let mut weights: Vec<f32> = vec![];
        for child in &children {
            let probability_of_selecting_child = (1.0 - self.gamma) * (std::f64::consts::E.powf(self.eta * child.cumulative_reward) / total_expected_reward) + self.gamma;
            weights.push(probability_of_selecting_child as f32);
        }

        let child_index = WalkerTableBuilder::new(&weights).build().next();

        let selected_child = node.children.get_mut(&children[child_index].choice.as_ref().unwrap().clone()).unwrap();
        //println!(
        //    "{:.4} {:.4} {:.4} {:.4} {:.4}",
        //    (1.0 - self.gamma),
        //    (std::f64::consts::E.powf(self.eta * selected_child.cumulative_reward)),
        //    total_expected_reward,
        //    self.gamma,
        //    (1.0 - self.gamma) * (std::f64::consts::E.powf(self.eta * selected_child.cumulative_reward) / total_expected_reward) + self.gamma,
        //);
        selected_child.node_statistics = (1.0 - self.gamma) * (std::f64::consts::E.powf(self.eta * selected_child.cumulative_reward) / total_expected_reward) + self.gamma;
        selected_child
    }

    // Back prop
    fn record_outcome<G: Game>(node: &mut MonteCarloTreeNode<G, Self::Stats<G::Choice>>, game: &G, outcome: &G::Outcome, _additional_data: &mut Self::SimulationData<G::Choice, G::PlayerId>) {
        node.cumulative_reward += game.get_reward_for_outcome(node.player_id, outcome) / node.node_statistics;
        node.games = node.cumulative_reward + 0.000000001;

    }
}
