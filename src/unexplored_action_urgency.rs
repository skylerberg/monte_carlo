use float_ord::FloatOrd;

use crate::Game;
use crate::monte_carlo::{TreePolicy, MonteCarloTreeNode};
use std::hash::Hash;
use std::fmt::Debug;

#[derive(Clone, Copy, Default)]
pub struct UnexploredActionUrgency;

impl TreePolicy for UnexploredActionUrgency {
    type Stats<Choice: Clone + Hash + Eq + Debug> = ();
    type SimulationData<Choice, PlayerId> = ();

    fn select<'a, G: Game>(&'a self, node: &'a mut MonteCarloTreeNode<G, Self::Stats<G::Choice>>, game: &'_ G, choices: Option<Vec<<G as Game>::Choice>>) -> &'a mut MonteCarloTreeNode<G, Self::Stats<G::Choice>> {
        let discount = if let Some(choices) = choices {
            let unexplored =
                choices.iter().filter(|choice| {
                    if let Some(child) = node.children.get(&choice) {
                        child.games == 0.0
                    }
                    else {
                        true
                    }
                }).count();
            unexplored as f64 / choices.len() as f64
        }
        else {
            let unexplored = node.children.values().filter(|child| child.games == 0.0).count() as f64;
            unexplored / node.children.len() as f64
        };

        let selected_choice = node
            .children
            .iter()
            .filter(|(choice, _)| {
                game.choice_is_available(choice)
            })
            .max_by_key(|(choice, child)| {
                FloatOrd(if child.games == 0.0 {
                    let total_game_count = if node.is_root() {
                        // The root is always fully expanded and the availability of nodes does not change
                        node.games
                    }
                    else {
                        *node.choice_availability_count.get(choice).unwrap() as f64
                    };

                    let unplayed_action_urgency = 0.5 + 0.4 * f64::sqrt(f64::ln(total_game_count)) * discount;

                    unplayed_action_urgency
                }
                else {
                    self.get_selection_value(node, child)
                })
            })
            .map(|(choice, _)| choice)
            .unwrap()
            .clone();
        node.children.get_mut(&selected_choice).unwrap()
    }

    //fn get_first_play_value<G: Game>(&self, _game: &G, parent: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>, child: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>, choices: &Option<Vec<<G as Game>::Choice>>) -> f64 {
    //    // TODO, we should only calculate this once per simulation
    //    let discount = if let Some(choices) = choices {
    //        let unexplored =
    //            choices.iter().filter(|choice| {
    //                if let Some(child) = parent.children.get(&choice) {
    //                    child.games == 0.0
    //                }
    //                else {
    //                    true
    //                }
    //            }).count();
    //        unexplored as f64 / choices.len() as f64
    //    }
    //    else {
    //        let unexplored = parent.children.values().filter(|child| child.games == 0.0).count() as f64;
    //        unexplored / parent.children.len() as f64
    //    };

    //    let total_game_count = if parent.is_root() {
    //        // The root is always fully expanded and the availability of nodes does not change
    //        parent.games
    //    }
    //    else {
    //        *parent.choice_availability_count.get(child.choice.as_ref().unwrap()).unwrap() as f64
    //    };

    //    0.5 + 0.4 * f64::sqrt(f64::ln(total_game_count)) * discount
    //}
}
