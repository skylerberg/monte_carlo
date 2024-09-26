use float_ord::FloatOrd;

use crate::{Game, Status};
use crate::monte_carlo::{TreePolicy, MonteCarloTreeNode};
use std::collections::{BTreeMap, HashSet};
use std::hash::Hash;
use std::fmt::Debug;

#[derive(Clone, Copy, Default)]
pub struct RaveTreePolicy {
    pub first_play_value: f64,
}

impl TreePolicy for RaveTreePolicy {
    type Stats<Choice: Clone + Hash + Eq + Debug> = RaveStat;
    type SimulationData<Choice, PlayerId> = BTreeMap<PlayerId, HashSet<Choice>>;

    fn get_first_play_value<G: Game>(&self, _game: &G, _parent: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>, _child: &MonteCarloTreeNode<G, Self::Stats<G::Choice>>, _choices: &Option<Vec<<G as Game>::Choice>>) -> f64 {
        self.first_play_value
    }

    fn select<'a, G: Game>(&'a self, node: &'a mut MonteCarloTreeNode<G, Self::Stats<G::Choice>>, game: &'_ G, choices: Option<Vec<<G as Game>::Choice>>) -> &'a mut MonteCarloTreeNode<G, Self::Stats<G::Choice>> {
        node
            .children
            .iter_mut()
            .filter(|(_, child)| {
                game.choice_is_available(child.choice.as_ref().unwrap())
            })
            .max_by_key(|(_, child)| {
                // TODO make this short-circuit if we find a child with an infinite value (e.g., a child not yet explored)
                FloatOrd(if child.games == 0.0 {
                    // Ugh, I passed child in twice just to appease the borrow checker. It doesn't use either
                    self.get_first_play_value(game, child, child, &choices)
                }
                else {
                    let equivalence_parameter = 500.0; // After 1000 iterations, RAVE and UTC will be equally weighted
                    // TODO switch to minimum MSE schedule mentioned in paper
                    let weighting_parameter = f64::sqrt(equivalence_parameter / (3.0 * f64::ln(node.games) + equivalence_parameter));

                    let rave_stat = child.node_statistics;
                    let win_rate = child.cumulative_reward / child.games;
                    let c = 0.4;
                    let exploration_term = c * f64::sqrt(f64::ln(node.games) / child.games);
                    weighting_parameter * rave_stat.average_reward() + (1.0 - weighting_parameter) * win_rate + exploration_term
                })
            })
            .map(|(_, child)| child)
            .unwrap()
    }

    fn rollout<G: Game>(&self, _node: &mut MonteCarloTreeNode<G, Self::Stats<G::Choice>>, game: &mut G) -> (G::Outcome, Self::SimulationData<G::Choice, G::PlayerId>) {
        let mut choices: BTreeMap<G::PlayerId, HashSet<G::Choice>> = Default::default();
        let outcome = loop {
            match game.status() {
                Status::AwaitingAction(player_id) => {
                    let choice = game.get_rollout_choice();
                    game.apply_choice(&choice);
                    let choices_by_player = choices.entry(player_id).or_default();
                    choices_by_player.insert(choice);
                }
                Status::Terminated(outcome) => {
                    break outcome;
                }
            }
        };

        (outcome, choices)
    }

    // Back prop
    fn record_outcome<G: Game>(node: &mut MonteCarloTreeNode<G, Self::Stats<G::Choice>>, game: &G, outcome: &G::Outcome, additional_data: &mut Self::SimulationData<G::Choice, G::PlayerId>) {
        if let Some(choice) = &node.choice {
            let choices = additional_data.entry(node.player_id).or_default();
            choices.insert(choice.clone());
        }
        for child in node.children.iter_mut().map(|(_, child)| child) {

            let reward = game.get_reward_for_outcome(child.player_id, outcome);
            if let Some(choices) = additional_data.get_mut(&child.player_id) {
                if choices.contains(child.choice.as_ref().unwrap()) {
                    child.node_statistics = child.node_statistics + RaveStat::new(reward, 1.0);
                }
            }
        }
        //let reward = outcome.reward_for(node.player_id);
        //for (acting_player, choice) in additional_data.iter() {
        //    if *acting_player == node.player_id {
        //        //println!("{} - {:?}", node.id, choice.clone());
        //        let entry = node.node_statistics.entry(choice.clone()).or_insert(RaveStat::new(0.0, 0.0));
        //        let new_stat = RaveStat::new(reward, 1.0);
        //        *entry = *entry + new_stat;
        //    }
        //}
        node.cumulative_reward += game.get_reward_for_outcome(node.player_id, outcome);
        node.games += 1.0;
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct RaveStat {
    cumulative_reward: f64,
    games: f64,
}

impl RaveStat {
    pub fn new(reward: f64, games: f64) -> Self {
        RaveStat {
            cumulative_reward: reward,
            games,
        }
    }

    pub fn average_reward(&self) -> f64 {
        self.cumulative_reward / self.games
    }
}

impl std::ops::Add<RaveStat> for RaveStat {
    type Output = RaveStat;

    fn add(self, rhs: RaveStat) -> Self::Output {
        RaveStat {
            cumulative_reward: self.cumulative_reward + rhs.cumulative_reward,
            games: self.games + rhs.games,
        }
    }
}
