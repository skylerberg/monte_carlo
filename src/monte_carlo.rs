use std::fmt::Debug;
use std::marker::PhantomData;

use float_ord::FloatOrd;
use rustc_hash::FxHashMap;
use rolling_stats::Stats;

use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::{Status, Game};
use crate::stats::MctsStats;


pub trait MonteCarloTreeSearch {
    type Game: Game;

    fn monte_carlo_tree_search(&mut self, game: Self::Game, iterations: usize) -> (<Self::Game as Game>::Choice, MctsStats) 
    {
        let player_id = game.status().get_active_player_id().unwrap();
        let mut tree: MonteCarloTreeNode<Self::Game> = MonteCarloTreeNode::new(player_id, None);

        for _ in 0..iterations {
            let mut determinization = game.get_determinization(game.status().get_active_player_id());
            let outcome = self.iteration(&mut tree, &mut determinization);
            self.after_iteration(&determinization, outcome);
        }

        let selected_child = tree
            .children
            .values()
            .max_by_key(|child| FloatOrd(child.games))
            .unwrap();

        let mut stats: Stats<f64> = Stats::new();
        tree.children.values().for_each(|child| stats.update(child.cumulative_reward / child.games));

        println!("All child Win %: {}", stats);
        println!(
            "Selected child: (games: {}, wins: {}, win_rate: {:2})",
            selected_child.games,
            selected_child.cumulative_reward,
            selected_child.cumulative_reward / selected_child.games
        );

        (
            selected_child.choice.clone() .unwrap(),
            MctsStats {
                tree_cumulative_reward: tree.cumulative_reward,
                tree_games: tree.games,
            }
        )
    }

    fn after_iteration(&mut self, _game: &Self::Game, _outcome: <Self::Game as Game>::Outcome) {
    }

    // Returns the winner of the iteration
    fn iteration(&mut self, node: &mut MonteCarloTreeNode<Self::Game>, game: &mut Self::Game) -> <Self::Game as Game>::Outcome {
        // If this is a terminal node, immediately retun the outcome
        if let Status::Terminated(outcome) = game.status() {
            self.record_outcome(node, game, &outcome);
            return outcome;
        }

        let choices_available_count = node.expand(game);

        let best_child = self.select(node, game, choices_available_count);
        self.after_selection(game, best_child);
        game.apply_choice(best_child.choice.as_ref().unwrap());

        let outcome = if best_child.games == 0.0 {
            //println!("Rolling out {}", best_child.id);
            let outcome = self.rollout(best_child, game);
            self.record_outcome(best_child, game, &outcome);
            outcome
        }
        else {
            //println!("Recursing from {} to {}", node_id, best_child.id);
            self.iteration(best_child, game)
        };
        //println!("Recording at {} after handling {}", node_id, best_child.id);
        self.record_outcome(node, game, &outcome);
        return outcome;
    }

    fn after_selection(&mut self, _game: &Self::Game, _selected: &MonteCarloTreeNode<Self::Game>) {
    }

    fn get_first_play_value(
        &self,
        _game: &Self::Game,
        _parent: &MonteCarloTreeNode<Self::Game>,
        _child: &MonteCarloTreeNode<Self::Game>,
        _choices: &Option<Vec<<Self::Game as Game>::Choice>>
    ) -> f64 {
        f64::MAX
    }

    fn get_selection_value(&self, _game: &Self::Game, parent: &MonteCarloTreeNode<Self::Game>, child: &MonteCarloTreeNode<Self::Game>) -> f64 {
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
        //upper_confidence_bound(cumulative_reward, games, total_game_count, c)
        //let c = 0.4;
        //let cumulative_reward = child.cumulative_reward;
        //let games = child.games;
        //let total_game_count = parent.games;
        upper_confidence_bound(cumulative_reward, games, total_game_count, c)
    }

    fn select<'a>(
        &'_ self,
        node: &'a mut MonteCarloTreeNode<Self::Game>,
        game: &'_ Self::Game,
        choices: Option<Vec<<Self::Game as Game>::Choice>>
    ) -> &'a mut MonteCarloTreeNode<Self::Game> {
        let selected_choice = node
            .children
            .iter()
            .filter(|(choice, _)| {
                game.choice_is_available(choice)
            })
            .max_by_key(|(_, child)| {
                // TODO make this short-circuit if we find a child with an infinite value (e.g., a child not yet explored)
                FloatOrd(if child.games == 0.0 {
                    self.get_first_play_value(game, node, child, &choices)
                }
                else {
                    self.get_selection_value(game, node, child)
                })
            })
            .map(|(choice, _)| choice)
            .unwrap()
            .clone();
        node.children.get_mut(&selected_choice).unwrap()
    }

    // Returns the winner of the simulation
    fn rollout(&mut self, node: &mut MonteCarloTreeNode<Self::Game>, game: &mut Self::Game) -> <Self::Game as Game>::Outcome  {
        loop {
            let game_status = game.status();
            match game_status {
                Status::AwaitingAction(_player_id) => {
                    let choice = game.get_rollout_choice();
                    let choice = self.intercept_rollout_choice(node, game, choice);
                    game.apply_choice(&choice);
                }
                Status::Terminated(outcome) => {
                    return outcome;
                }
            }
            if let Some(outcome) = game.heuristic_early_terminate() {
                return outcome;
            }
        }
    }

    fn intercept_rollout_choice(
        &mut self,
        _node: &mut MonteCarloTreeNode<Self::Game>,
        _game: &mut Self::Game,
        choice: <Self::Game as Game>::Choice,
    ) -> <Self::Game as Game>::Choice {
        choice
    }

    // Back prop
    fn record_outcome(&mut self, node: &mut MonteCarloTreeNode<Self::Game>, game: &Self::Game, outcome: &<Self::Game as Game>::Outcome) {
        node.cumulative_reward += game.get_reward_for_outcome(node.player_id, outcome);
        node.games += 1.0;
    }
}

#[derive(Clone, Copy, Default)]
pub struct VanillaMcts<G: Game> {
    phantom: PhantomData<G>,
}

impl<G: Game> VanillaMcts<G> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<G: Game> MonteCarloTreeSearch for VanillaMcts<G> {
    type Game = G;
}

#[derive(Debug)]
pub struct MonteCarloTreeNode<G: Game> {
    pub games: f64,
    pub cumulative_reward: f64,
    pub player_id: G::PlayerId,
    pub choice: Option<G::Choice>,
    pub children: FxHashMap<G::Choice, Self>,
    pub choice_availability_count: FxHashMap<G::Choice, usize>,
}


impl<G> MonteCarloTreeNode<G> where G: Game {
    pub fn new(owner: G::PlayerId, choice: Option<G::Choice>) -> Self {
        Self {
            games: 0.0,
            cumulative_reward: 0.0,
            player_id: owner,
            choice,
            children: Default::default(),
            choice_availability_count: Default::default(),
        }
    }

    // Returns the choices available for non-root nodes
    fn expand(&mut self, game: &G) -> Option<Vec<<G as Game>::Choice>> {
        if self.is_root() && !self.children.is_empty() {
            return None;
        }
        let mut rng = thread_rng();
        let active_player = game.status().get_active_player_id().unwrap();
        let mut choices = game.get_all_choices();
        let mut added_new_node = false;
        choices.shuffle(&mut rng);
        for choice in &choices {
            //self.choice_availability_count.entry(choice.clone()).and_modify(|e| *e += 1).or_insert(0);
            if let Some(count) = self.choice_availability_count.get_mut(&choice) {
                *count += 1;
            }
            else {
                self.choice_availability_count.insert(choice.clone(), 0);
            }
            if self.is_root() || (!added_new_node && !self.children.contains_key(&choice)) {
                self.children.insert(choice.clone(), MonteCarloTreeNode::new(active_player, Some(choice.clone())));
                added_new_node = true;
            }
        }
        Some(choices)
    }

    pub fn is_root(&self) -> bool {
        self.choice.is_none()
    }
}

pub fn upper_confidence_bound(cumulative_reward: f64, games: f64, total_game_count: f64, c: f64) -> f64 {
    //let c = 2.0_f64.sqrt();
    let win_rate = cumulative_reward / games;
    win_rate + c * f64::sqrt(f64::ln(total_game_count) / games)
}
