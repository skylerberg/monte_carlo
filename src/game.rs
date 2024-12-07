use std::hash::Hash;

use rand::{seq::SliceRandom, thread_rng};

pub use crate::{MonteCarloTreeSearch, VanillaMcts};

pub trait Game: Clone {
    type Choice: Eq + Hash + Clone;
    type PlayerId: Copy;

    fn get_all_choices(&self) -> Vec<Self::Choice>;

    fn apply_choice(&mut self, choice: &Self::Choice);

    fn get_active_player_id(&self) -> Self::PlayerId;

    fn is_terminal(&self) -> bool;

    fn reward_for(&self, player_id: Self::PlayerId) -> f64;

    // Change for non-deterministic games
    fn choice_is_available(&self, _choice: &Self::Choice) -> bool {
        return true;
    }

    // Change for non-deterministic games
    fn get_determinization(&self, _from_perspective: Self::PlayerId) -> Self {
        self.clone()
    }

    fn get_rollout_choice(&self) -> Self::Choice {
        let mut rng = thread_rng();
        self.get_all_choices().choose(&mut rng).unwrap().clone()
    }

    fn heuristic_early_terminate(&self) -> bool {
        false
    }

    fn shuffle_on_expand(&self) -> bool {
        true
    }

    // Meant for quick debugging purposes
    fn run(&mut self, iterations: usize) {
        let mut mcts: VanillaMcts<Self> = VanillaMcts::new();
        while !self.is_terminal() {
            let (choice, _) = mcts.monte_carlo_tree_search(self.clone(), iterations);
            self.apply_choice(&choice);
        }
    }
}
