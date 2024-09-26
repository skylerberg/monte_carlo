use std::fmt::Debug;
use std::hash::Hash;

use rand::{thread_rng, seq::SliceRandom};

pub use crate::{MonteCarloTreeSearch, VanillaMcts};

pub enum Status<P: Copy, O: Outcome<P>> {
    AwaitingAction(P),
    Terminated(O),
}

impl<P: Copy, O: Outcome<P>> Status<P, O> {
    pub fn get_active_player_id(&self) -> Option<P> {
        match self {
            Status::AwaitingAction(player_id) => Some(*player_id),
            Status::Terminated(_) => None,
        }
    }
}

pub trait Outcome<P: Copy> {
    fn reward_for(&self, player_id: P) -> f64;
}

pub trait Game: Clone {
    type Choice: Eq + Hash + Clone + Default + Debug;
    type PlayerId: Copy + Eq;
    type Outcome: Outcome<<Self as Game>::PlayerId>;

    fn get_all_choices(&self) -> Vec<Self::Choice>;
    fn apply_choice(&mut self, choice: &Self::Choice);
    fn status(&self) -> Status<Self::PlayerId, Self::Outcome>;

    // Change for non-deterministic games
    fn choice_is_available(&self, _choice: &Self::Choice) -> bool {
        return true;
    }

    // Change for non-deterministic games
    fn get_determinization(&self, _from_perspective: Option<Self::PlayerId>) -> Self {
        self.clone()
    }

    fn get_rollout_choice(&self) -> Self::Choice {
        let mut rng = thread_rng();
        self.get_all_choices().choose(&mut rng).unwrap().clone()
    }

    fn heuristic_early_terminate(&self) -> Option<Self::Outcome> {
        None
    }

    fn get_reward_for_outcome(&self, player_id: Self::PlayerId, outcome: &Self::Outcome) -> f64 {
        outcome.reward_for(player_id)
    }

    // Meant for quick debugging purposes
    fn run(&mut self, iterations: usize) {
        let mut mcts: VanillaMcts<Self> = VanillaMcts::new();
        loop {
            match self.status() {
                Status::AwaitingAction(_) => {
                    let (choice, _) = mcts.monte_carlo_tree_search(self.clone(), iterations);
                    self.apply_choice(&choice);
                }
                Status::Terminated(_) => {
                    return;
                }
            }
        }
    }
}
