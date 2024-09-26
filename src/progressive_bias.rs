use crate::Game;
use crate::monte_carlo::{upper_confidence_bound, MonteCarloTreeSearch, MonteCarloTreeNode};

pub struct ProgressiveBiasPolicy<G: Game>
{
    pub heuristic_function: Box<dyn Fn(&G, G::PlayerId) -> f64>,
}

impl<G: Game> MonteCarloTreeSearch for ProgressiveBiasPolicy<G> {
    type Game = G;

    fn get_selection_value(&self, game: &Self::Game, parent: &MonteCarloTreeNode<Self::Game>, child: &MonteCarloTreeNode<Self::Game>) -> f64 {
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

        let mut cloned_game = game.clone();
        cloned_game.apply_choice(child.choice.as_ref().unwrap());

        upper_confidence_bound(cumulative_reward, games, total_game_count, c) + (self.heuristic_function)(&cloned_game, child.player_id) / (child.games + 1.0)
    }
}
