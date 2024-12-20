use std::fmt::Debug;
use std::marker::PhantomData;

use float_ord::FloatOrd;
use rolling_stats::Stats;
use rustc_hash::FxHashMap;

use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::stats::MctsStats;
use crate::Game;

pub trait MonteCarloTreeSearch {
    type Game: Game;

    fn monte_carlo_tree_search(
        &mut self,
        game: Self::Game,
        iterations: usize,
    ) -> (<Self::Game as Game>::Choice, MctsStats) {
        let tree = self.build_tree(game, iterations);

        let selected_child = tree
            .children
            .values()
            .max_by_key(|child| FloatOrd(child.games))
            .unwrap();

        let mut stats: Stats<f64> = Stats::new();
        tree.children
            .values()
            .for_each(|child| stats.update(child.cumulative_reward / child.games));

        println!("All child Win %: {}", stats);
        println!(
            "Selected child: (games: {}, wins: {}, win_rate: {:2})",
            selected_child.games,
            selected_child.cumulative_reward,
            selected_child.cumulative_reward / selected_child.games
        );

        (
            selected_child.choice.clone().unwrap(),
            MctsStats {
                tree_cumulative_reward: tree.cumulative_reward,
                tree_games: tree.games,
            },
        )
    }

    fn build_tree(
        &mut self,
        game: Self::Game,
        iterations: usize,
    ) -> MonteCarloTreeNode<Self::Game>  {
        let player_id = game.get_active_player_id();
        let mut tree: MonteCarloTreeNode<Self::Game> = MonteCarloTreeNode::new(player_id, None);

        for _ in 0..iterations {
            let determinization = game.get_determinization(game.get_active_player_id());
            let game = self.iteration(&mut tree, determinization);
            self.after_iteration(&game);
        }
        tree
    }

    fn after_iteration(&mut self, _game: &Self::Game) {}

    // Returns the winner of the iteration
    fn iteration(
        &mut self,
        node: &mut MonteCarloTreeNode<Self::Game>,
        mut game: Self::Game,
    ) -> Self::Game {
        if game.is_terminal() {
            self.record_outcome(node, &game);
            return game;
        }

        let choices = node.expand(&game, game.shuffle_on_expand());

        let best_child = self.select(node, &game, choices);
        self.after_selection(&game, best_child);
        game.apply_choice(best_child.choice.as_ref().unwrap());

        let game = if best_child.games == 0.0 {
            //println!("Rolling out {}", best_child.id);
            let game = self.rollout(best_child, game);
            self.record_outcome(best_child, &game);
            game
        } else {
            //println!("Recursing from {} to {}", node_id, best_child.id);
            self.iteration(best_child, game)
        };
        //println!("Recording at {} after handling {}", node_id, best_child.id);
        self.record_outcome(node, &game);
        return game;
    }

    fn after_selection(&mut self, _game: &Self::Game, _selected: &MonteCarloTreeNode<Self::Game>) {}

    fn get_first_play_value(
        &self,
        _game: &Self::Game,
        _parent: &MonteCarloTreeNode<Self::Game>,
        _child: &MonteCarloTreeNode<Self::Game>,
        _choices: &Option<Vec<<Self::Game as Game>::Choice>>,
    ) -> f64 {
        f64::MAX
    }

    fn get_selection_value(
        &self,
        _game: &Self::Game,
        parent: &MonteCarloTreeNode<Self::Game>,
        child: &MonteCarloTreeNode<Self::Game>,
    ) -> f64 {
        let c = 0.4;
        let cumulative_reward = child.cumulative_reward;
        let games = child.games;
        let total_game_count = if parent.is_root() {
            // The root is always fully expanded and the availability of nodes does not change
            parent.games
        } else {
            *parent
                .choice_availability_count
                .get(child.choice.as_ref().unwrap())
                .unwrap() as f64
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
        choices: Option<Vec<<Self::Game as Game>::Choice>>,
    ) -> &'a mut MonteCarloTreeNode<Self::Game> {
        let selected_choice = node
            .children
            .iter()
            .filter(|(choice, _)| game.choice_is_available(choice))
            .max_by_key(|(_, child)| {
                // TODO make this short-circuit if we find a child with an infinite value (e.g., a child not yet explored)
                FloatOrd(if child.games == 0.0 {
                    self.get_first_play_value(game, node, child, &choices)
                } else {
                    self.get_selection_value(game, node, child)
                })
            })
            .map(|(choice, _)| choice)
            .unwrap()
            .clone();
        node.children.get_mut(&selected_choice).unwrap()
    }

    // Returns the winner of the simulation
    fn rollout(
        &mut self,
        node: &mut MonteCarloTreeNode<Self::Game>,
        mut game: Self::Game,
    ) -> Self::Game {
        while !game.is_terminal() && !game.heuristic_early_terminate() {
            let choice = game.get_rollout_choice();
            let choice = self.intercept_rollout_choice(node, &mut game, choice);
            game.apply_choice(&choice);
        }

        return game;
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
    fn record_outcome(
        &mut self,
        node: &mut MonteCarloTreeNode<Self::Game>,
        game: &Self::Game,
    ) {
        node.cumulative_reward += game.reward_for(node.player_id);
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

impl<G> MonteCarloTreeNode<G>
where
    G: Game,
{
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
    fn expand(&mut self, game: &G, shuffle: bool) -> Option<Vec<<G as Game>::Choice>> {
        if self.is_root() && !self.children.is_empty() {
            return None;
        }
        let active_player = game.get_active_player_id();
        let mut choices = game.get_all_choices();
        let mut added_new_node = false;

        if shuffle {
            let mut rng = thread_rng();
            choices.shuffle(&mut rng);
        }

        for choice in &choices {
            if let Some(count) = self.choice_availability_count.get_mut(&choice) {
                *count += 1;
            } else {
                self.choice_availability_count.insert(choice.clone(), 0);
            }
            if self.is_root() || (!added_new_node && !self.children.contains_key(&choice)) {
                self.children.insert(
                    choice.clone(),
                    MonteCarloTreeNode::new(active_player, Some(choice.clone())),
                );
                added_new_node = true;
            }
        }
        Some(choices)
    }

    pub fn is_root(&self) -> bool {
        self.choice.is_none()
    }
}

pub fn upper_confidence_bound(
    cumulative_reward: f64,
    games: f64,
    total_game_count: f64,
    c: f64,
) -> f64 {
    //let c = 2.0_f64.sqrt();
    let win_rate = cumulative_reward / games;
    win_rate + c * f64::sqrt(f64::ln(total_game_count) / games)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    pub struct ThreeBranchThreeDepthAlwaysWin {
        turn_number: usize,
    }

    impl ThreeBranchThreeDepthAlwaysWin {
        pub fn new() -> Self {
            ThreeBranchThreeDepthAlwaysWin {
                turn_number: 0,
            }
        }
    }

    impl Game for ThreeBranchThreeDepthAlwaysWin {
        type Choice = usize;

        type PlayerId = usize;

        fn get_all_choices(&self) -> Vec<Self::Choice> {
            return vec![1, 2, 3];
        }

        fn apply_choice(&mut self, _choice: &Self::Choice) {
            self.turn_number += 1;
        }

        fn get_active_player_id(&self) -> Self::PlayerId {
            return 1;
        }

        fn is_terminal(&self) -> bool {
            self.turn_number >= 3
        }

        fn reward_for(&self, _player_id: Self::PlayerId) -> f64 {
            1.0
        }
    }

    #[test]
    fn test_explores_each_option_once() {
        let game = ThreeBranchThreeDepthAlwaysWin::new();
        let mut mcts: VanillaMcts<ThreeBranchThreeDepthAlwaysWin> = VanillaMcts::new();
        let tree = mcts.build_tree(game.clone(), 3);
        assert_eq!(tree.children.len(), 3);
        assert!(tree.children.iter().all(|(_, child)| child.games == 1.0));
        assert!(tree.children.iter().all(|(_, child)| child.cumulative_reward == 1.0));
    }

    #[test]
    fn test_even_exploration() {
        let game = ThreeBranchThreeDepthAlwaysWin::new();
        let mut mcts: VanillaMcts<ThreeBranchThreeDepthAlwaysWin> = VanillaMcts::new();
        let tree = mcts.build_tree(game.clone(), 3 * 3 * 3 * 3);
        assert_eq!(tree.children.len(), 3);
        assert!(tree.children.iter().all(|(_, child)| child.games == 3.0 * 3.0 * 3.0));
        assert!(tree.children.iter().all(|(_, child)| child.cumulative_reward == 3.0 * 3.0 * 3.0));
    }

    #[derive(Clone)]
    pub struct BinaryTreeDepthThreeZeroWins {
        turn_number: usize,
        points: usize,
    }

    impl BinaryTreeDepthThreeZeroWins {
        pub fn new() -> Self {
            BinaryTreeDepthThreeZeroWins {
                turn_number: 0,
                points: 0,
            }
        }
    }

    impl Game for BinaryTreeDepthThreeZeroWins {
        type Choice = usize;

        type PlayerId = usize;

        fn get_all_choices(&self) -> Vec<Self::Choice> {
            return vec![0, 1];
        }

        fn apply_choice(&mut self, choice: &Self::Choice) {
            println!("{}", choice);
            self.turn_number += 1;
            if *choice == 1 {
                self.points += 1;
            }
        }

        fn get_active_player_id(&self) -> Self::PlayerId {
            return 1;
        }

        fn is_terminal(&self) -> bool {
            self.turn_number >= 3
        }

        fn reward_for(&self, _player_id: Self::PlayerId) -> f64 {
            if self.points == 3 { 1.0 } else { 0.0 }
        }

        fn shuffle_on_expand(&self) -> bool {
            false
        }
    }

    #[test]
    fn finds_best() {
        let game = BinaryTreeDepthThreeZeroWins::new();
        let mut mcts: VanillaMcts<BinaryTreeDepthThreeZeroWins> = VanillaMcts::new();
        let (choice, _) = mcts.monte_carlo_tree_search(game.clone(), 8);
        assert_eq!(choice, 1);
    }

    #[derive(Clone)]
    pub struct GameNode {
        children: Vec<GameNode>,
        winner: Option<usize>,
    }

    impl GameNode {
        pub fn new(children: Vec<GameNode>, winner: Option<usize>) -> Self {
            GameNode {
                children,
                winner,
            }
        }

        pub fn you_choose(children: Vec<GameNode>) -> Self {
            GameNode {
                children,
                winner: None,
            }
        }

        pub fn they_choose(children: Vec<GameNode>) -> Self {
            GameNode {
                children,
                winner: None,
            }
        }

        pub fn winner(player_id: usize) -> Self {
            GameNode {
                children: vec![],
                winner: Some(player_id),
            }
        }
    }

    #[derive(Clone)]
    pub struct CustomGameTree {
        state: GameNode,
        active_player: usize,
        player_count: usize,
    }

    impl CustomGameTree {
        pub fn minimal_trap() -> Self {
            let you = 0;
            let them = 1;
            CustomGameTree {
                state: GameNode::you_choose(vec![
                    // If you chose this node it seems like you have 2/3 win chance, but you always lose
                    GameNode::they_choose(vec![
                        GameNode::winner(you),
                        GameNode::winner(you),
                        GameNode::winner(them),
                    ]),
                    // If you chose this node it seems like you have 1/3 win chance, but you always win
                    GameNode::they_choose(vec![
                        GameNode::you_choose(vec![
                            GameNode::winner(you),
                            GameNode::winner(them),
                            GameNode::winner(them),
                        ]),
                        GameNode::you_choose(vec![
                            GameNode::winner(you),
                            GameNode::winner(them),
                            GameNode::winner(them),
                        ]),
                        GameNode::you_choose(vec![
                            GameNode::winner(you),
                            GameNode::winner(them),
                            GameNode::winner(them),
                        ]),
                    ]),
                ]),
                active_player: you,
                player_count: 2,
            }
        }
    }

    impl Game for CustomGameTree {
        type Choice = usize;

        type PlayerId = usize;

        fn get_all_choices(&self) -> Vec<Self::Choice> {
            (0..self.state.children.len()).collect()
        }

        fn apply_choice(&mut self, choice: &Self::Choice) {
            self.state = self.state.children.remove(*choice);
            self.active_player = (self.active_player + 1) % self.player_count;
        }

        fn get_active_player_id(&self) -> Self::PlayerId {
            return self.active_player;
        }

        fn is_terminal(&self) -> bool {
            self.state.winner.is_some()
        }

        fn reward_for(&self, player_id: Self::PlayerId) -> f64 {
            if self.state.winner.unwrap() == player_id { 1.0 } else { 0.0 }
        }

        fn shuffle_on_expand(&self) -> bool {
            false
        }
    }

    #[test]
    fn defeats_trap() {
        let game = CustomGameTree::minimal_trap();
        let mut mcts: VanillaMcts<CustomGameTree> = VanillaMcts::new();
        let (choice, _) = mcts.monte_carlo_tree_search(game.clone(), 40);
        assert_eq!(choice, 1);
    }
}
