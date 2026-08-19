//! Small fixture games shared by the integration tests.

use mcts::rand_core::Rng;
use mcts::{Game, Status};

/// Three choices, three plies deep, everybody always wins. Used to check that
/// exploration is even when nothing distinguishes the choices.
#[derive(Clone)]
pub struct AlwaysWin {
    pub ply: u32,
}

impl Game for AlwaysWin {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.ply >= 3 {
            Status::Terminal([1.0, 1.0])
        } else {
            Status::Active { player: 1 }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend([1, 2, 3]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &usize, _: &mut R) {
        self.ply += 1;
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
        self.ply = 3;
        [1.0, 1.0]
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// Binary choices, three plies deep. Only the all-ones line wins.
#[derive(Clone)]
pub struct CountToThree {
    pub ply: u32,
    pub points: u32,
}

impl CountToThree {
    pub fn new() -> Self {
        Self { ply: 0, points: 0 }
    }

    fn rewards(&self) -> [f64; 2] {
        if self.points == 3 {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        }
    }
}

impl Game for CountToThree {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if self.ply >= 3 {
            Status::Terminal(self.rewards())
        } else {
            Status::Active { player: 0 }
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend([0, 1]);
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, _: &mut R) {
        self.ply += 1;
        if *choice == 1 {
            self.points += 1;
        }
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        while self.ply < 3 {
            let choice = (rng.next_u64() & 1) as usize;
            self.apply_choice(&(), &choice, rng);
        }
        self.rewards()
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}

/// An explicit game tree, so a test can build a position with a known answer.
#[derive(Clone)]
pub struct TreeNode {
    pub children: Vec<TreeNode>,
    pub winner: Option<u8>,
}

impl TreeNode {
    pub fn branch(children: Vec<TreeNode>) -> Self {
        Self {
            children,
            winner: None,
        }
    }

    pub fn winner(player: u8) -> Self {
        Self {
            children: Vec::new(),
            winner: Some(player),
        }
    }
}

#[derive(Clone)]
pub struct GameTree {
    pub state: TreeNode,
    pub active_player: u8,
    pub player_count: u8,
}

impl GameTree {
    /// The first branch looks like a 2/3 win but is always lost; the second
    /// looks like 1/3 but is always won. A search that trusts the shallow
    /// average picks 0; a search that reads the tree picks 1.
    pub fn minimal_trap() -> Self {
        let you = 0;
        let them = 1;
        let losing_line = || {
            TreeNode::branch(vec![
                TreeNode::winner(you),
                TreeNode::winner(them),
                TreeNode::winner(them),
            ])
        };
        Self {
            state: TreeNode::branch(vec![
                TreeNode::branch(vec![
                    TreeNode::winner(you),
                    TreeNode::winner(you),
                    TreeNode::winner(them),
                ]),
                TreeNode::branch(vec![losing_line(), losing_line(), losing_line()]),
            ]),
            active_player: you,
            player_count: 2,
        }
    }

    pub fn apply(&mut self, choice: &usize) {
        self.state = self.state.children.swap_remove(*choice);
        self.active_player = (self.active_player + 1) % self.player_count;
    }

    /// Two plies of `width` choices each, so a subtree re-rooted after the
    /// first move still faces a wide choice set at the second.
    pub fn wide_two_ply(width: usize) -> Self {
        let second_ply = || {
            let mut leaves: Vec<TreeNode> = (0..width - 1).map(|_| TreeNode::winner(1)).collect();
            leaves.push(TreeNode::winner(0));
            TreeNode::branch(leaves)
        };
        Self {
            state: TreeNode::branch((0..width).map(|_| second_ply()).collect()),
            active_player: 0,
            player_count: 2,
        }
    }

    /// A root with `width` choices, each leading straight to a terminal node.
    /// Choice `width - 1` is the only win, so the search has to open every
    /// child — which is what exercises wide-node child lookup.
    pub fn wide(width: usize) -> Self {
        let mut children: Vec<TreeNode> = (0..width - 1).map(|_| TreeNode::winner(1)).collect();
        children.push(TreeNode::winner(0));
        Self {
            state: TreeNode::branch(children),
            active_player: 0,
            player_count: 2,
        }
    }
}

impl Game for GameTree {
    type Choice = usize;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    // Exercised by the wide test; the default is 32.
    const CHILD_INDEX_THRESHOLD: usize = 8;

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        match self.state.winner {
            Some(winner) => {
                let mut rewards = [0.0; 2];
                rewards[winner as usize] = 1.0;
                Status::Terminal(rewards)
            }
            None => Status::Active {
                player: self.active_player,
            },
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<usize>) {
        out.extend(0..self.state.children.len());
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &usize, _: &mut R) {
        self.apply(choice);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        loop {
            match self.status(&()) {
                Status::Terminal(rewards) => return rewards,
                Status::Active { .. } => {
                    let count = self.state.children.len() as u64;
                    let choice = ((rng.next_u64() as u128 * count as u128) >> 64) as usize;
                    self.apply(&choice);
                }
            }
        }
    }

    fn new_buffer(&self) -> Self {
        self.clone()
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        dest.clone_from(self);
    }
}
