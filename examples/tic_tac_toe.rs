//! A complete `Game` implementation, start to finish.
//!
//! Tic-tac-toe is perfect information and deterministic, so the hooks for
//! hidden information and side models stay at their defaults: `Context` and
//! `Side` are `()`, and `determinize_into` is a plain copy.
//!
//! Run with `cargo run --release --example tic_tac_toe`.

use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, Searcher, Status};
use wyrand::WyRand;

const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

#[derive(Clone, Copy)]
struct TicTacToe {
    /// 0 empty, 1 for player 0, 2 for player 1.
    cells: [u8; 9],
    to_move: u8,
}

impl TicTacToe {
    fn new() -> Self {
        Self {
            cells: [0; 9],
            to_move: 0,
        }
    }

    fn winner(&self) -> Option<u8> {
        LINES.iter().find_map(|line| {
            let first = self.cells[line[0]];
            (first != 0 && line.iter().all(|&i| self.cells[i] == first)).then_some(first - 1)
        })
    }

    fn play(&mut self, cell: u8) {
        self.cells[cell as usize] = self.to_move + 1;
        self.to_move ^= 1;
    }
}

impl Game for TicTacToe {
    type Choice = u8;
    type Rewards = [f64; 2];
    type Context = ();
    type Side = ();

    fn status(&self, _: &()) -> Status<[f64; 2]> {
        if let Some(winner) = self.winner() {
            let mut rewards = [0.0; 2];
            rewards[winner as usize] = 1.0;
            return Status::Terminal(rewards);
        }
        if self.cells.iter().all(|&c| c != 0) {
            return Status::Terminal([0.5, 0.5]);
        }
        Status::Active {
            player: self.to_move,
        }
    }

    fn choices_into(&self, _: &(), out: &mut Vec<u8>) {
        out.extend(
            self.cells
                .iter()
                .enumerate()
                .filter(|(_, &c)| c == 0)
                .map(|(i, _)| i as u8),
        );
    }

    fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), choice: &u8, _: &mut R) {
        self.play(*choice);
    }

    fn rollout<R: Rng + ?Sized>(&mut self, _: &(), rng: &mut R) -> [f64; 2] {
        let mut empty = [0u8; 9];
        loop {
            match self.status(&()) {
                Status::Terminal(rewards) => return rewards,
                Status::Simultaneous { .. } => unreachable!(),
                Status::Active { .. } => {
                    let mut count = 0;
                    for (i, &cell) in self.cells.iter().enumerate() {
                        if cell == 0 {
                            empty[count] = i as u8;
                            count += 1;
                        }
                    }
                    let pick = ((rng.next_u64() as u128 * count as u128) >> 64) as usize;
                    self.play(empty[pick]);
                }
            }
        }
    }

    fn new_buffer(&self) -> Self {
        *self
    }

    fn determinize_into<R: Rng + ?Sized>(&self, dest: &mut Self, _: &(), _: u8, _: &mut R) {
        *dest = *self;
    }
}

fn render(game: &TicTacToe) -> String {
    game.cells
        .chunks(3)
        .map(|row| {
            row.iter()
                .map(|&c| match c {
                    1 => "X",
                    2 => "O",
                    _ => ".",
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    let config = Config {
        iterations: 20_000,
        exploration_constant: 1.4,
        ..Config::default()
    };

    let mut game = TicTacToe::new();
    let mut rng = WyRand::seed_from_u64(0xC0FFEE);

    // One searcher per player, each keeping its own tree between moves.
    let mut searchers = [Searcher::new(&game), Searcher::new(&game)];

    while let Status::Active { player } = game.status(&()) {
        let result = searchers[player as usize].search(&game, &(), player, &config, None, &mut rng);

        println!(
            "player {player} plays {} after {} iterations ({} reused, {:?}), win rate {:.3}",
            result.choice,
            result.iterations_used,
            result.reused_iterations,
            result.stop_reason,
            result.best_mean_reward,
        );

        game.play(result.choice);
        for searcher in &mut searchers {
            searcher.reuse_subtree(&result.choice);
        }
        println!("{}\n", render(&game));
    }

    match game.winner() {
        Some(winner) => println!("player {winner} wins"),
        None => println!("draw, as it should be"),
    }
}
