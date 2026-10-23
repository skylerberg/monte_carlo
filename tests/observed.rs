//! The observed searcher: one tree per player, keyed by what each can see.
//!
//! The fixture is [`common::Bluff`], a game whose only secrecy is what a move
//! conceals. See its module docs for the payoff table and the two equilibrium
//! values: 0.4667 when the opponent can see through the concealment, 0.5556
//! when it cannot — and the mixed equilibrium in which seat 0 conceals its weak
//! card about a third of the time.

// `common` holds every shared fixture; this binary uses one of them.
#[allow(dead_code)]
mod common;

use common::{Bluff, Key, Move};
use mcts::rand_core::{Rng, SeedableRng};
use mcts::{Config, Game, Observed, ObservedSearcher, Searcher, Status};
use wyrand::WyRand;

fn config(iterations: u32) -> Config {
    Config {
        iterations,
        early_termination: false,
        exploration_constant: 0.5,
        ..Config::default()
    }
}

fn rng(seed: u64) -> WyRand {
    WyRand::seed_from_u64(seed)
}

/// `(visits, share of the node's visits)` for each of a node's children, by the
/// answer they lead to.
fn answers<K: std::fmt::Debug>(node: &mcts::ObservedNode<K>) -> Vec<(u32, f64)> {
    let total = node.visits().max(1) as f64;
    node.children()
        .iter()
        .map(|child| (child.visits(), child.visits() as f64 / total))
        .collect()
}

#[test]
fn a_clairvoyant_opponent_answers_each_concealment_differently() {
    let state = Bluff::new();
    let mut plain = Searcher::new(&state);
    plain.search(&state, &(), 0, &config(4_000), None, &mut rng(3));

    let root = plain.tree().expect("a search leaves a tree");
    assert_eq!(
        root.children().len(),
        4,
        "every move is its own edge in a Choice-keyed tree"
    );

    // The opponent's node under each concealment is a different node, and each
    // one settles on the answer that punishes the card it can see: call the
    // weak card, pass against the strong one.
    let dominated = |concealed: usize, expected: Move| {
        let child = root
            .children()
            .iter()
            .find(|child| child.edge().choice() == Some(&Move::Hide(concealed)))
            .expect("the root offers concealing either card");
        let leader = child
            .children()
            .iter()
            .map(|answer| {
                (
                    *answer.edge().choice().expect("an answer is a choice"),
                    answer.visits(),
                )
            })
            .max_by_key(|&(_, visits)| visits)
            .expect("the search descended into the opponent's decision");
        assert_eq!(leader.0, expected, "concealing card {concealed}");
    };
    dominated(0, Move::Call);
    dominated(1, Move::Pass);
}

#[test]
fn the_opponent_decides_once_inside_one_information_set() {
    let state = Bluff::new();
    let mut observed = ObservedSearcher::new(&state, 2);
    observed.search(&state, &(), 0, &config(4_000), None, &mut rng(3));

    // The concealing player knows what it concealed, so its own tree keeps the
    // four moves apart.
    let own = observed.tree(0).expect("a search leaves a tree per player");
    assert_eq!(own.children().len(), 4);

    // The opponent sees "a card, face down" and nothing else: one node for both
    // concealments, and the two revelations stay apart because revealing is
    // public.
    let opponent = observed.tree(1).expect("a search leaves a tree per player");
    assert_eq!(opponent.children().len(), 3);
    let hidden = opponent
        .children()
        .iter()
        .find(|child| child.observed_key() == Some(&Key::Hidden))
        .expect("both concealments share one edge in the opponent's tree");

    // And from inside it, the answers are a genuine guess: both are played,
    // because the equilibrium leaves it indifferent.
    let mut keys: Vec<Key> = hidden
        .children()
        .iter()
        .filter_map(|child| child.observed_key().copied())
        .collect();
    keys.sort_by_key(|key| format!("{key:?}"));
    assert_eq!(keys, vec![Key::Exact(Move::Call), Key::Exact(Move::Pass)]);
    for (visits, share) in answers(hidden) {
        assert!(
            share > 0.25,
            "the merged node played one answer {visits} times and the other barely at all, so it \
             is not weighting its belief: {:?}",
            answers(hidden)
        );
    }
}

#[test]
fn seeing_through_the_concealment_costs_seat_zero_the_bluff() {
    let state = Bluff::new();
    let cfg = config(4_000);

    let mut plain = Searcher::new(&state);
    let plain_result = plain.search(&state, &(), 0, &cfg, None, &mut rng(3));
    let mut observed = ObservedSearcher::new(&state, 2);
    let observed_result = observed.search(&state, &(), 0, &cfg, None, &mut rng(3));

    // A searcher that lets the opponent tell the concealments apart caps seat 0
    // at `conceal the strong card` = 0.4667 (with `reveal both` at 0.4).
    assert!(
        plain_result.best_mean_reward < 0.50,
        "the clairvoyant opponent leaves seat 0 {:.4}",
        plain_result.best_mean_reward
    );
    // One information set makes the opponent guess, and the mixed equilibrium
    // is worth 0.5556. The gap is the bluff.
    assert!(
        observed_result.best_mean_reward > 0.52,
        "seat 0 is worth {:.4} against an opponent that cannot tell",
        observed_result.best_mean_reward
    );

    // And the mixed searcher plays it: concealing the weak card is an
    // equilibrium action taken about a third of the time, where the clairvoyant
    // one is barely worth exploring.
    let share = |searcher: &ObservedSearcher<Bluff>, plain: Option<&Searcher<Bluff>>| {
        let total = plain
            .and_then(Searcher::tree)
            .map(|root| root.visits())
            .unwrap_or_else(|| searcher.tree(0).expect("a tree").visits());
        let visits = match plain {
            Some(plain) => plain
                .tree()
                .expect("a tree")
                .children()
                .iter()
                .find(|child| child.edge().choice() == Some(&Move::Hide(0)))
                .map(|child| child.visits())
                .unwrap_or(0),
            None => searcher
                .tree(0)
                .expect("a tree")
                .children()
                .iter()
                .find(|child| child.observed_key() == Some(&Key::Exact(Move::Hide(0))))
                .map(|child| child.visits())
                .unwrap_or(0),
        };
        visits as f64 / total.max(1) as f64
    };
    assert!(
        share(&observed, None) > 0.20,
        "the merged searcher spent {:.3} of its budget on the bluff",
        share(&observed, None)
    );
    assert!(
        share(&observed, Some(&plain)) < 0.10,
        "the clairvoyant searcher spent {:.3} of its budget on a bluff that cannot work",
        share(&observed, Some(&plain))
    );
}

#[test]
fn re_rooting_keeps_every_observers_tree_and_a_miss_drops_them() {
    let state = Bluff::new();
    let mut searcher = ObservedSearcher::new(&state, 2);
    let result = searcher.search(&state, &(), 0, &config(300), None, &mut rng(1));

    assert!(
        searcher.reuse_subtree(&state, &(), &result.choice),
        "the answered move is a root edge in every observer's tree"
    );
    assert!(searcher.tree(0).expect("a tree").visits() > 0);

    // A position with no decision to search has no edge to re-root at: the
    // trees are dropped rather than re-rooted one by one.
    let mut finished = state.clone();
    finished.apply_choice(&(), &Move::Reveal(0), &mut rng(0));
    assert!(!searcher.reuse_subtree(&finished, &(), &Move::Reveal(0)));
    assert_eq!(searcher.tree(0).expect("a tree").visits(), 0);
}

#[test]
fn one_legal_move_is_the_answer_without_a_search() {
    #[derive(Clone)]
    struct ForcedState;

    impl Game for ForcedState {
        type Choice = ();
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        fn status(&self, _: &()) -> Status<[f64; 2]> {
            Status::Active { player: 0 }
        }

        fn choices_into(&self, _: &(), out: &mut Vec<()>) {
            out.push(());
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &(), _: &mut R) {}

        fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
            [0.5, 0.5]
        }

        fn new_buffer(&self) -> Self {
            ForcedState
        }

        fn determinize_into<R: Rng + ?Sized>(&self, _: &mut Self, _: &(), _: u8, _: &mut R) {}
    }

    impl Observed for ForcedState {
        type Key = ();

        fn observed_key(&self, _: &(), _: &(), _: u8, _: u8) {}
    }

    let mut searcher = ObservedSearcher::new(&ForcedState, 1);
    let result = searcher.search(&ForcedState, &(), 0, &config(10), None, &mut rng(0));
    assert_eq!(result.choice, ());
    assert_eq!(result.iterations_used, 0);
    assert_eq!(result.stop_reason, mcts::StopReason::SingleChoice);
}

/// A key that reads state the observer cannot see is the failure this searcher
/// exists to prevent, and a debug build refuses it.
///
/// The game deals one card to each seat and nothing else. `determinize_into`
/// from a seat's perspective resamples the *other* seat's card — the same
/// information a leaky key would be reading. The check itself is compiled out
/// of release builds, so this is gated to match.
#[cfg(debug_assertions)]
mod leaky {
    use super::*;

    #[derive(Clone)]
    pub struct Leaky {
        cards: [u8; 2],
        to_move: u8,
    }

    impl Leaky {
        pub fn new() -> Self {
            Self {
                cards: [0, 1],
                to_move: 0,
            }
        }

        fn new_buffer() -> Self {
            Self {
                cards: [0, 1],
                to_move: 0,
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub enum Leak {
        Pass,
        Hold,
        /// Reads the card of the player who is *not* looking.
        Peek(u8),
    }

    impl Game for Leaky {
        type Choice = Leak;
        type Rewards = [f64; 2];
        type Context = ();
        type Side = ();

        fn status(&self, _: &()) -> Status<[f64; 2]> {
            if self.to_move >= 2 {
                Status::Terminal([0.5, 0.5])
            } else {
                Status::Active {
                    player: self.to_move,
                }
            }
        }

        fn choices_into(&self, _: &(), out: &mut Vec<Leak>) {
            out.extend([Leak::Pass, Leak::Hold]);
        }

        fn apply_choice<R: Rng + ?Sized>(&mut self, _: &(), _: &Leak, _: &mut R) {
            self.to_move += 1;
        }

        fn rollout<R: Rng + ?Sized>(&mut self, _: &(), _: &mut R) -> [f64; 2] {
            self.to_move = 2;
            [0.5, 0.5]
        }

        fn new_buffer(&self) -> Self {
            Self::new_buffer()
        }

        fn determinize_into<R: Rng + ?Sized>(
            &self,
            dest: &mut Self,
            _: &(),
            perspective: u8,
            _: &mut R,
        ) {
            dest.cards = self.cards;
            dest.to_move = self.to_move;
            // The observer keeps its own card and learns nothing else, so a
            // world consistent with what it knows may deal the other seat either
            // of the two values it has not seen.
            let hidden = 1 - perspective;
            dest.cards[hidden as usize] = (self.cards[perspective as usize] + 1) % 3;
        }
    }

    impl Observed for Leaky {
        type Key = Leak;

        fn observed_key(&self, _: &(), choice: &Leak, actor: u8, observer: u8) -> Leak {
            if actor == observer {
                return *choice;
            }
            Leak::Peek(self.cards[1 - observer as usize])
        }
    }

    #[test]
    #[should_panic(expected = "not a function of what player")]
    fn a_key_that_reads_hidden_state_is_caught() {
        let game = Leaky::new();
        let mut searcher = ObservedSearcher::new(&game, 2);
        searcher.search(&game, &(), 0, &config(20), None, &mut rng(0));
    }
}
