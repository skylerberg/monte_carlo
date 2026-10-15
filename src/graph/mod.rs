//! Search over a graph of positions, for games that implement
//! [`crate::Transposition`].
//!
//! A separate searcher rather than a mode of [`crate::Searcher`], so that a game
//! which never merges positions pays nothing for the ability: the tree's node,
//! descent and backup are untouched, and none of what a graph needs — links
//! with their own statistics, a merge table, cycle stamps — is in them.

mod node;
mod search;
mod view;

pub use node::Graph;
pub use search::GraphSearcher;
pub use view::{GraphEdge, GraphNode};

#[cfg(feature = "parallel")]
pub(crate) use search::link_stats;
