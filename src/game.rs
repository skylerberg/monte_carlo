use core::hash::Hash;

use rand_core::Rng;

/// Per-player payoffs, produced once per iteration at the leaf.
///
/// Implemented for `[f64; N]`, which is what both known consumers use. Backup is
/// max^n: each node accumulates only the payoff of the player who moved there,
/// so nothing here assumes two players or zero sum.
pub trait Rewards: Copy {
    /// This player's payoff.
    fn reward(&self, player: u8) -> f64;

    /// All-zero payoffs, used for the degenerate node with no legal choices.
    ///
    /// `Default` would do, but std only implements it for arrays up to length
    /// 32, and requiring it would cap the player count for no reason.
    fn zero() -> Self;
}

impl<const N: usize> Rewards for [f64; N] {
    #[inline(always)]
    fn reward(&self, player: u8) -> f64 {
        self[player as usize]
    }

    #[inline(always)]
    fn zero() -> Self {
        [0.0; N]
    }
}

/// Whether a state needs a decision, or is finished and worth `Rw`.
///
/// Games fold their own cutoffs into `Terminal` — a round horizon with a
/// heuristic evaluation is reported here exactly like a real game end.
#[derive(Clone, Copy, Debug)]
pub enum Status<Rw> {
    /// `player` must choose.
    Active {
        /// The player to act.
        player: u8,
    },
    /// Finished, or cut off by the game's own horizon.
    Terminal(Rw),
}

/// The interface a game implements to be searched.
///
/// Players are `u8` indices so that rewards can be a plain array. Keep your own
/// player type and convert at the boundary.
pub trait Game: Sized {
    /// An edge in the tree.
    ///
    /// `Eq` drives the linear child scan; `Hash` is used only once a node grows
    /// past [`Game::CHILD_INDEX_THRESHOLD`] children.
    type Choice: Clone + Eq + Hash;

    /// Usually `[f64; N]` for `N` players.
    type Rewards: Rewards;

    /// Immutable data shared by every iteration of one search: evaluation
    /// tables, tuned parameters, precomputed scores, move-filtering flags.
    ///
    /// This exists so such data does not have to live in the game state, where
    /// it would be copied on every determinization. Use `()` if you have none.
    type Context;

    /// Mutable state that belongs to the search but not to the tree, such as a
    /// flat bandit over decisions the tree does not model. Use `()` if unused.
    type Side: Default;

    /// Set when the root's legal choices do not vary across determinizations —
    /// true whenever determinization only reshuffles information hidden *from*
    /// the perspective player, which is the usual case.
    ///
    /// When true the root is enumerated once per search instead of once per
    /// iteration. Verify it for your game: if the root's choice set really can
    /// change, leaving this on silently searches stale moves.
    const ROOT_CHOICES_INVARIANT: bool = true;

    /// Child count above which a node builds a hash index for lookup.
    ///
    /// Below it, `expand` finds children by linear scan, which wins for small
    /// branching factors and cheap `Choice` equality. Above it the scan is
    /// quadratic in the branching factor and loses badly.
    ///
    /// The default is where `benches/child_storage.rs` measures the crossover:
    /// scanning is up to 1.17x faster at 4 children and break-even at 16, while
    /// indexing is 2.4x faster at 64 and 36x faster at 1024. A `Choice` that
    /// owns heap data crosses over sooner — around 8 — because both equality
    /// and hashing chase a pointer, so lower this if yours does.
    const CHILD_INDEX_THRESHOLD: usize = 16;

    // ---- hot path: called at every node of every iteration ----

    /// Whether this state needs a decision, and from whom.
    fn status(&self, ctx: &Self::Context) -> Status<Self::Rewards>;

    /// Fill `out` with the legal choices. `out` is cleared by the caller and
    /// reused across iterations, so this should not allocate in steady state.
    fn choices_into(&self, ctx: &Self::Context, out: &mut Vec<Self::Choice>);

    /// Play `choice`. Resolve any chance events with `rng`.
    fn apply_choice<R: Rng + ?Sized>(
        &mut self,
        ctx: &Self::Context,
        choice: &Self::Choice,
        rng: &mut R,
    );

    /// Play out from here and return the payoffs.
    ///
    /// The whole rollout, not one step: the game owns step granularity, its own
    /// step budget, its cutoff, and the evaluation applied at that cutoff. A
    /// per-move callback would forbid policies that resolve several game steps
    /// at once.
    fn rollout<R: Rng + ?Sized>(&mut self, ctx: &Self::Context, rng: &mut R) -> Self::Rewards;

    // ---- cold path: called once per search, or once per iteration ----

    /// Build the single scratch state that every iteration determinizes into.
    /// Its contents are irrelevant; only its shape and capacity matter.
    fn new_buffer(&self) -> Self;

    /// Overwrite `dest` with a determinization of `self` from `perspective`.
    ///
    /// Writing into a caller-owned buffer (rather than returning `Self`) is what
    /// lets an implementation reuse its inner allocations via `clone_from`. For
    /// a perfect-information game this is `dest.clone_from(self)`.
    fn determinize_into<R: Rng + ?Sized>(
        &self,
        dest: &mut Self,
        ctx: &Self::Context,
        perspective: u8,
        rng: &mut R,
    );

    // ---- optional hooks ----

    /// Fast-forward past decisions the tree does not model, such as opponent
    /// moves resolved by a side model.
    ///
    /// Called once after each determinization and again after every applied
    /// choice during descent.
    fn advance<R: Rng + ?Sized>(
        &mut self,
        _ctx: &Self::Context,
        _side: &mut Self::Side,
        _perspective: u8,
        _rng: &mut R,
    ) {
    }

    /// Reset the side model at the start of a search.
    fn init_side(&self, _ctx: &Self::Context, side: &mut Self::Side) {
        *side = Default::default();
    }

    /// Start of an iteration, before determinization.
    fn begin_iteration(_side: &mut Self::Side) {}

    /// End of an iteration, after backpropagation. Credit anything the side
    /// model recorded during the descent.
    fn credit_iteration(_side: &mut Self::Side, _rewards: &Self::Rewards) {}

    /// Prior for progressive bias, evaluated once when a child is first visited.
    /// Only called when `Config::progressive_bias_weight` is non-zero.
    fn heuristic_bias(&self, _ctx: &Self::Context, _perspective: u8) -> f32 {
        0.0
    }
}
