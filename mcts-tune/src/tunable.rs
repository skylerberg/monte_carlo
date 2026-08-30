/// A parameter struct the optimizers can search over.
///
/// The gene vector is the optimizer's whole view of a candidate, so what a game
/// leaves out of it is pinned for the run. Leaving a parameter out is the fix
/// for a *redundant* one: an evaluation of the form `tanh(weighted_sum /
/// scale)` is unchanged when every weight and `scale` are multiplied by the
/// same constant, and an optimizer handed both will wander along that flat
/// direction spending games on candidates it cannot tell apart. Pin one end of
/// any such family before tuning.
pub trait Tunable: Clone + Send + Sync {
    /// Gene names, in gene order. Its length is the dimension searched, and it
    /// is what run output is labelled with.
    fn gene_names() -> &'static [&'static str];

    /// This candidate as a gene vector, of length `gene_names().len()`.
    fn to_genes(&self) -> Vec<f64>;

    /// These parameters with `genes` applied. Every field the gene vector
    /// does not carry keeps `self`'s value, which is how a pinned parameter
    /// stays pinned.
    fn with_genes(&self, genes: &[f64]) -> Self;

    /// Force a gene vector back into the set the game can represent — clamping
    /// a probability to `0..=1`, rounding a count to an integer, holding a
    /// weight non-negative.
    ///
    /// Called on every candidate before it is evaluated, so a repair here is
    /// the only thing standing between the optimizer's unconstrained proposals
    /// and [`Tunable::with_genes`].
    ///
    /// Clamping to exactly zero is safe here. Mutation is additive and
    /// [`Tunable::gene_scales`] gives a zeroed gene a step borrowed from its
    /// neighbours, so zero is a value the search can leave again rather than one
    /// it falls into. That is not true of every optimizer — under multiplicative
    /// mutation zero is absorbing — so it is a property of this crate worth
    /// knowing rather than one to rely on generally.
    fn repair(genes: &mut [f64]) {
        let _ = genes;
    }

    /// The natural size of a meaningful change to each gene, in gene order.
    ///
    /// Sets the initial step size per coordinate, so it is what lets a run mix
    /// a weight around `12.0` with a probability around `0.05` without the
    /// small one being ignored.
    ///
    /// The default is each gene's own magnitude, and for a gene seeded at zero
    /// — which has no magnitude to take a step from — the median of the others.
    /// A hardcoded floor is the tempting answer there and the wrong one: it is
    /// only right for parameters that happen to be near it, and where the rest
    /// of the vector sits around `5.0` a floor of `1e-3` leaves that gene taking
    /// steps ten thousand times too small to get anywhere. Additive mutation
    /// means such a gene can move in principle; a step that size means it never
    /// does in practice, and the run reports it as a weight the search chose to
    /// leave alone. Borrowing the typical magnitude keeps the rule scale-free:
    /// it reads the same whether the weights are thousandths or thousands.
    fn gene_scales(&self) -> Vec<f64> {
        let magnitudes: Vec<f64> = self.to_genes().iter().map(|gene| gene.abs()).collect();

        let mut known: Vec<f64> = magnitudes.iter().copied().filter(|m| *m > 1e-12).collect();
        known.sort_by(|a, b| a.partial_cmp(b).expect("a magnitude is never NaN"));
        // Median rather than mean: one weight an order of magnitude above the
        // rest should not drag every zeroed gene up with it. All-zero
        // parameters have nothing to borrow from, so they fall back to one.
        let typical = known.get(known.len() / 2).copied().unwrap_or(1.0);

        magnitudes
            .into_iter()
            .map(|magnitude| {
                if magnitude > 1e-12 {
                    magnitude
                } else {
                    typical
                }
            })
            .collect()
    }
}
