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
    /// Prefer clamping to a small positive floor over clamping to exactly zero
    /// for anything a mutation scales multiplicatively: zero is absorbing under
    /// multiplication, so a gene that reaches it can never move again.
    fn repair(genes: &mut [f64]) {
        let _ = genes;
    }

    /// The natural size of a meaningful change to each gene, in gene order.
    ///
    /// Sets the initial step size per coordinate, so it is what lets a run mix
    /// a weight around `12.0` with a probability around `0.05` without the
    /// small one being ignored. The default — each gene's own magnitude, with a
    /// floor so a gene seeded at zero can still move — is reasonable when every
    /// parameter is a weight on the same scale.
    fn gene_scales(&self) -> Vec<f64> {
        self.to_genes()
            .iter()
            .map(|gene| {
                let magnitude = gene.abs();
                if magnitude > 1e-6 {
                    magnitude
                } else {
                    1e-3
                }
            })
            .collect()
    }
}
