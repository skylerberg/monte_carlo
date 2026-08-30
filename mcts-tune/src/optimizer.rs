/// A black-box optimizer over a fixed-dimension gene vector, driven ask/tell.
///
/// The loop in [`crate::run`] never learns which strategy it is holding: it
/// asks for a generation's candidates, measures them, and tells the optimizer
/// what it found. Fitness is a win rate, so **higher is better** — an
/// implementation ported from a minimizer has to flip its comparisons.
///
/// Implementations own their RNG rather than borrowing one, which keeps the
/// trait object safe and makes a run reproducible from the seed it was built
/// with.
pub trait Optimizer {
    /// The candidates to evaluate this generation.
    ///
    /// Every vector is `dimension` long. The caller repairs them through
    /// [`crate::Tunable::repair`] before use and hands the repaired values back
    /// through [`Optimizer::tell`], so an implementation must not assume it
    /// sees its own proposals unmodified.
    fn ask(&mut self) -> Vec<Vec<f64>>;

    /// Fitness for each candidate from the last [`Optimizer::ask`], in the same
    /// order, alongside the repaired genes that were actually played.
    fn tell(&mut self, candidates: &[Vec<f64>], fitness: &[f64]);

    /// The best genes seen so far and the fitness measured for them.
    ///
    /// Measured, not true: with a few hundred games per candidate this is a
    /// noisy maximum and so is biased upward. Re-measure the winner of a run
    /// before believing its number.
    fn best(&self) -> (&[f64], f64);

    /// How many candidates a generation holds, for reporting a run's cost
    /// before it starts.
    fn population(&self) -> usize;

    /// Name for logs, and the tag that stops one strategy's checkpoint being
    /// loaded into another.
    fn name(&self) -> &'static str;

    /// Everything this optimizer needs to carry on from here.
    ///
    /// A tuning run is measured in hours, so the state that decides whether an
    /// interruption costs minutes or the whole run is the state in here: for a
    /// covariance-adapting strategy that is the learned covariance, which is
    /// the entire reason to use one. Re-centring a fresh optimizer on the best
    /// parameters found so far is not the same thing and is not a resume.
    fn snapshot(&self) -> serde_json::Value;

    /// Restore state written by [`Optimizer::snapshot`] from the same strategy
    /// at the same dimension.
    fn restore(&mut self, snapshot: &serde_json::Value) -> Result<(), crate::ResumeError>;
}
