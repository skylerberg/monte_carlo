use mcts::rand_core::Rng;
use wyrand::WyRand;

/// A uniform draw from `[0, 1)`, built from the top 53 bits so every value is a
/// distinct `f64`.
pub fn uniform(rng: &mut WyRand) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// A uniform index into `0..len`, without the modulo bias of `% len`.
pub fn index(rng: &mut WyRand, len: usize) -> usize {
    ((rng.next_u64() as u128 * len as u128) >> 64) as usize
}

/// One standard normal by Box–Muller.
///
/// Both optimizers draw a handful of these per generation, against hundreds of
/// games, so discarding the transform's second variate costs nothing worth the
/// state to keep it in.
pub fn standard_normal(rng: &mut WyRand) -> f64 {
    // Clamped away from zero: `ln(0)` is negative infinity, and one such draw
    // would poison a whole covariance matrix.
    let u1 = uniform(rng).max(1e-300);
    let u2 = uniform(rng);
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}
