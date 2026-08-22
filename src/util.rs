use core::hash::{Hash, Hasher};

use rand_core::Rng;
use rustc_hash::FxHasher;

/// Uniform integer in `[0, n)` via Lemire's multiply-shift.
///
/// Rejection sampling would need a division; this needs a widening multiply and
/// a shift. The bias is under 2^-64 per draw, which is far below anything a
/// playout can observe. Returns 0 when `n == 0`.
#[inline(always)]
pub(crate) fn below<R: Rng + ?Sized>(rng: &mut R, n: u64) -> u64 {
    ((rng.next_u64() as u128 * n as u128) >> 64) as u64
}

#[inline]
pub(crate) fn hash_of<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Uniform `f64` in `[0, 1)`.
///
/// The top 53 bits scaled by `2^-53`, which is every value f64 can represent
/// in that interval without a gap. Dividing a whole 64-bit draw by `u64::MAX`
/// would instead round the largest draws to exactly `1.0`, and an inverse-CDF
/// sampler handed `1.0` walks off the end of its distribution.
#[inline(always)]
pub(crate) fn uniform_01<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}
