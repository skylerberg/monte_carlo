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
