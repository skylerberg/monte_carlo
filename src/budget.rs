//! When a wall-clock budget has run out.
//!
//! Two clocks, one rule. With the `time` feature the crate reads
//! [`std::time::Instant`]; without it — on `wasm32-unknown-unknown`, where
//! `Instant` panics — the caller supplies a monotonic millisecond reader in
//! [`crate::Config::clock`]. Neither is consulted every iteration: the search
//! polls every 32 iterations, so a deadline costs one indirect call per 32
//! visits and nothing else.

/// The deadline's source, if there is one.
enum Source {
    /// A caller-supplied monotonic clock and the millisecond reading at which
    /// the budget is spent.
    Clock(fn() -> u64, u64),
    #[cfg(feature = "time")]
    Instant(std::time::Instant),
}

pub(crate) struct Deadline {
    source: Option<Source>,
}

impl Deadline {
    /// `clock` is the caller's, when it gave one. It wins over `Instant`
    /// whenever both are available, so a caller that installs a clock gets
    /// exactly the clock it asked for in every build profile.
    pub(crate) fn new(limit_ms: Option<u64>, clock: Option<fn() -> u64>) -> Self {
        let source = match (limit_ms, clock) {
            (None, _) => None,
            (Some(ms), Some(clock)) => {
                // Saturating rather than wrapping: a clock near `u64::MAX` is a
                // deadline in the far future, not one already past.
                Some(Source::Clock(clock, clock().saturating_add(ms)))
            }
            #[cfg(feature = "time")]
            (Some(ms), None) => Some(Source::Instant(
                std::time::Instant::now() + std::time::Duration::from_millis(ms),
            )),
            #[cfg(not(feature = "time"))]
            (Some(_), None) => panic!(
                "mcts: Config::time_limit_ms needs a clock. This build has no `time` feature \
                 (it is off for `wasm32-unknown-unknown`, where `Instant` panics), so pass \
                 `Config::clock` — a function returning monotonic milliseconds."
            ),
        };
        Self { source }
    }

    #[inline]
    pub(crate) fn expired(&self) -> bool {
        match self.source {
            None => false,
            Some(Source::Clock(now, end)) => now() >= end,
            #[cfg(feature = "time")]
            Some(Source::Instant(end)) => std::time::Instant::now() >= end,
        }
    }
}
