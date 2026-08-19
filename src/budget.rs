#[cfg(feature = "time")]
pub(crate) struct Deadline {
    end: Option<std::time::Instant>,
}

#[cfg(feature = "time")]
impl Deadline {
    pub(crate) fn new(limit_ms: Option<u64>) -> Self {
        Self {
            end: limit_ms
                .map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms)),
        }
    }

    #[inline]
    pub(crate) fn expired(&self) -> bool {
        match self.end {
            Some(end) => std::time::Instant::now() >= end,
            None => false,
        }
    }
}

/// Without the `time` feature there is no clock — `Instant` panics on
/// `wasm32-unknown-unknown`, so wasm builds turn the feature off.
#[cfg(not(feature = "time"))]
pub(crate) struct Deadline;

#[cfg(not(feature = "time"))]
impl Deadline {
    pub(crate) fn new(limit_ms: Option<u64>) -> Self {
        assert!(
            limit_ms.is_none(),
            "mcts: Config::time_limit_ms requires the `time` feature"
        );
        Self
    }

    #[inline]
    pub(crate) fn expired(&self) -> bool {
        false
    }
}
