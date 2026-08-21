use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerLifecycleStateV1 {
    Loading,
    Ready,
    Draining,
    Failed,
    Shutdown,
}

impl ServerLifecycleStateV1 {
    const fn encode(self) -> u8 {
        match self {
            Self::Loading => 0,
            Self::Ready => 1,
            Self::Draining => 2,
            Self::Failed => 3,
            Self::Shutdown => 4,
        }
    }

    const fn decode(value: u8) -> Self {
        match value {
            0 => Self::Loading,
            1 => Self::Ready,
            2 => Self::Draining,
            3 => Self::Failed,
            _ => Self::Shutdown,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ServerLifecycleV1 {
    state: Arc<AtomicU8>,
}

impl ServerLifecycleV1 {
    pub fn new(state: ServerLifecycleStateV1) -> Self {
        Self {
            state: Arc::new(AtomicU8::new(state.encode())),
        }
    }

    pub fn state(&self) -> ServerLifecycleStateV1 {
        ServerLifecycleStateV1::decode(self.state.load(Ordering::Acquire))
    }

    pub fn transition(&self, state: ServerLifecycleStateV1) {
        self.state.store(state.encode(), Ordering::Release);
    }

    pub fn is_ready(&self) -> bool {
        self.state() == ServerLifecycleStateV1::Ready
    }
}

impl Default for ServerLifecycleV1 {
    fn default() -> Self {
        Self::new(ServerLifecycleStateV1::Ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_observe_fail_closed_transitions() {
        let lifecycle = ServerLifecycleV1::new(ServerLifecycleStateV1::Loading);
        let observer = lifecycle.clone();
        assert!(!observer.is_ready());
        lifecycle.transition(ServerLifecycleStateV1::Ready);
        assert!(observer.is_ready());
        lifecycle.transition(ServerLifecycleStateV1::Draining);
        assert!(!observer.is_ready());
        lifecycle.transition(ServerLifecycleStateV1::Failed);
        assert_eq!(observer.state(), ServerLifecycleStateV1::Failed);
        lifecycle.transition(ServerLifecycleStateV1::Shutdown);
        assert_eq!(observer.state(), ServerLifecycleStateV1::Shutdown);
    }
}
