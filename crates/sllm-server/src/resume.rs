use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

pub const MAX_REPLAY_EVENT_BYTES: usize = 64 * 1024;
pub const MAX_REPLAY_SESSION_BYTES: usize = 256 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayEventV1 {
    pub id: u64,
    pub data: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayReadV1 {
    pub events: Vec<ReplayEventV1>,
    pub terminal: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayErrorV1 {
    Capacity,
    NotFound,
    CursorOutOfRange,
    EventTooLarge,
    IdentifierExhausted,
    Terminal,
}

#[derive(Clone, Debug)]
pub struct ResumableStoreV1 {
    inner: Arc<Mutex<StoreInnerV1>>,
    max_sessions: usize,
    max_events_per_session: usize,
}

#[derive(Debug, Default)]
struct StoreInnerV1 {
    order: VecDeque<String>,
    sessions: BTreeMap<String, ReplaySessionV1>,
}

#[derive(Debug)]
struct ReplaySessionV1 {
    next_id: u64,
    events: VecDeque<ReplayEventV1>,
    retained_bytes: usize,
    terminal: bool,
}

impl ResumableStoreV1 {
    pub fn new(max_sessions: usize, max_events_per_session: usize) -> Result<Self, String> {
        if !(1..=1_024).contains(&max_sessions) {
            return Err("resumable session capacity must be in 1..=1024".to_owned());
        }
        if !(4..=65_536).contains(&max_events_per_session) {
            return Err("resumable event capacity must be in 4..=65536".to_owned());
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(StoreInnerV1::default())),
            max_sessions,
            max_events_per_session,
        })
    }

    pub fn create(&self, id: &str) -> Result<(), ReplayErrorV1> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if inner.sessions.contains_key(id) {
            return Err(ReplayErrorV1::Capacity);
        }
        while inner.sessions.len() == self.max_sessions {
            let Some(candidate) = inner
                .order
                .iter()
                .find(|candidate| {
                    inner
                        .sessions
                        .get(candidate.as_str())
                        .is_some_and(|session| session.terminal)
                })
                .cloned()
            else {
                return Err(ReplayErrorV1::Capacity);
            };
            inner.order.retain(|entry| entry != &candidate);
            inner.sessions.remove(&candidate);
        }
        inner.order.push_back(id.to_owned());
        inner.sessions.insert(
            id.to_owned(),
            ReplaySessionV1 {
                next_id: 1,
                events: VecDeque::new(),
                retained_bytes: 0,
                terminal: false,
            },
        );
        Ok(())
    }

    pub(crate) fn can_retain_batch(&self, event_lengths: &[usize]) -> bool {
        if event_lengths.is_empty() || event_lengths.len() > self.max_events_per_session {
            return false;
        }
        let Some(total) = event_lengths
            .iter()
            .try_fold(0_usize, |total, length| total.checked_add(*length))
        else {
            return false;
        };
        event_lengths
            .iter()
            .all(|length| *length <= MAX_REPLAY_EVENT_BYTES)
            && total <= MAX_REPLAY_SESSION_BYTES
    }

    pub fn append(&self, id: &str, data: String, terminal: bool) -> Result<u64, ReplayErrorV1> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let session = inner.sessions.get_mut(id).ok_or(ReplayErrorV1::NotFound)?;
        if session.terminal {
            return Err(ReplayErrorV1::Terminal);
        }
        if data.len() > MAX_REPLAY_EVENT_BYTES {
            return Err(ReplayErrorV1::EventTooLarge);
        }
        let event_id = session.next_id;
        session.next_id = session
            .next_id
            .checked_add(1)
            .ok_or(ReplayErrorV1::IdentifierExhausted)?;
        session.retained_bytes = session
            .retained_bytes
            .checked_add(data.len())
            .ok_or(ReplayErrorV1::Capacity)?;
        session
            .events
            .push_back(ReplayEventV1 { id: event_id, data });
        while session.events.len() > self.max_events_per_session
            || session.retained_bytes > MAX_REPLAY_SESSION_BYTES
        {
            if let Some(removed) = session.events.pop_front() {
                session.retained_bytes -= removed.data.len();
            }
        }
        session.terminal = terminal;
        Ok(event_id)
    }

    pub fn read_after(&self, id: &str, cursor: u64) -> Result<ReplayReadV1, ReplayErrorV1> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let session = inner.sessions.get(id).ok_or(ReplayErrorV1::NotFound)?;
        if let Some(first) = session.events.front() {
            if (cursor == 0 && first.id > 1) || cursor.saturating_add(1) < first.id {
                return Err(ReplayErrorV1::CursorOutOfRange);
            }
        }
        Ok(ReplayReadV1 {
            events: session
                .events
                .iter()
                .filter(|event| event.id > cursor)
                .cloned()
                .collect(),
            terminal: session.terminal,
        })
    }

    pub(crate) fn discard(&self, id: &str) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        inner.order.retain(|entry| entry != id);
        inner.sessions.remove(id);
    }

    pub(crate) fn terminate(&self, id: &str) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(session) = inner.sessions.get_mut(id) {
            session.terminal = true;
        }
    }

    pub fn session_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .sessions
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_is_monotonic_bounded_and_reports_overrun() {
        let store = ResumableStoreV1::new(2, 4).unwrap();
        store.create("a").unwrap();
        for value in 1..=5 {
            assert_eq!(
                store.append("a", value.to_string(), value == 5).unwrap(),
                value
            );
        }
        assert_eq!(
            store.read_after("a", 0),
            Err(ReplayErrorV1::CursorOutOfRange)
        );
        let replay = store.read_after("a", 3).unwrap();
        assert_eq!(
            replay
                .events
                .iter()
                .map(|event| event.id)
                .collect::<Vec<_>>(),
            vec![4, 5]
        );
        assert!(replay.terminal);
    }

    #[test]
    fn capacity_evicts_only_terminal_sessions() {
        let store = ResumableStoreV1::new(1, 4).unwrap();
        store.create("active").unwrap();
        assert_eq!(store.create("blocked"), Err(ReplayErrorV1::Capacity));
        store.append("active", "done".to_owned(), true).unwrap();
        store.create("next").unwrap();
        assert_eq!(store.read_after("active", 0), Err(ReplayErrorV1::NotFound));
        assert_eq!(store.session_count(), 1);
    }

    #[test]
    fn event_identifier_overflow_fails_without_duplicate_publication() {
        let store = ResumableStoreV1::new(1, 4).unwrap();
        store.create("overflow").unwrap();
        {
            let mut inner = store.inner.lock().unwrap();
            inner.sessions.get_mut("overflow").unwrap().next_id = u64::MAX;
        }
        assert_eq!(
            store.append("overflow", "never-published".to_owned(), false),
            Err(ReplayErrorV1::IdentifierExhausted)
        );
        assert!(store.read_after("overflow", 0).unwrap().events.is_empty());
    }

    #[test]
    fn reserved_session_can_be_discarded_after_downstream_rejection() {
        let store = ResumableStoreV1::new(1, 4).unwrap();
        store.create("reserved").unwrap();
        store.discard("reserved");
        assert_eq!(store.session_count(), 0);
        assert_eq!(
            store.read_after("reserved", 0),
            Err(ReplayErrorV1::NotFound)
        );
        store.create("replacement").unwrap();
    }

    #[test]
    fn replay_bytes_are_bounded_on_both_sides() {
        let store = ResumableStoreV1::new(1, 65_536).unwrap();
        store.create("bytes").unwrap();
        assert_eq!(
            store.append("bytes", "x".repeat(MAX_REPLAY_EVENT_BYTES + 1), false),
            Err(ReplayErrorV1::EventTooLarge)
        );
        for _ in 0..=MAX_REPLAY_SESSION_BYTES / MAX_REPLAY_EVENT_BYTES {
            store
                .append("bytes", "x".repeat(MAX_REPLAY_EVENT_BYTES), false)
                .unwrap();
        }
        let inner = store.inner.lock().unwrap();
        let session = inner.sessions.get("bytes").unwrap();
        assert_eq!(session.retained_bytes, MAX_REPLAY_SESSION_BYTES);
        assert_eq!(session.events.len(), 4);
    }

    #[test]
    fn batch_preflight_checks_event_count_and_both_byte_limits() {
        let store = ResumableStoreV1::new(1, 8).unwrap();
        assert!(store.can_retain_batch(&[1, MAX_REPLAY_EVENT_BYTES]));
        assert!(!store.can_retain_batch(&[]));
        assert!(
            !ResumableStoreV1::new(1, 4)
                .unwrap()
                .can_retain_batch(&[1, 1, 1, 1, 1])
        );
        assert!(!store.can_retain_batch(&[MAX_REPLAY_EVENT_BYTES + 1]));
        assert!(!store.can_retain_batch(&[
            MAX_REPLAY_EVENT_BYTES,
            MAX_REPLAY_EVENT_BYTES,
            MAX_REPLAY_EVENT_BYTES,
            MAX_REPLAY_EVENT_BYTES,
            1,
        ]));
    }

    #[test]
    fn forced_termination_stops_polling_without_growing_replay() {
        let store = ResumableStoreV1::new(1, 4).unwrap();
        store.create("failed").unwrap();
        store.terminate("failed");
        let replay = store.read_after("failed", 0).unwrap();
        assert!(replay.terminal);
        assert!(replay.events.is_empty());
        assert_eq!(
            store.append("failed", "late".to_owned(), false),
            Err(ReplayErrorV1::Terminal)
        );
    }
}
