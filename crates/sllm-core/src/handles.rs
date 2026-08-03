//! Backend-independent resource and lifetime contracts.
//!
//! These types carry opaque identity and ownership metadata only.  They do not
//! submit work, wait, poll, or claim that an operation is asynchronous.

use std::num::NonZeroU64;
use std::sync::Arc;

/// Access required for one use of a buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum AccessMode {
    Read = 1,
    Write = 2,
    ReadWrite = 3,
}

impl AccessMode {
    pub const fn permits_read(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    pub const fn permits_write(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }

    /// Returns the combined access required by two uses of one buffer.
    pub const fn join(self, other: Self) -> Self {
        match (
            self.permits_read() || other.permits_read(),
            self.permits_write() || other.permits_write(),
        ) {
            (true, true) => Self::ReadWrite,
            (true, false) => Self::Read,
            (false, true) => Self::Write,
            (false, false) => unreachable!(),
        }
    }
}

macro_rules! opaque_handle {
    ($name:ident) => {
        /// Backend-owned identity.  The value is never dereferenced by this crate.
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        #[repr(transparent)]
        pub struct $name(NonZeroU64);

        impl $name {
            /// Creates an identity from a non-zero backend token.
            pub const fn from_raw(raw: u64) -> Option<Self> {
                match NonZeroU64::new(raw) {
                    Some(raw) => Some(Self(raw)),
                    None => None,
                }
            }

            pub const fn raw(self) -> u64 {
                self.0.get()
            }
        }
    };
}

opaque_handle!(QueueHandle);
opaque_handle!(BufferHandle);
opaque_handle!(EventHandle);

/// One buffer reference retained by a completion lifetime token.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferUse {
    buffer: Arc<BufferHandle>,
    access: AccessMode,
}

impl BufferUse {
    pub fn new(buffer: Arc<BufferHandle>, access: AccessMode) -> Self {
        Self { buffer, access }
    }

    pub fn buffer(&self) -> &BufferHandle {
        self.buffer.as_ref()
    }

    pub fn buffer_arc(&self) -> &Arc<BufferHandle> {
        &self.buffer
    }

    pub const fn access(&self) -> AccessMode {
        self.access
    }
}

/// Ownership-only token for resources needed until a backend completion is
/// observed by a higher layer.
///
/// Constructing this value does not enqueue work and does not make any
/// completion or asynchronous-execution claim.  Its only contract is that the
/// queue, event, and buffer references remain strongly owned by the token.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionLease {
    queue: Arc<QueueHandle>,
    event: Arc<EventHandle>,
    buffers: Vec<BufferUse>,
}

impl CompletionLease {
    pub fn new(
        queue: Arc<QueueHandle>,
        event: Arc<EventHandle>,
        buffers: impl IntoIterator<Item = BufferUse>,
    ) -> Self {
        Self {
            queue,
            event,
            buffers: buffers.into_iter().collect(),
        }
    }

    pub fn queue(&self) -> &QueueHandle {
        self.queue.as_ref()
    }

    pub fn queue_arc(&self) -> &Arc<QueueHandle> {
        &self.queue
    }

    pub fn event(&self) -> &EventHandle {
        self.event.as_ref()
    }

    pub fn event_arc(&self) -> &Arc<EventHandle> {
        &self.event
    }

    pub fn buffers(&self) -> &[BufferUse] {
        &self.buffers
    }

    pub fn holds_buffer(&self, buffer: &BufferHandle) -> bool {
        self.buffers.iter().any(|use_| use_.buffer() == buffer)
    }
}

/// Name used by the runtime architecture for the same ownership-only token.
pub type InFlightSubmission = CompletionLease;
