use crate::{Backend, FakeBackend};

pub struct BackendRegistration {
    name: &'static str,
    create: fn() -> Box<dyn Backend>,
}

impl BackendRegistration {
    pub const fn new(name: &'static str, create: fn() -> Box<dyn Backend>) -> Self {
        Self { name, create }
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }

    pub fn create(&self) -> Box<dyn Backend> {
        (self.create)()
    }
}

fn create_fake_backend() -> Box<dyn Backend> {
    Box::new(FakeBackend::new())
}

/// The Phase 1 registry is static and intentionally contains no dynamic plugin loading.
pub static BACKEND_REGISTRY: &[BackendRegistration] =
    &[BackendRegistration::new("fake", create_fake_backend)];

pub fn backend_registry() -> &'static [BackendRegistration] {
    BACKEND_REGISTRY
}
