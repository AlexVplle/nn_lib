mod cpu;
mod metal;

pub use cpu::CpuBackend;

use std::sync::{Arc, RwLock};

use crate::error::NeuralNetworkError;
use crate::tensor::{storage::storage::StorageBackend, Device};

pub trait Backend: Send + Sync {
    fn add(
        &self,
        lhs: &Arc<RwLock<Box<dyn StorageBackend>>>,
        rhs: &Arc<RwLock<Box<dyn StorageBackend>>>,
    ) -> Result<Box<dyn StorageBackend>, NeuralNetworkError>;

    fn mul(
        &self,
        lhs: &Arc<RwLock<Box<dyn StorageBackend>>>,
        rhs: &Arc<RwLock<Box<dyn StorageBackend>>>,
    ) -> Result<Box<dyn StorageBackend>, NeuralNetworkError>;

    fn device(&self) -> Device;
}
