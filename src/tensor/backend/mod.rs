mod cpu;

pub use cpu::CpuBackend;

use std::sync::{Arc, RwLock};

use crate::tensor::{Device, storage::storage::StorageBackend};
use crate::error::NeuralNetworkError;

pub trait Backend: Send + Sync {
    fn add(&self, lhs: &Arc<RwLock<Box<dyn StorageBackend>>>, rhs: &Arc<RwLock<Box<dyn StorageBackend>>>) -> Result<Box<dyn StorageBackend>, NeuralNetworkError>;
    fn relu(&self, storage: &Arc<RwLock<Box<dyn StorageBackend>>>) -> Result<Box<dyn StorageBackend>, NeuralNetworkError>;
    fn device(&self) -> Device;
}
