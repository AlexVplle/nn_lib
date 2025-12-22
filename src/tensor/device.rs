use std::sync::Arc;

use crate::tensor::backend::{Backend, CpuBackend};

#[derive(Debug, PartialEq, Clone)]
pub enum Device {
    CPU,
    CUDA(usize),
    Metal(usize),
}

impl Device {
    pub fn backend(&self) -> Arc<dyn Backend> {
        match self {
            Device::CPU => Arc::new(CpuBackend),
            &Device::CUDA(_) | &Device::Metal(_) => todo!()
        }
    }
}