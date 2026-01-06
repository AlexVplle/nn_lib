use std::sync::Arc;

use crate::tensor::backend::{Backend, CpuBackend};

#[derive(Debug, PartialEq, Clone)]
pub enum Device {
    CPU,
    Metal(usize),
}

impl Device {
    pub fn backend(&self) -> Arc<dyn Backend> {
        match self {
            Device::CPU => Arc::new(CpuBackend),
            &Device::Metal(_) => todo!(),
        }
    }
}
