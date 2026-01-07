use crate::error::NeuralNetworkError;
use crate::tensor::backend::cpu::CpuStorage;
use crate::tensor::backend::metal::storage::MetalStorage;
use crate::tensor::Layout;

pub enum Storage {
    Cpu(CpuStorage),
    Metal(MetalStorage),
}

impl Storage {
    pub fn try_clone(&self, layout: &Layout) -> Result<Self, NeuralNetworkError> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.clone())),
            Self::Metal(storage) => Ok(Self::Metal(storage.clone())),
        }
    }
}
