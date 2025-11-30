use crate::{
    error::NeuralNetworkError,
    tensor::{storage::cpu_storage::CpuStorage, Device},
};

#[derive(PartialEq, Debug, Clone, PartialOrd)]
pub enum Storage {
    Cpu(CpuStorage),
}

impl Storage {
    pub fn zeros(size: usize, device: Device) -> Result<Storage, NeuralNetworkError> {
        match device {
            Device::CPU => Ok(Storage::Cpu(CpuStorage::new(size))),
            Device::CUDA => 
        }
    }
}
