use crate::{
    error::NeuralNetworkError,
    tensor::{storage::storage::StorageBackend, Device},
};

#[derive(PartialEq, Debug, Clone, Default, PartialOrd)]
pub struct MetalStorage {
    device_id: usize,
    len: usize,
}

impl MetalStorage {
    pub fn new(_device_id: usize, _size: usize) -> Result<Self, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Metal not yet implemented",
        ))
    }

    pub fn from_vec(_data: Vec<f32>) -> Result<Self, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Metal not yet implemented",
        ))
    }

    pub fn filled(_size: usize, _value: f32) -> Result<Self, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Metal not yet implemented",
        ))
    }
}

impl StorageBackend for MetalStorage {
    fn len(&self) -> usize {
        self.len
    }

    fn device(&self) -> Device {
        Device::Metal(self.device_id)
    }

    fn try_clone(&self) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Metal not yet implemented",
        ))
    }

    fn to_cpu(&self) -> Result<Vec<f32>, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Metal not yet implemented",
        ))
    }
}
