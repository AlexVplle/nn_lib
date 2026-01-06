use crate::{
    error::NeuralNetworkError,
    tensor::{storage::storage::StorageBackend, Device},
};

use std::any::Any;

#[derive(PartialEq, Debug, Clone, Default, PartialOrd)]
pub struct CudaStorage {
    device_id: usize,
    len: usize,
}

impl CudaStorage {
    pub fn new(_device_id: usize, _size: usize) -> Result<Self, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Cuda not yet implemented",
        ))
    }

    pub fn from_vec(_data: Vec<f32>) -> Result<Self, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Cuda not yet implemented",
        ))
    }

    pub fn filled(_size: usize, _value: f32) -> Result<Self, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Cuda not yet implemented",
        ))
    }
}

impl StorageBackend for CudaStorage {
    fn len(&self) -> usize {
        self.len
    }

    fn device(&self) -> Device {
        todo!()
    }

    fn try_clone(&self) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Cuda not yet implemented",
        ))
    }

    fn to_cpu(&self) -> Result<Vec<f32>, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Cuda not yet implemented",
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
