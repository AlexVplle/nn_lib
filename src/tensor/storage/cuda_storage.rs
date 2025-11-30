use crate::error::NeuralNetworkError;

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

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn try_clone(&self) -> Result<Self, NeuralNetworkError> {
        Err(NeuralNetworkError::NotImplemented(
            "Cuda not yet implemented",
        ))
    }
}
