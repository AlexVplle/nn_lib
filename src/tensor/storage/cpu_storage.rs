use crate::{
    error::NeuralNetworkError,
    tensor::{storage::storage::StorageBackend, Device},
};

#[derive(PartialEq, Debug, Clone, Default, PartialOrd)]
pub struct CpuStorage {
    data: Box<[f32]>,
}

impl CpuStorage {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size].into_boxed_slice(),
        }
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        Self {
            data: data.into_boxed_slice(),
        }
    }

    pub fn filled(size: usize, value: f32) -> Self {
        Self {
            data: vec![value; size].into_boxed_slice(),
        }
    }

    fn as_slice(&self) -> &[f32] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    fn to_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }
}

impl StorageBackend for CpuStorage {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn try_clone(&self) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        Ok(Box::new(CpuStorage {
            data: self.data.to_vec().into_boxed_slice(),
        }))
    }

    fn to_cpu(&self) -> Result<Vec<f32>, NeuralNetworkError> {
        Ok(self.data.to_vec())
    }
}
