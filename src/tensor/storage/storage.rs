use crate::{error::NeuralNetworkError, tensor::Device};
use std::any::Any;

pub trait StorageBackend {
    fn len(&self) -> usize;
    fn device(&self) -> Device;
    fn try_clone(&self) -> Result<Box<dyn StorageBackend>, NeuralNetworkError>;
    fn to_cpu(&self) -> Result<Vec<f32>, NeuralNetworkError>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
        
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}