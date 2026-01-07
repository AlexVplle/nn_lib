use crate::{error::NeuralNetworkError, tensor::backend::backend_storage::BackendStorage};

pub trait BackendDevice {
    type Storage: BackendStorage;

    fn new(_: usize) -> Result<Self, NeuralNetworkError>;
    fn same_device(&self, _: &Self) -> bool;
}
