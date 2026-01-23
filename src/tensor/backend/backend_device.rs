use crate::tensor::{
    backend::{backend_storage::BackendStorage, cpu::CpuStorage},
    TensorError,
};

pub trait BackendDevice: Sized {
    type Storage: BackendStorage;

    fn new(_: usize) -> Result<Self, TensorError>;
    fn storage_from_cpu_storage_owned(
        &self,
        _: CpuStorage,
    ) -> Result<Self::Storage, TensorError>;
    fn storage_from_cpu_storage(&self, _: &CpuStorage)
        -> Result<Self::Storage, TensorError>;
    fn storage_from_vec(&self, data: Vec<f32>) -> Result<Self::Storage, TensorError>;
    fn same_device(&self, _: &Self) -> bool;
}
