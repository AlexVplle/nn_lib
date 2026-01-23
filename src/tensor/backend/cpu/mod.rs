pub mod storage;
pub use storage::CpuStorage;

use crate::tensor::{backend::backend_device::BackendDevice, TensorError};

pub struct CpuDevice;

impl BackendDevice for CpuDevice {
    type Storage = CpuStorage;

    fn new(_: usize) -> Result<Self, TensorError> {
        Ok(Self)
    }

    fn storage_from_cpu_storage_owned(
        &self,
        cpu_storage: CpuStorage,
    ) -> Result<Self::Storage, TensorError> {
        Ok(cpu_storage)
    }

    fn storage_from_cpu_storage(
        &self,
        cpu_storage: &CpuStorage,
    ) -> Result<Self::Storage, TensorError> {
        Ok(cpu_storage.clone())
    }

    fn storage_from_vec(&self, data: Vec<f32>) -> Result<Self::Storage, TensorError> {
        Ok(CpuStorage(data))
    }

    fn same_device(&self, _: &Self) -> bool {
        true
    }
}
