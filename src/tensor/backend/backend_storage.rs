use crate::{
    error::NeuralNetworkError,
    tensor::backend::{backend_device::BackendDevice, cpu::CpuStorage},
};

pub trait BackendStorage {
    type Device: BackendDevice;

    fn device(&self) -> &Self::Device;
    fn to_cpu_storage(&self) -> Result<CpuStorage, NeuralNetworkError>;
}

