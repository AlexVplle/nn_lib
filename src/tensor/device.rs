use crate::{
    error::NeuralNetworkError,
    tensor::backend::{backend_device::BackendDevice, metal::MetalDevice},
};

#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    Metal(MetalDevice),
}

impl Device {
    pub fn new_metal(ordinal: usize) -> Result<Self, NeuralNetworkError> {
        Ok(Self::Metal(MetalDevice::new(ordinal)?))
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::CPU, Self::CPU) => true,
            (Self::Metal(lhs), Self::Metal(rhs)) => lhs.same_device(rhs),
            _ => false,
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::CPU)
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Self::Metal(_))
    }
}
