use crate::{
    error::NeuralNetworkError,
    tensor::backend::backend_device::BackendDevice,
};

#[cfg(feature = "metal")]
use crate::tensor::backend::metal::MetalDevice;

#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    #[cfg(feature = "metal")]
    Metal(MetalDevice),
}

impl Device {
    #[cfg(feature = "metal")]
    pub fn new_metal(ordinal: usize) -> Result<Self, NeuralNetworkError> {
        Ok(Self::Metal(MetalDevice::new(ordinal)?))
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::CPU, Self::CPU) => true,
            #[cfg(feature = "metal")]
            (Self::Metal(lhs), Self::Metal(rhs)) => lhs.same_device(rhs),
            #[cfg(not(feature = "metal"))]
            _ => false,
            #[cfg(feature = "metal")]
            _ => false,
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::CPU)
    }

    #[cfg(feature = "metal")]
    pub fn is_metal(&self) -> bool {
        matches!(self, Self::Metal(_))
    }

    #[cfg(not(feature = "metal"))]
    pub fn is_metal(&self) -> bool {
        false
    }
}
