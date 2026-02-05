use crate::error::NeuralNetworkError;

#[cfg(feature = "metal")]
use crate::tensor::backend::metal::MetalDevice;

#[cfg(feature = "metal")]
use crate::tensor::backend::backend_device::BackendDevice;

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

    #[cfg(not(feature = "metal"))]
    pub fn new_metal(_ordinal: usize) -> Result<Self, NeuralNetworkError> {
        Err(NeuralNetworkError::TensorError(
            crate::tensor::TensorError::UnsupportedBackend("metal"),
        ))
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::CPU, Self::CPU) => true,
            #[cfg(feature = "metal")]
            (Self::Metal(lhs), Self::Metal(rhs)) => lhs.same_device(rhs),
            _ => false,
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::CPU)
    }

    pub fn is_metal(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            matches!(self, Self::Metal(_))
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }
}
