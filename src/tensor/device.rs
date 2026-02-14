use crate::error::NeuralNetworkError;

#[cfg(target_os = "macos")]
use crate::tensor::backend::{backend_device::BackendDevice, metal::MetalDevice};

#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    #[cfg(target_os = "macos")]
    Metal(MetalDevice),
}

impl Device {
    #[cfg(target_os = "macos")]
    pub fn new_metal(ordinal: usize) -> Result<Self, NeuralNetworkError> {
        Ok(Self::Metal(MetalDevice::new(ordinal)?))
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new_metal(_ordinal: usize) -> Result<Self, NeuralNetworkError> {
        Err(NeuralNetworkError::Message(
            "Metal backend is only available on macOS".to_string(),
        ))
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        #[allow(unreachable_patterns)]
        match (self, rhs) {
            (Self::CPU, Self::CPU) => true,
            #[cfg(target_os = "macos")]
            (Self::Metal(lhs), Self::Metal(rhs)) => lhs.same_device(rhs),
            _ => false,
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::CPU)
    }

    pub fn is_metal(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            matches!(self, Self::Metal(_))
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}
