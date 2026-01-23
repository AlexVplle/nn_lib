use crate::tensor::backend::backend_storage::BackendStorage;
use crate::tensor::backend::cpu::CpuStorage;
use crate::tensor::backend::metal::storage::MetalStorage;
use crate::tensor::{Device, TensorError};

#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Metal(MetalStorage),
}

impl Storage {
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.clone())),
            Self::Metal(storage) => Ok(Self::Metal(storage.clone())),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::CPU,
            Self::Metal(storage) => Device::Metal(storage.device().clone()),
        }
    }

    pub fn to_cpu_storage(&self) -> Result<CpuStorage, TensorError> {
        match self {
            Self::Cpu(storage) => storage.to_cpu_storage(),
            Self::Metal(storage) => storage.to_cpu_storage(),
        }
    }

    pub fn same_device(&self, rhs: &Self) -> Result<(), TensorError> {
        let lhs_device = self.device();
        let rhs_device = rhs.device();
        if !lhs_device.same_device(&rhs_device) {
            return Err(TensorError::DeviceMismatch {
                first: lhs_device,
                second: rhs_device,
            });
        }
        Ok(())
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.same_device(rhs)?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => Ok(Storage::Cpu(lhs.add(rhs)?)),
            (Storage::Metal(lhs), Storage::Metal(rhs)) => Ok(Storage::Metal(lhs.add(rhs)?)),
            _ => Err(TensorError::DeviceMismatch {
                first: self.device(),
                second: rhs.device(),
            }),
        }
    }

    pub fn matmul(&self, rhs: &Self, m: usize, k: usize, n: usize) -> Result<Self, TensorError> {
        self.same_device(rhs)?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => Ok(Storage::Cpu(lhs.matmul(rhs, m, k, n)?)),
            (Storage::Metal(lhs), Storage::Metal(rhs)) => {
                Ok(Storage::Metal(lhs.matmul(rhs, m, k, n)?))
            }
            _ => Err(TensorError::DeviceMismatch {
                first: self.device(),
                second: rhs.device(),
            }),
        }
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.same_device(rhs)?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => Ok(Storage::Cpu(lhs.mul(rhs)?)),
            (Storage::Metal(lhs), Storage::Metal(rhs)) => Ok(Storage::Metal(lhs.mul(rhs)?)),
            _ => Err(TensorError::DeviceMismatch {
                first: self.device(),
                second: rhs.device(),
            }),
        }
    }

    pub fn relu(&self) -> Result<Self, TensorError> {
        match self {
            Storage::Cpu(storage) => Ok(Storage::Cpu(storage.relu()?)),
            Storage::Metal(storage) => Ok(Storage::Metal(storage.relu()?)),
        }
    }

    pub fn tanh(&self) -> Result<Self, TensorError> {
        match self {
            Storage::Cpu(storage) => Ok(Storage::Cpu(storage.tanh()?)),
            Storage::Metal(storage) => Ok(Storage::Metal(storage.tanh()?)),
        }
    }

    pub fn sigmoid(&self) -> Result<Self, TensorError> {
        match self {
            Storage::Cpu(storage) => Ok(Storage::Cpu(storage.sigmoid()?)),
            Storage::Metal(storage) => Ok(Storage::Metal(storage.sigmoid()?)),
        }
    }

    pub fn softmax(&self, batch_size: usize, vector_size: usize) -> Result<Self, TensorError> {
        match self {
            Storage::Cpu(storage) => Ok(Storage::Cpu(storage.softmax(batch_size, vector_size)?)),
            Storage::Metal(storage) => {
                Ok(Storage::Metal(storage.softmax(batch_size, vector_size)?))
            }
        }
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.same_device(rhs)?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => Ok(Storage::Cpu(lhs.sub(rhs)?)),
            (Storage::Metal(lhs), Storage::Metal(rhs)) => Ok(Storage::Metal(lhs.sub(rhs)?)),
            _ => Err(TensorError::DeviceMismatch {
                first: self.device(),
                second: rhs.device(),
            }),
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        match self {
            Storage::Cpu(storage) => Ok(Storage::Cpu(storage.mul_scalar(scalar)?)),
            Storage::Metal(storage) => Ok(Storage::Metal(storage.mul_scalar(scalar)?)),
        }
    }
}
