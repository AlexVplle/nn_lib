use std::{
    ops::Deref,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{
    error::NeuralNetworkError,
    tensor::{
        storage::{cpu_storage::CpuStorage, storage::StorageBackend},
        Device, Layout,
    },
};

pub struct Tensor_ {
    storage: Arc<RwLock<Box<dyn StorageBackend>>>,
    layout: Layout,
    device: Device,
    gradient: Option<Tensor>,
    require_gradient: bool,
}

#[derive(Clone)]
pub struct Tensor(Arc<Tensor_>);

impl Deref for Tensor {
    type Target = Tensor_;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    pub fn new(
        data: Vec<f32>,
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, NeuralNetworkError> {
        let size: usize = shape.iter().product();
        if data.len() != size {
            return Err(NeuralNetworkError::IncompatibleShape {
                shape_given: shape,
                tensor_shape: vec![data.len()],
            });
        }

        let storage: Box<dyn StorageBackend> = match device {
            Device::CPU => Box::new(CpuStorage::from_vec(data)),
            Device::CUDA(_id) => todo!(),
            Device::Metal(_id) => todo!(),
        };

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::new(shape),
            device,
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn zeros(shape: Vec<usize>, device: Device) -> Result<Self, NeuralNetworkError> {
        let size: usize = shape.iter().product();
        let storage: Box<dyn StorageBackend> = match device {
            Device::CPU => Box::new(CpuStorage::new(size)),
            Device::CUDA(_id) => todo!(),
            Device::Metal(_id) => todo!(),
        };

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::new(shape),
            device,
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn ones(shape: Vec<usize>, device: Device) -> Result<Self, NeuralNetworkError> {
        let size: usize = shape.iter().product();
        let storage: Box<dyn StorageBackend> = match device {
            Device::CPU => Box::new(CpuStorage::filled(size, 1.0)),
            Device::CUDA(_id) => todo!(),
            Device::Metal(_id) => todo!(),
        };

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::new(shape),
            device,
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn storage(&self) -> RwLockReadGuard<'_, Box<dyn StorageBackend>> {
        self.storage.read().unwrap()
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn to_cpu(&self) -> Result<Vec<f32>, NeuralNetworkError> {
        let storage: RwLockReadGuard<'_, Box<dyn StorageBackend>> = self
            .storage
            .read()
            .map_err(|_| NeuralNetworkError::LockError)?;
        storage.to_cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape: Vec<usize> = vec![2, 3];
        let tensor: Result<Tensor, NeuralNetworkError> = Tensor::new(data.clone(), shape, Device::CPU);

        assert!(tensor.is_ok());
        let tensor: Tensor = tensor.unwrap();

        assert_eq!(tensor.to_cpu().unwrap(), data);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(*tensor.device(), Device::CPU);
    }

    #[test]
    fn test_new_incompatible_shape() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let shape: Vec<usize> = vec![2, 3];
        let result: Result<Tensor, NeuralNetworkError> = Tensor::new(data, shape, Device::CPU);

        assert!(result.is_err());
    }

    #[test]
    fn test_new_1d() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let shape: Vec<usize> = vec![3];
        let tensor: Tensor = Tensor::new(data.clone(), shape, Device::CPU).unwrap();

        assert_eq!(tensor.to_cpu().unwrap(), data);
        assert_eq!(tensor.shape(), &[3]);
    }

    #[test]
    fn test_new_3d() {
        let data: Vec<f32> = vec![1.0; 24];
        let shape: Vec<usize> = vec![2, 3, 4];
        let tensor: Tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.layout.num_elements(), 24);
    }

    #[test]
    fn test_zeros() {
        let shape: Vec<usize> = vec![2, 3];
        let tensor: Tensor = Tensor::zeros(shape, Device::CPU).unwrap();

        assert_eq!(tensor.to_cpu().unwrap(), vec![0.0; 6]);
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_zeros_1d() {
        let shape: Vec<usize> = vec![5];
        let tensor: Tensor = Tensor::zeros(shape, Device::CPU).unwrap();

        assert_eq!(tensor.to_cpu().unwrap(), vec![0.0; 5]);
    }

    #[test]
    fn test_ones() {
        let shape: Vec<usize> = vec![2, 3];
        let tensor: Tensor = Tensor::ones(shape, Device::CPU).unwrap();

        assert_eq!(tensor.to_cpu().unwrap(), vec![1.0; 6]);
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_ones_1d() {
        let shape: Vec<usize> = vec![4];
        let tensor: Tensor = Tensor::ones(shape, Device::CPU).unwrap();

        assert_eq!(tensor.to_cpu().unwrap(), vec![1.0; 4]);
    }

    #[test]
    fn test_storage() {
        let tensor: Tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], Device::CPU).unwrap();
        let storage: RwLockReadGuard<'_, Box<dyn StorageBackend>> = tensor.storage();

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.device(), Device::CPU);
    }

    #[test]
    fn test_shape() {
        let tensor: Tensor = Tensor::zeros(vec![2, 3, 4], Device::CPU).unwrap();
        assert_eq!(tensor.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_strides() {
        let tensor: Tensor = Tensor::zeros(vec![2, 3, 4], Device::CPU).unwrap();
        assert_eq!(tensor.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_is_contiguous() {
        let tensor: Tensor = Tensor::zeros(vec![2, 3], Device::CPU).unwrap();
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_device() {
        let tensor: Tensor = Tensor::zeros(vec![2, 3], Device::CPU).unwrap();
        assert_eq!(*tensor.device(), Device::CPU);
    }

    #[test]
    fn test_to_cpu() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor: Tensor = Tensor::new(data.clone(), vec![2, 2], Device::CPU).unwrap();

        let result: Result<Vec<f32>, NeuralNetworkError> = tensor.to_cpu();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), data);
    }

    #[test]
    fn test_clone() {
        let tensor1: Tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], Device::CPU).unwrap();
        let tensor2: Tensor = tensor1.clone();

        assert_eq!(tensor1.to_cpu().unwrap(), tensor2.to_cpu().unwrap());
        assert_eq!(tensor1.shape(), tensor2.shape());
    }

    #[test]
    fn test_clone_shares_inner() {
        let tensor1: Tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], Device::CPU).unwrap();
        let tensor2: Tensor = tensor1.clone();

        assert!(Arc::ptr_eq(&tensor1.0, &tensor2.0));
    }

    #[test]
    fn test_deref_layout() {
        let tensor: Tensor = Tensor::zeros(vec![2, 3], Device::CPU).unwrap();
        assert_eq!(tensor.layout.shape(), &[2, 3]);
        assert_eq!(tensor.layout.ndim(), 2);
    }

    #[test]
    fn test_deref_gradient() {
        let tensor: Tensor = Tensor::zeros(vec![2, 3], Device::CPU).unwrap();
        assert!(tensor.gradient.is_none());
    }

    #[test]
    fn test_deref_require_gradient() {
        let tensor: Tensor = Tensor::zeros(vec![2, 3], Device::CPU).unwrap();
        assert!(!tensor.require_gradient);
    }
}
