use std::{
    ops::{Add, Deref, Mul},
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{
    error::NeuralNetworkError,
    tensor::{
        backend::{cpu::CpuStorage, Backend},
        storage::StorageBackend,
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

impl Add for Tensor {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(
            self.device(),
            other.device(),
            "Cannot add tensors on different devices: {:?} and {:?}",
            self.device(),
            other.device()
        );
        assert_eq!(
            self.shape(),
            other.shape(),
            "Cannot add tensors with different shapes: {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        let backend: Arc<dyn Backend> = self.device().backend();
        let result_storage = backend
            .add(&self.storage, &other.storage)
            .expect("Backend add failed");
        Self(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(result_storage)),
            layout: Layout::new(self.shape().to_vec()),
            device: self.device.clone(),
            gradient: None,
            require_gradient: self.require_gradient || other.require_gradient,
        }))
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(
            self.device(),
            other.device(),
            "Cannot add tensors on different devices: {:?} and {:?}",
            self.device(),
            other.device()
        );
        assert_eq!(
            self.shape(),
            other.shape(),
            "Cannot add tensors with different shapes: {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        let backend: Arc<dyn Backend> = self.device().backend();
        let result_storage = backend
            .mul(&self.storage, &other.storage)
            .expect("Backend add failed");
        Self(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(result_storage)),
            layout: Layout::new(self.shape().to_vec()),
            device: self.device.clone(),
            gradient: None,
            require_gradient: self.require_gradient || other.require_gradient,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape: Vec<usize> = vec![2, 3];
        let tensor: Result<Tensor, NeuralNetworkError> =
            Tensor::new(data.clone(), shape, Device::CPU);

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

    #[test]
    fn test_add_chain() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![10.0, 20.0], vec![2], Device::CPU).unwrap();
        let c: Tensor = Tensor::new(vec![100.0, 200.0], vec![2], Device::CPU).unwrap();

        let result: Tensor = a + b + c;

        let data: Vec<f32> = result.to_cpu().unwrap();
        assert_eq!(data, vec![111.0, 222.0]);
    }

    #[test]
    fn test_add_same_tensor() {
        let a: Tensor = Tensor::new(vec![5.0, 10.0, 15.0], vec![3], Device::CPU).unwrap();
        let b: Tensor = a.clone();

        let result: Tensor = a + b;

        let data: Vec<f32> = result.to_cpu().unwrap();
        assert_eq!(data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_add_result_device() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![3.0, 4.0], vec![2], Device::CPU).unwrap();

        let c: Tensor = a + b;

        assert_eq!(*c.device(), Device::CPU);
    }

    #[test]
    fn test_add_result_shape() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        let c: Tensor = a + b;

        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_add_result_is_contiguous() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![4.0, 5.0, 6.0], vec![3], Device::CPU).unwrap();

        let c: Tensor = a + b;

        assert!(c.is_contiguous());
    }

    #[test]
    fn test_add_result_strides() {
        let a: Tensor =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::CPU).unwrap();
        let b: Tensor =
            Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3], Device::CPU).unwrap();

        let c: Tensor = a + b;

        assert_eq!(c.strides(), &[3, 1]);
    }

    #[test]
    #[should_panic(expected = "Cannot add tensors with different shapes")]
    fn test_add_shape_mismatch() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], Device::CPU).unwrap();

        let _c: Tensor = a + b;
    }

    #[test]
    #[should_panic(expected = "Cannot add tensors with different shapes")]
    fn test_add_dimension_mismatch() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        let _c: Tensor = a + b;
    }

    #[test]
    fn test_add_preserves_clone_semantics() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![3.0, 4.0], vec![2], Device::CPU).unwrap();

        let c: Tensor = a + b;
        let d: Tensor = c.clone();

        // c et d doivent partager le même Tensor_ interne
        assert!(Arc::ptr_eq(&c.0, &d.0));
    }

    #[test]
    fn test_add_multiple_operations() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![3.0, 4.0], vec![2], Device::CPU).unwrap();

        let c: Tensor = a.clone() + b.clone();
        let d: Tensor = c + a.clone();
        let e: Tensor = d + b;

        let result: Vec<f32> = e.to_cpu().unwrap();
        // (1+3) + 1 + 3 = 8
        // (2+4) + 2 + 4 = 12
        assert_eq!(result, vec![8.0, 12.0]);
    }

    #[test]
    fn test_add_with_zeros_constructor() {
        let a: Tensor = Tensor::new(vec![5.0, 10.0, 15.0], vec![3], Device::CPU).unwrap();
        let b: Tensor = Tensor::zeros(vec![3], Device::CPU).unwrap();

        let c: Tensor = a + b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_add_with_ones_constructor() {
        let a: Tensor = Tensor::ones(vec![3], Device::CPU).unwrap();
        let b: Tensor = Tensor::ones(vec![3], Device::CPU).unwrap();

        let c: Tensor = a + b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_add_operator_simple() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![4], Device::CPU).unwrap();

        let c: Tensor = a + b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_add_operator_2d() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        let c: Tensor = a + b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_consumes_operands() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![3.0, 4.0], vec![2], Device::CPU).unwrap();

        let c: Tensor = a + b;

        assert_eq!(c.to_cpu().unwrap(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_mul_chain() {
        let a: Tensor = Tensor::new(vec![2.0, 3.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![4.0, 5.0], vec![2], Device::CPU).unwrap();
        let c: Tensor = Tensor::new(vec![10.0, 10.0], vec![2], Device::CPU).unwrap();

        let result: Tensor = a * b * c;

        let data: Vec<f32> = result.to_cpu().unwrap();
        // (2*4)*10 = 80
        // (3*5)*10 = 150
        assert_eq!(data, vec![80.0, 150.0]);
    }

    #[test]
    fn test_mul_same_tensor() {
        let a: Tensor = Tensor::new(vec![2.0, 3.0, 4.0], vec![3], Device::CPU).unwrap();
        let b: Tensor = a.clone();

        let result: Tensor = a * b;

        let data: Vec<f32> = result.to_cpu().unwrap();
        assert_eq!(data, vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_mul_result_device() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![3.0, 4.0], vec![2], Device::CPU).unwrap();

        let c: Tensor = a * b;

        assert_eq!(*c.device(), Device::CPU);
    }

    #[test]
    fn test_mul_result_shape() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        let c: Tensor = a * b;

        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_mul_result_is_contiguous() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![4.0, 5.0, 6.0], vec![3], Device::CPU).unwrap();

        let c: Tensor = a * b;

        assert!(c.is_contiguous());
    }

    #[test]
    fn test_mul_result_strides() {
        let a: Tensor =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::CPU).unwrap();
        let b: Tensor =
            Tensor::new(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0], vec![2, 3], Device::CPU).unwrap();

        let c: Tensor = a * b;

        assert_eq!(c.strides(), &[3, 1]);
    }

    #[test]
    #[should_panic(expected = "Cannot add tensors with different shapes")]
    fn test_mul_shape_mismatch() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], Device::CPU).unwrap();

        let _c: Tensor = a * b;
    }

    #[test]
    #[should_panic(expected = "Cannot add tensors with different shapes")]
    fn test_mul_dimension_mismatch() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        let _c: Tensor = a * b;
    }

    #[test]
    fn test_mul_preserves_clone_semantics() {
        let a: Tensor = Tensor::new(vec![2.0, 3.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![4.0, 5.0], vec![2], Device::CPU).unwrap();

        let c: Tensor = a * b;
        let d: Tensor = c.clone();

        // c et d doivent partager le même Tensor_ interne
        assert!(Arc::ptr_eq(&c.0, &d.0));
    }

    #[test]
    fn test_mul_multiple_operations() {
        let a: Tensor = Tensor::new(vec![2.0, 3.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![4.0, 5.0], vec![2], Device::CPU).unwrap();

        let c: Tensor = a.clone() * b.clone();
        let d: Tensor = c * a.clone();
        let e: Tensor = d * b;

        let result: Vec<f32> = e.to_cpu().unwrap();
        // (2*4)*2*4 = 64
        // (3*5)*3*5 = 225
        assert_eq!(result, vec![64.0, 225.0]);
    }

    #[test]
    fn test_mul_with_zeros_constructor() {
        let a: Tensor = Tensor::new(vec![5.0, 10.0, 15.0], vec![3], Device::CPU).unwrap();
        let b: Tensor = Tensor::zeros(vec![3], Device::CPU).unwrap();

        let c: Tensor = a * b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mul_with_ones_constructor() {
        let a: Tensor = Tensor::new(vec![7.0, 13.0, 21.0], vec![3], Device::CPU).unwrap();
        let b: Tensor = Tensor::ones(vec![3], Device::CPU).unwrap();

        let c: Tensor = a * b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![7.0, 13.0, 21.0]);
    }

    #[test]
    fn test_mul_operator_simple() {
        let a: Tensor = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![4], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![4], Device::CPU).unwrap();

        let c: Tensor = a * b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![20.0, 60.0, 120.0, 200.0]);
    }

    #[test]
    fn test_mul_operator_2d() {
        let a: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        let c: Tensor = a * b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_mul_consumes_operands() {
        let a: Tensor = Tensor::new(vec![2.0, 3.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![4.0, 5.0], vec![2], Device::CPU).unwrap();

        let c: Tensor = a * b;

        assert_eq!(c.to_cpu().unwrap(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_mul_negatives() {
        let a: Tensor = Tensor::new(vec![-2.0, 3.0, -4.0], vec![3], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![5.0, -2.0, -3.0], vec![3], Device::CPU).unwrap();

        let c: Tensor = a * b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![-10.0, -6.0, 12.0]);
    }

    #[test]
    fn test_mul_floats() {
        let a: Tensor = Tensor::new(vec![1.5, 2.5, 3.5], vec![3], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![2.0, 4.0, 0.5], vec![3], Device::CPU).unwrap();

        let c: Tensor = a * b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        let epsilon: f32 = 1e-6;
        assert!((result[0] - 3.0).abs() < epsilon);
        assert!((result[1] - 10.0).abs() < epsilon);
        assert!((result[2] - 1.75).abs() < epsilon);
    }

    #[test]
    fn test_mul_single_element() {
        let a: Tensor = Tensor::new(vec![7.0], vec![1], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![6.0], vec![1], Device::CPU).unwrap();

        let c: Tensor = a * b;

        let result: Vec<f32> = c.to_cpu().unwrap();
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_mul_mixed_operations_with_add() {
        let a: Tensor = Tensor::new(vec![2.0, 3.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![4.0, 5.0], vec![2], Device::CPU).unwrap();
        let c: Tensor = Tensor::new(vec![1.0, 1.0], vec![2], Device::CPU).unwrap();

        // (a * b) + c = (2*4) + 1 = 9, (3*5) + 1 = 16
        let result: Tensor = (a * b) + c;

        let data: Vec<f32> = result.to_cpu().unwrap();
        assert_eq!(data, vec![9.0, 16.0]);
    }

    #[test]
    fn test_add_mul_distributivity() {
        // Teste (a + b) * c vs (a * c) + (b * c)
        let a: Tensor = Tensor::new(vec![2.0, 3.0], vec![2], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![4.0, 5.0], vec![2], Device::CPU).unwrap();
        let c: Tensor = Tensor::new(vec![10.0, 10.0], vec![2], Device::CPU).unwrap();

        let result1: Tensor = (a.clone() + b.clone()) * c.clone();
        let result2: Tensor = (a * c.clone()) + (b * c);

        assert_eq!(result1.to_cpu().unwrap(), result2.to_cpu().unwrap());
    }

    #[test]
    fn test_mul_require_gradient_propagation() {
        let mut a_inner = Tensor::new(vec![2.0], vec![1], Device::CPU).unwrap();
        Arc::get_mut(&mut a_inner.0).unwrap().require_gradient = true;

        let b: Tensor = Tensor::new(vec![3.0], vec![1], Device::CPU).unwrap();

        let c: Tensor = a_inner * b;

        assert!(c.require_gradient);
    }

    #[test]
    fn test_mul_both_require_gradient() {
        let mut a_inner = Tensor::new(vec![2.0], vec![1], Device::CPU).unwrap();
        Arc::get_mut(&mut a_inner.0).unwrap().require_gradient = true;

        let mut b_inner = Tensor::new(vec![3.0], vec![1], Device::CPU).unwrap();
        Arc::get_mut(&mut b_inner.0).unwrap().require_gradient = true;

        let c: Tensor = a_inner * b_inner;

        assert!(c.require_gradient);
    }

    #[test]
    fn test_mul_no_gradient_propagation() {
        let a: Tensor = Tensor::new(vec![2.0], vec![1], Device::CPU).unwrap();
        let b: Tensor = Tensor::new(vec![3.0], vec![1], Device::CPU).unwrap();

        let c: Tensor = a * b;

        assert!(!c.require_gradient);
    }
}
