use std::{
    ops::{Add, Deref, Mul, Range},
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::tensor::{
    backend::{backend_device::BackendDevice, cpu::CpuStorage},
    device::Device,
    storage::Storage,
    Layout, TensorError,
};
use ndarray::ArrayD;

#[derive(Debug)]
pub struct Tensor_ {
    storage: Arc<RwLock<Box<Storage>>>,
    layout: Layout,
    device: Device,
    gradient: Option<Tensor>,
    require_gradient: bool,
}

#[derive(Clone, Debug)]
pub struct Tensor(Arc<Tensor_>);

impl Deref for Tensor {
    type Target = Tensor_;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, device: Device) -> Result<Self, TensorError> {
        let size: usize = shape.iter().product();
        if data.len() != size {
            return Err(TensorError::IncompatibleShape {
                shape_given: shape,
                tensor_shape: vec![data.len()],
            });
        }

        let storage: Box<Storage> = match device {
            Device::CPU => Box::new(Storage::Cpu(CpuStorage(data))),
            Device::Metal(device) => {
                let storage = device.storage_from_vec(data)?;
                Box::new(Storage::Metal(storage))
            }
        };

        let device = storage.device();

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::new(shape),
            device,
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn contiguous(&self) -> Result<Self, TensorError> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let shape = self.shape().to_vec();
        let strides = self.layout.strides();
        let storage = self.storage();
        let new_storage = storage.copy_strided(&shape, strides)?;

        Ok(Self(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(new_storage))),
            layout: crate::tensor::Layout::new(shape),
            device: self.device.clone(),
            gradient: None,
            require_gradient: self.require_gradient,
        })))
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidDimension {
                got: self.ndim(),
                max_dimension: 2,
            });
        }
        let new_layout = self.layout.permute(&[1, 0])?;
        Ok(Self(Arc::new(Tensor_ {
            storage: Arc::clone(&self.storage),
            layout: new_layout,
            device: self.device.clone(),
            gradient: None,
            require_gradient: self.require_gradient,
        })))
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self, TensorError> {
        let new_layout = self.layout.permute(dims)?;
        Ok(Self(Arc::new(Tensor_ {
            storage: Arc::clone(&self.storage),
            layout: new_layout,
            device: self.device.clone(),
            gradient: None,
            require_gradient: self.require_gradient,
        })))
    }

    pub fn slice(&self, dimension: usize, range: Range<usize>) -> Result<Self, TensorError> {
        // For 2D tensors with row-slicing on dimension 0, create a contiguous copy
        // This is necessary because matmul and other ops expect contiguous storage
        let old_shape = self.shape();

        if old_shape.len() == 2 && dimension == 0 {
            // Slicing rows - this is the common case for batching
            let row_size = old_shape[1];
            let num_rows = range.end - range.start;
            let new_shape = vec![num_rows, row_size];

            // Check for overflow before allocating
            let total_size = num_rows
                .checked_mul(row_size)
                .ok_or(TensorError::DimensionMismatch)?;

            // Get data once
            let old_data = self.to_vec()?;

            // Extract the sliced rows
            let mut new_data = Vec::with_capacity(total_size);
            for i in range.start..range.end {
                let start_idx = i * row_size;
                let end_idx = start_idx + row_size;
                new_data.extend_from_slice(&old_data[start_idx..end_idx]);
            }

            Self::new(new_data, new_shape, self.device.clone())
        } else {
            // General case (less efficient but correct)
            // This is more complex, for now we'll handle only the row-slicing case
            // which is what we need for batching
            Err(TensorError::InvalidDimension {
                got: dimension,
                max_dimension: 0,
            })
        }
    }

    pub fn storage(&self) -> RwLockReadGuard<'_, Box<Storage>> {
        self.storage.read().unwrap()
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn to_vec(&self) -> Result<Vec<f32>, TensorError> {
        if self.is_contiguous() {
            let storage = self.storage();
            let cpu_storage = storage.to_cpu_storage()?;
            Ok(cpu_storage.0)
        } else {
            let contiguous = self.contiguous()?;
            let storage = contiguous.storage();
            let cpu_storage = storage.to_cpu_storage()?;
            Ok(cpu_storage.0)
        }
    }

    /// Transfer tensor to a different device
    pub fn to_device(&self, target_device: Device) -> Result<Self, TensorError> {
        // If already on target device, return clone
        if self.device.same_device(&target_device) {
            return Ok(self.clone());
        }

        // Transfer via CPU
        let data = self.to_vec()?;
        let shape = self.shape().to_vec();

        Self::new(data, shape, target_device)
    }

    pub fn relu(&self) -> Result<Self, TensorError> {
        let storage = self.storage();
        let result_storage = storage.relu()?;

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: self.layout.clone(),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn tanh(&self) -> Result<Self, TensorError> {
        let storage = self.storage();
        let result_storage = storage.tanh()?;

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: self.layout.clone(),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn sigmoid(&self) -> Result<Self, TensorError> {
        let storage = self.storage();
        let result_storage = storage.sigmoid()?;

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: self.layout.clone(),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn softmax(&self) -> Result<Self, TensorError> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(TensorError::InvalidDimension {
                got: shape.len(),
                max_dimension: 2,
            });
        }

        let batch_size = shape[0];
        let vector_size = shape[1..].iter().product();

        let storage = self.storage();
        let result_storage = storage.softmax(batch_size, vector_size)?;

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: self.layout.clone(),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        let lhs_storage = self.storage();
        let rhs_storage = rhs.storage();
        let result_storage = lhs_storage.sub(&*rhs_storage)?;

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: self.layout.clone(),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        let storage = self.storage();
        let result_storage = storage.mul_scalar(scalar)?;

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: self.layout.clone(),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.ndim() != 2 || rhs.ndim() != 2 {
            return Err(TensorError::InvalidDimension {
                got: self.ndim().max(rhs.ndim()),
                max_dimension: 2,
            });
        }

        let m = self.shape()[0];
        let k = self.shape()[1];
        let k_rhs = rhs.shape()[0];
        let n = rhs.shape()[1];

        if k != k_rhs {
            return Err(TensorError::DimensionMismatch);
        }

        let lhs_storage = self.storage();
        let rhs_storage = rhs.storage();
        let lhs_strides = self.layout.strides();
        let rhs_strides = rhs.layout.strides();
        let result_storage = lhs_storage
            .matmul(&*rhs_storage, m, k, n, lhs_strides, rhs_strides)
            .unwrap_or_else(|e| panic!("Tensor matmul failed: {}", e));

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: Layout::new(vec![m, n]),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn sum_axis(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.ndim() {
            return Err(TensorError::InvalidDimension {
                got: axis,
                max_dimension: self.ndim(),
            });
        }

        let input_shape: &[usize] = self.shape();
        let mut output_shape: Vec<usize> = Vec::with_capacity(input_shape.len() - 1);
        for (i, &dim) in input_shape.iter().enumerate() {
            if i != axis {
                output_shape.push(dim);
            }
        }

        let storage = self.storage();
        let result_storage = storage.sum_axis(axis, input_shape, &output_shape)?;

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: Layout::new(output_shape),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        })))
    }
}

impl Add for Tensor {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let lhs_storage = self.storage();
        let rhs_storage = other.storage();
        let result_storage = lhs_storage
            .add(&rhs_storage)
            .unwrap_or_else(|e| panic!("Tensor addition failed: {}", e));

        Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: self.layout.clone(),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        }))
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // Multiplication élément par élément (Hadamard product)
        let lhs_storage = self.storage();
        let rhs_storage = other.storage();

        // Les deux tenseurs doivent avoir la même forme
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shapes must match for element-wise multiplication"
        );

        let result_storage = lhs_storage
            .mul(&*rhs_storage)
            .unwrap_or_else(|e| panic!("Tensor multiplication failed: {}", e));

        Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(Box::new(result_storage))),
            layout: self.layout.clone(),
            device: self.device.clone(),
            gradient: None,
            require_gradient: false,
        }))
    }
}

impl From<ArrayD<f64>> for Tensor {
    fn from(arr: ArrayD<f64>) -> Self {
        let data: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
        let shape: Vec<usize> = arr.shape().to_vec();
        Tensor::new(data, shape, Device::CPU).unwrap()
    }
}

impl From<Tensor> for ArrayD<f64> {
    fn from(tensor: Tensor) -> Self {
        let data = tensor.to_vec().unwrap();
        let shape = tensor.shape().to_vec();
        ArrayD::from_shape_vec(shape, data.iter().map(|&x| x as f64).collect()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new_cpu() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data.clone(), shape.clone(), Device::CPU).unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.strides(), &[3, 1]);
        assert_eq!(tensor.ndim(), 2);
        assert!(tensor.is_contiguous());
        assert!(matches!(tensor.device(), Device::CPU));
    }

    #[test]
    fn test_tensor_new_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data, shape, device).unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.strides(), &[3, 1]);
        assert_eq!(tensor.ndim(), 2);
        assert!(tensor.is_contiguous());
        assert!(matches!(tensor.device(), Device::Metal(_)));
    }

    #[test]
    fn test_tensor_new_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![5];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        assert_eq!(tensor.shape(), &[5]);
        assert_eq!(tensor.strides(), &[1]);
        assert_eq!(tensor.ndim(), 1);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_tensor_new_3d() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shape = vec![2, 3, 4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.strides(), &[12, 4, 1]);
        assert_eq!(tensor.ndim(), 3);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_tensor_new_3d_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shape = vec![2, 3, 4];
        let tensor = Tensor::new(data, shape, device).unwrap();

        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.strides(), &[12, 4, 1]);
        assert_eq!(tensor.ndim(), 3);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_tensor_new_incompatible_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 3];

        let result = Tensor::new(data, shape, Device::CPU);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TensorError::IncompatibleShape { .. }
        ));
    }

    #[test]
    fn test_tensor_new_incompatible_shape_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 3];

        let result = Tensor::new(data, shape, device);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TensorError::IncompatibleShape { .. }
        ));
    }

    #[test]
    fn test_tensor_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_tensor_transpose_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data, shape, device).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_tensor_transpose_invalid_dimension() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();
        let result = tensor.transpose();

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TensorError::InvalidDimension { .. }
        ));
    }

    #[test]
    fn test_tensor_permute_3d() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shape = vec![2, 3, 4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();
        let permuted = tensor.permute(&[2, 0, 1]).unwrap();

        assert_eq!(permuted.shape(), &[4, 2, 3]);
        assert_eq!(permuted.strides(), &[1, 12, 4]);
        assert!(!permuted.is_contiguous());
    }

    #[test]
    fn test_tensor_permute_3d_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shape = vec![2, 3, 4];
        let tensor = Tensor::new(data, shape, device).unwrap();
        let permuted = tensor.permute(&[2, 0, 1]).unwrap();

        assert_eq!(permuted.shape(), &[4, 2, 3]);
        assert_eq!(permuted.strides(), &[1, 12, 4]);
        assert!(!permuted.is_contiguous());
    }

    #[test]
    fn test_tensor_permute_invalid_dimensions() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shape = vec![2, 3, 4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        let result = tensor.permute(&[0, 1]);
        assert!(result.is_err());

        let result = tensor.permute(&[0, 1, 5]);
        assert!(result.is_err());

        let result = tensor.permute(&[0, 1, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_slice_first_dimension() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let shape = vec![5, 4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();
        let sliced = tensor.slice(0, 1..3).unwrap();

        assert_eq!(sliced.shape(), &[2, 4]);
        assert_eq!(sliced.strides(), &[4, 1]);
        assert!(sliced.is_contiguous());
    }

    #[test]
    fn test_tensor_slice_first_dimension_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let shape = vec![5, 4];
        let tensor = Tensor::new(data, shape, device).unwrap();
        let sliced = tensor.slice(0, 1..3).unwrap();

        assert_eq!(sliced.shape(), &[2, 4]);
        assert_eq!(sliced.strides(), &[4, 1]);
        assert!(sliced.is_contiguous());
    }

    #[test]
    fn test_tensor_slice_last_dimension() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let shape = vec![5, 4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();
        let sliced = tensor.slice(1, 1..3).unwrap();

        assert_eq!(sliced.shape(), &[5, 2]);
        assert_eq!(sliced.strides(), &[4, 1]);
        assert!(!sliced.is_contiguous());
    }

    #[test]
    fn test_tensor_slice_invalid_dimension() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let shape = vec![5, 4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        let result = tensor.slice(2, 0..2);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_slice_out_of_bounds() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let shape = vec![5, 4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        let result = tensor.slice(0, 0..10);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_clone() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();
        let cloned = tensor.clone();

        assert_eq!(tensor.shape(), cloned.shape());
        assert_eq!(tensor.strides(), cloned.strides());
    }

    #[test]
    fn test_tensor_storage_sharing() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor1 = Tensor::new(data, shape, Device::CPU).unwrap();
        let tensor2 = tensor1.transpose().unwrap();

        assert_eq!(Arc::as_ptr(&tensor1.storage), Arc::as_ptr(&tensor2.storage));
    }

    #[test]
    fn test_tensor_storage_sharing_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor1 = Tensor::new(data, shape, device).unwrap();
        let tensor2 = tensor1.transpose().unwrap();

        assert_eq!(Arc::as_ptr(&tensor1.storage), Arc::as_ptr(&tensor2.storage));
    }

    #[test]
    fn test_tensor_complex_operations() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shape = vec![2, 3, 4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        let permuted = tensor.permute(&[1, 0, 2]).unwrap();
        assert_eq!(permuted.shape(), &[3, 2, 4]);

        let sliced = permuted.slice(0, 1..3).unwrap();
        assert_eq!(sliced.shape(), &[2, 2, 4]);
    }

    #[test]
    fn test_tensor_complex_operations_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shape = vec![2, 3, 4];
        let tensor = Tensor::new(data, shape, device).unwrap();

        let permuted = tensor.permute(&[1, 0, 2]).unwrap();
        assert_eq!(permuted.shape(), &[3, 2, 4]);

        let sliced = permuted.slice(0, 1..3).unwrap();
        assert_eq!(sliced.shape(), &[2, 2, 4]);
    }

    #[test]
    fn test_tensor_multiple_slices() {
        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        let shape = vec![5, 4, 3];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        let sliced1 = tensor.slice(0, 1..4).unwrap();
        assert_eq!(sliced1.shape(), &[3, 4, 3]);

        let sliced2 = sliced1.slice(1, 1..3).unwrap();
        assert_eq!(sliced2.shape(), &[3, 2, 3]);

        let sliced3 = sliced2.slice(2, 0..2).unwrap();
        assert_eq!(sliced3.shape(), &[3, 2, 2]);
    }

    #[test]
    fn test_tensor_multiple_slices_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        let shape = vec![5, 4, 3];
        let tensor = Tensor::new(data, shape, device).unwrap();

        let sliced1 = tensor.slice(0, 1..4).unwrap();
        assert_eq!(sliced1.shape(), &[3, 4, 3]);

        let sliced2 = sliced1.slice(1, 1..3).unwrap();
        assert_eq!(sliced2.shape(), &[3, 2, 3]);

        let sliced3 = sliced2.slice(2, 0..2).unwrap();
        assert_eq!(sliced3.shape(), &[3, 2, 2]);
    }

    #[test]
    fn test_tensor_is_contiguous_after_operations() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let shape = vec![3, 4];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        assert!(tensor.is_contiguous());

        let transposed = tensor.transpose().unwrap();
        assert!(!transposed.is_contiguous());

        let sliced_first = tensor.slice(0, 0..2).unwrap();
        assert!(sliced_first.is_contiguous());

        let sliced_last = tensor.slice(1, 0..2).unwrap();
        assert!(!sliced_last.is_contiguous());
    }

    #[test]
    fn test_tensor_device_preserved() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        let transposed = tensor.transpose().unwrap();
        assert!(matches!(transposed.device(), Device::CPU));

        let permuted = tensor.permute(&[1, 0]).unwrap();
        assert!(matches!(permuted.device(), Device::CPU));

        let sliced = tensor.slice(0, 0..1).unwrap();
        assert!(matches!(sliced.device(), Device::CPU));
    }

    #[test]
    fn test_tensor_device_preserved_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, shape, device).unwrap();

        let transposed = tensor.transpose().unwrap();
        assert!(matches!(transposed.device(), Device::Metal(_)));

        let permuted = tensor.permute(&[1, 0]).unwrap();
        assert!(matches!(permuted.device(), Device::Metal(_)));

        let sliced = tensor.slice(0, 0..1).unwrap();
        assert!(matches!(sliced.device(), Device::Metal(_)));
    }

    #[test]
    fn test_tensor_large_tensor() {
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let shape = vec![size];
        let tensor = Tensor::new(data, shape, Device::CPU).unwrap();

        assert_eq!(tensor.shape(), &[size]);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_tensor_large_tensor_metal() {
        let device = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device = device.unwrap();

        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let shape = vec![size];
        let tensor = Tensor::new(data, shape, device).unwrap();

        assert_eq!(tensor.shape(), &[size]);
        assert!(tensor.is_contiguous());
    }
}
