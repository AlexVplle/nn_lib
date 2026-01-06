use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

use crate::{
    error::NeuralNetworkError,
    tensor::{storage::StorageBackend, Device},
};

use std::{any::Any, ptr::NonNull};

#[derive(PartialEq, Debug, Clone)]
pub struct MetalStorage {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    device_id: usize,
    len: usize,
}

impl MetalStorage {
    pub fn new(device_id: usize, size: usize) -> Result<Self, NeuralNetworkError> {
        if size == 0 {
            return Err(NeuralNetworkError::EmptyTensorNotAllowed);
        }
        let device = MTLCreateSystemDefaultDevice().ok_or(NeuralNetworkError::AllocationFailed("Metal device not found".to_string()))?;
        let byte_size = (size * std::mem::size_of::<f32>()) as NSUInteger;
        let buffer = device
            .newBufferWithLength_options(byte_size, MTLResourceOptions::StorageModeShared)
            .ok_or(NeuralNetworkError::AllocationFailed("Metal buffer creation failed".to_string()))?;
        Ok(Self {
            buffer,
            device_id,
            len: size,
        })
    }

    pub fn from_vec(device_id: usize, data: Vec<f32>) -> Result<Self, NeuralNetworkError> {
        let len = data.len();
        if len == 0 {
            return Err(NeuralNetworkError::EmptyTensorNotAllowed);
        }
        let device =
            MTLCreateSystemDefaultDevice().ok_or(NeuralNetworkError::AllocationFailed("Metal device not found".to_string()))?;
        let byte_size = (len * std::mem::size_of::<f32>()) as NSUInteger;
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
            .ok_or(NeuralNetworkError::AllocationFailed("Null pointer".to_string()))?;
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                ptr,
                byte_size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(NeuralNetworkError::AllocationFailed("Metal buffer creation failed".to_string()))?;
        Ok(Self {
            buffer,
            device_id,
            len,
        })
    }

    pub fn filled(device_id: usize, size: usize, value: f32) -> Result<Self, NeuralNetworkError> {
        if size == 0 {
            return Err(NeuralNetworkError::EmptyTensorNotAllowed);
        }
        let data: Vec<f32> = vec![value; size];
        Self::from_vec(device_id, data)
    }

    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

impl StorageBackend for MetalStorage {
    fn len(&self) -> usize {
        self.len
    }

    fn device(&self) -> Device {
        Device::Metal(self.device_id)
    }

    fn try_clone(&self) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        let data = self.to_cpu()?;
        let storage = Self::from_vec(self.device_id, data)?;
        Ok(Box::new(storage))
    }

    fn to_cpu(&self) -> Result<Vec<f32>, NeuralNetworkError> {
        let contents = self.buffer.contents();
        let ptr = contents.as_ptr() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len()) };
        Ok(slice.to_vec())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::f32;

    use super::*;

    #[test]
    fn test_metal_storage_new() {
        let storage = MetalStorage::new(0, 100).unwrap();
        assert_eq!(storage.len(), 100);
        assert_eq!(storage.device(), Device::Metal(0));
    }

    #[test]
    fn test_metal_storage_new_different_device_id() {
        let storage = MetalStorage::new(1, 50).unwrap();
        assert_eq!(storage.len(), 50);
        assert_eq!(storage.device(), Device::Metal(1));
    }

    #[test]
    fn test_metal_storage_new_large() {
        let size = 1_000_000;
        let storage = MetalStorage::new(0, size).unwrap();
        assert_eq!(storage.len(), size);
    }

    #[test]
    fn test_metal_storage_from_vec_simple() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        assert_eq!(storage.len(), 4);
        assert_eq!(storage.to_cpu().unwrap(), data);
    }

    #[test]
    fn test_metal_storage_from_vec_single_element() {
        let data = vec![42.0];
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        assert_eq!(storage.len(), 1);
        assert_eq!(storage.to_cpu().unwrap(), data);
    }

    #[test]
    fn test_metal_storage_from_vec_zeros() {
        let data = vec![0.0; 10];
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        assert_eq!(storage.to_cpu().unwrap(), data);
    }

    #[test]
    fn test_metal_storage_from_vec_negatives() {
        let data = vec![-1.0, -2.0, -3.0, 4.0, 5.0];
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        assert_eq!(storage.to_cpu().unwrap(), data);
    }

    #[test]
    fn test_metal_storage_from_vec_floats() {
        let data = vec![1.5, 2.7, f32::consts::PI, 0.5];
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        let result = storage.to_cpu().unwrap();
        for (a, b) in result.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_metal_storage_from_vec_large() {
        let size = 100_000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        assert_eq!(storage.len(), size);
        assert_eq!(storage.to_cpu().unwrap(), data);
    }

    #[test]
    fn test_metal_storage_to_cpu() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        let result = storage.to_cpu().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_metal_storage_to_cpu_multiple_times() {
        let data = vec![10.0, 20.0, 30.0];
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();
        assert_eq!(storage.to_cpu().unwrap(), data);
        assert_eq!(storage.to_cpu().unwrap(), data);
        assert_eq!(storage.to_cpu().unwrap(), data);
    }

    #[test]
    fn test_metal_storage_try_clone() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        let cloned = storage.try_clone().unwrap();

        assert_eq!(cloned.len(), storage.len());
        assert_eq!(cloned.to_cpu().unwrap(), storage.to_cpu().unwrap());
    }

    #[test]
    fn test_metal_storage_try_clone_independence() {
        let data = vec![5.0, 10.0, 15.0];
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        let cloned = storage.try_clone().unwrap();

        assert_eq!(storage.to_cpu().unwrap(), cloned.to_cpu().unwrap());

        assert_eq!(cloned.len(), 3);
    }

    #[test]
    fn test_metal_storage_device_id() {
        let storage0 = MetalStorage::new(0, 10).unwrap();
        let storage1 = MetalStorage::new(1, 10).unwrap();

        assert_eq!(storage0.device(), Device::Metal(0));
        assert_eq!(storage1.device(), Device::Metal(1));
    }

    #[test]
    fn test_metal_storage_len() {
        let sizes = vec![1, 10, 100, 1000, 10000];

        for size in sizes {
            let storage = MetalStorage::new(0, size).unwrap();
            assert_eq!(storage.len(), size);
        }
    }

    #[test]
    fn test_metal_storage_as_any() {
        let storage = MetalStorage::new(0, 10).unwrap();
        let any = storage.as_any();

        assert!(any.downcast_ref::<MetalStorage>().is_some());
    }

    #[test]
    fn test_metal_storage_as_any_mut() {
        let mut storage = MetalStorage::new(0, 10).unwrap();
        let any_mut = storage.as_any_mut();

        assert!(any_mut.downcast_mut::<MetalStorage>().is_some());
    }

    #[test]
    fn test_metal_storage_buffer_access() {
        let storage = MetalStorage::new(0, 100).unwrap();
        let _buffer = storage.buffer();

        assert!(true);
    }

    #[test]
    fn test_metal_storage_roundtrip() {
        let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // CPU -> Metal
        let storage = MetalStorage::from_vec(0, original_data.clone()).unwrap();

        // Metal -> CPU
        let retrieved_data = storage.to_cpu().unwrap();

        assert_eq!(retrieved_data, original_data);
    }

    #[test]
    fn test_metal_storage_precision() {
        let data = vec![
            std::f32::consts::PI,
            std::f32::consts::E,
            std::f32::consts::SQRT_2,
        ];

        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();
        let result = storage.to_cpu().unwrap();

        for (a, b) in result.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_metal_storage_extreme_values() {
        let data = vec![f32::MAX, f32::MIN, f32::MIN_POSITIVE, -f32::MAX, 0.0, -0.0];

        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();
        let result = storage.to_cpu().unwrap();

        assert_eq!(result.len(), data.len());
        for i in 0..data.len() {
            if data[i].is_nan() {
                assert!(result[i].is_nan());
            } else {
                assert_eq!(result[i], data[i]);
            }
        }
    }

    #[test]
    fn test_metal_storage_performance_large_buffer() {
        let size = 10_000_000;
        let data: Vec<f32> = vec![1.0; size];

        let start = std::time::Instant::now();
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();
        let creation_time = start.elapsed();

        let start = std::time::Instant::now();
        let _result = storage.to_cpu().unwrap();
        let retrieval_time = start.elapsed();

        println!("Creation (10M floats): {:?}", creation_time);
        println!("Retrieval (10M floats): {:?}", retrieval_time);

        assert_eq!(storage.len(), size);
    }

    #[test]
    fn test_metal_storage_sequential_access() {
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let storage = MetalStorage::from_vec(0, data.clone()).unwrap();

        let result = storage.to_cpu().unwrap();

        for i in 0..size {
            assert_eq!(result[i], i as f32);
        }
    }
}
