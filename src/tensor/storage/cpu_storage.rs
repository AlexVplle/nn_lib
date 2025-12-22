use crate::{
    error::NeuralNetworkError,
    tensor::{storage::storage::StorageBackend, Device},
};

use std::any::Any;

#[derive(PartialEq, Debug, Clone, Default, PartialOrd)]
pub struct CpuStorage {
    data: Box<[f32]>,
}

impl CpuStorage {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size].into_boxed_slice(),
        }
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        Self {
            data: data.into_boxed_slice(),
        }
    }

    pub fn filled(size: usize, value: f32) -> Self {
        Self {
            data: vec![value; size].into_boxed_slice(),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    fn to_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }
}

impl StorageBackend for CpuStorage {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn try_clone(&self) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        Ok(Box::new(CpuStorage {
            data: self.data.to_vec().into_boxed_slice(),
        }))
    }

    fn to_cpu(&self) -> Result<Vec<f32>, NeuralNetworkError> {
        Ok(self.data.to_vec())
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
    use super::*;

    #[test]
    fn test_from_vec() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let storage: CpuStorage = CpuStorage::from_vec(data);
        assert_eq!(&*storage.data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_vec_empty() {
        let data: Vec<f32> = vec![];
        let storage: CpuStorage = CpuStorage::from_vec(data);
        assert_eq!(&*storage.data, &[]);
    }

    #[test]
    fn test_as_slice() {
        let storage: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(storage.as_slice(), &*storage.data);
    }

    #[test]
    fn test_new() {
        let storage: CpuStorage = CpuStorage::new(5);
        assert_eq!(&*storage.data, &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_new_zero_size() {
        let storage: CpuStorage = CpuStorage::new(0);
        assert_eq!(&*storage.data, &[]);
    }

    #[test]
    fn test_filled() {
        let storage: CpuStorage = CpuStorage::filled(4, 3.0);
        assert_eq!(&*storage.data, &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_filled_zero_size() {
        let storage: CpuStorage = CpuStorage::filled(0, 1.0);
        assert_eq!(&*storage.data, &[]);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut storage: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let slice: &mut [f32] = storage.as_mut_slice();
        slice[0] = 10.0;
        slice[2] = 30.0;
        assert_eq!(&*storage.data, &[10.0, 2.0, 30.0]);
    }

    #[test]
    fn test_to_vec() {
        let storage: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let vec: Vec<f32> = storage.to_vec();
        assert_eq!(vec, storage.data.to_vec());
    }

    #[test]
    fn test_to_vec_independence() {
        let mut storage: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let vec1: Vec<f32> = storage.to_vec();
        storage.as_mut_slice()[0] = 99.0;
        let vec2: Vec<f32> = storage.to_vec();
        assert_eq!(vec1, vec![1.0, 2.0, 3.0]);
        assert_eq!(vec2, vec![99.0, 2.0, 3.0]);
    }

    #[test]
    fn test_len() {
        let storage1: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let storage2: CpuStorage = CpuStorage::from_vec(vec![]);
        let storage3: CpuStorage = CpuStorage::from_vec(vec![5.0; 10]);
        assert_eq!(storage1.len(), storage1.data.len());
        assert_eq!(storage2.len(), storage2.data.len());
        assert_eq!(storage3.len(), storage3.data.len());
    }

    #[test]
    fn test_device() {
        let storage: CpuStorage = CpuStorage::from_vec(vec![1.0]);
        assert_eq!(storage.device(), Device::CPU);
    }

    #[test]
    fn test_is_empty() {
        let empty: CpuStorage = CpuStorage::from_vec(vec![]);
        let non_empty: CpuStorage = CpuStorage::from_vec(vec![1.0]);
        assert!(empty.is_empty());
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_to_cpu() {
        let storage: CpuStorage = CpuStorage::from_vec(vec![1.5, 2.5, 3.5]);
        let result: Result<Vec<f32>, NeuralNetworkError> = storage.to_cpu();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), storage.data.to_vec());
    }

    #[test]
    fn test_to_cpu_empty() {
        let storage: CpuStorage = CpuStorage::from_vec(vec![]);
        let result: Result<Vec<f32>, NeuralNetworkError> = storage.to_cpu();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn test_try_clone() {
        let mut original: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let result: Result<Box<dyn StorageBackend>, NeuralNetworkError> = original.try_clone();
        assert!(result.is_ok());

        let cloned: Box<dyn StorageBackend> = result.unwrap();
        assert_eq!(cloned.to_cpu().unwrap(), original.data.to_vec());

        original.as_mut_slice()[0] = 99.0;
        assert_eq!(cloned.to_cpu().unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(original.data[0], 99.0);
    }

    #[test]
    fn test_clone() {
        let original: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let cloned: CpuStorage = original.clone();
        assert_eq!(original, cloned);
        assert_eq!(&*original.data, &*cloned.data);
    }

    #[test]
    fn test_partial_eq() {
        let storage1: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let storage2: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 3.0]);
        let storage3: CpuStorage = CpuStorage::from_vec(vec![1.0, 2.0, 4.0]);
        assert_eq!(storage1, storage2);
        assert_ne!(storage1, storage3);
    }

    #[test]
    fn test_default() {
        let storage: CpuStorage = CpuStorage::default();
        assert_eq!(&*storage.data, &[]);
    }

    #[test]
    fn test_large_storage() {
        let storage: CpuStorage = CpuStorage::new(1_000_000);
        assert_eq!(storage.data.len(), 1_000_000);
    }

    #[test]
    fn test_special_floats() {
        let storage: CpuStorage = CpuStorage::from_vec(vec![
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN,
            f32::MAX,
        ]);
        assert_eq!(storage.data[0], f32::INFINITY);
        assert_eq!(storage.data[1], f32::NEG_INFINITY);
        assert!(storage.data[2].is_nan());
        assert_eq!(storage.data[3], f32::MIN);
        assert_eq!(storage.data[4], f32::MAX);
    }
}
