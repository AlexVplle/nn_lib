use std::sync::{RwLock, Arc};

use crate::tensor::{Device, storage::{cpu_storage::CpuStorage, storage::StorageBackend}, backend::Backend};
use crate::error::NeuralNetworkError;

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn add(&self, lhs: &Arc<RwLock<Box<dyn StorageBackend>>>, rhs: &Arc<RwLock<Box<dyn StorageBackend>>>) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        let lhs_storage = lhs.read().expect("Lock poisoned");
        let rhs_storage = rhs.read().expect("Lock poisoned");

        let lhs_cpu = lhs_storage.as_any().downcast_ref::<CpuStorage>().expect("CpuBackend expects CPU storage");
        let rhs_cpu = rhs_storage.as_any().downcast_ref::<CpuStorage>().expect("CpuBackend expects CPU storage");

        let lhs_data = lhs_cpu.as_slice();
        let rhs_data = rhs_cpu.as_slice();

        let result: Vec<f32> = lhs_data.into_iter().zip(rhs_data.into_iter()).map(|(a, b)| a + b).collect();

        Ok(Box::new(CpuStorage::from_vec(result)))
    }

    fn relu(&self, storage: &Arc<RwLock<Box<dyn StorageBackend>>>) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        let storage = storage.read().expect("Lock poisoned");

        let storage_cpu = storage.as_any().downcast_ref::<CpuStorage>().expect("CpuBackend expects CPU storage");

        let storage_data = storage_cpu.as_slice();

        let result: Vec<f32> = storage_data.into_iter().map(|a| a.max(0.0)).collect();

        Ok(Box::new(CpuStorage::from_vec(result)))
    }

    fn tanh(&self, storage: &Arc<RwLock<Box<dyn StorageBackend>>>) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        let storage = storage.read().expect("Lock poisoned");

        let storage_cpu = storage.as_any().downcast_ref::<CpuStorage>().expect("CpuBackend expects CPU storage");

        let storage_data = storage_cpu.as_slice();

        let result: Vec<f32> = storage_data.into_iter().map(|a| a.tanh()).collect();

        Ok(Box::new(CpuStorage::from_vec(result)))
    }

    fn sigmoid(&self, storage: &Arc<RwLock<Box<dyn StorageBackend>>>) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        let storage = storage.read().expect("Lock poisoned");

        let storage_cpu = storage.as_any().downcast_ref::<CpuStorage>().expect("CpuBackend expects CPU storage");

        let storage_data = storage_cpu.as_slice();

        let result: Vec<f32> = storage_data.into_iter().map(|a| 1.0 / (1.0 + (-a).exp())).collect();

        Ok(Box::new(CpuStorage::from_vec(result)))
    }

    fn softmax(&self, storage: &Arc<RwLock<Box<dyn StorageBackend>>>) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        let storage = storage.read().expect("Lock poisoned");

        let storage_cpu = storage.as_any().downcast_ref::<CpuStorage>().expect("CpuBackend expects CPU storage");

        let storage_data = storage_cpu.as_slice();

        let max_val = storage_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = storage_data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp_values: f32 = exp_values.iter().sum();
        let result: Vec<f32> = exp_values.iter().map(|&x| x / sum_exp_values).collect();

        Ok(Box::new(CpuStorage::from_vec(result)))
    }

    fn device(&self) -> Device {
        Device::CPU
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_add_simple() {
        let backend: CpuBackend = CpuBackend;
        
        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![10.0, 20.0, 30.0, 40.0]))));
        
        let result: Box<dyn StorageBackend> = backend.add(&lhs_storage, &rhs_storage).unwrap();
        
        let result_cpu: &CpuStorage = result.as_any()
            .downcast_ref::<CpuStorage>()
            .expect("Result should be CpuStorage");
        
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_cpu_backend_add_zeros() {
        let backend: CpuBackend = CpuBackend;
        
        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![5.0, 10.0, 15.0]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![0.0, 0.0, 0.0]))));
        
        let result: Box<dyn StorageBackend> = backend.add(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_cpu_backend_add_negatives() {
        let backend: CpuBackend = CpuBackend;
        
        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![10.0, 5.0, -3.0]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-5.0, -5.0, 8.0]))));
        
        let result: Box<dyn StorageBackend> = backend.add(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[5.0, 0.0, 5.0]);
    }

    #[test]
    fn test_cpu_backend_add_large() {
        let backend: CpuBackend = CpuBackend;
        
        let size: usize = 1000;
        let lhs_data: Vec<f32> = (0..size).map(|i: usize| i as f32).collect();
        let rhs_data: Vec<f32> = (0..size).map(|i: usize| (i * 2) as f32).collect();
        
        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(lhs_data.clone()))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(rhs_data.clone()))));
        
        let result: Box<dyn StorageBackend> = backend.add(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        let expected: Vec<f32> = (0..size).map(|i: usize| (i * 3) as f32).collect();
        assert_eq!(result_data, &expected[..]);
    }

    #[test]
    fn test_cpu_backend_device() {
        let backend: CpuBackend = CpuBackend;
        let device: Device = backend.device();
        assert_eq!(device, Device::CPU);
    }

    #[test]
    fn test_cpu_backend_add_single_element() {
        let backend: CpuBackend = CpuBackend;
        
        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![42.0]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![8.0]))));
        
        let result: Box<dyn StorageBackend> = backend.add(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[50.0]);
    }

    #[test]
    fn test_cpu_backend_add_floats() {
        let backend: CpuBackend = CpuBackend;
        
        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![1.5, 2.7, 3.3]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![0.5, 0.3, 0.7]))));
        
        let result: Box<dyn StorageBackend> = backend.add(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        let epsilon: f32 = 1e-6;
        assert!((result_data[0] - 2.0).abs() < epsilon);
        assert!((result_data[1] - 3.0).abs() < epsilon);
        assert!((result_data[2] - 4.0).abs() < epsilon);
    }

     #[test]
    fn test_cpu_backend_relu_simple() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        
        let result_cpu: &CpuStorage = result.as_any()
            .downcast_ref::<CpuStorage>()
            .expect("Result should be CpuStorage");
        
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_cpu_backend_relu_all_positive() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![1.0, 2.0, 3.0, 4.0]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cpu_backend_relu_all_negative() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-5.0, -3.0, -1.0, -10.0]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cpu_backend_relu_zeros() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![0.0, 0.0, 0.0]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cpu_backend_relu_mixed() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![5.0, -3.0, 0.0, -1.0, 10.0, -7.0]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[5.0, 0.0, 0.0, 0.0, 10.0, 0.0]);
    }

    #[test]
    fn test_cpu_backend_relu_single_element() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![42.0]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[42.0]);
    }

    #[test]
    fn test_cpu_backend_relu_single_negative() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-42.0]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[0.0]);
    }

    #[test]
    fn test_cpu_backend_relu_large() {
        let backend: CpuBackend = CpuBackend;
        
        let size: usize = 1000;
        let input_data: Vec<f32> = (0..size).map(|i: usize| (i as f32) - 500.0).collect();
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(input_data.clone()))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        let expected: Vec<f32> = input_data.iter().map(|&x| x.max(0.0)).collect();
        assert_eq!(result_data, &expected[..]);
    }

    #[test]
    fn test_cpu_backend_relu_floats() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![1.5, -2.7, 0.0, -0.3, 3.14]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        let epsilon: f32 = 1e-6;
        assert!((result_data[0] - 1.5).abs() < epsilon);
        assert!((result_data[1] - 0.0).abs() < epsilon);
        assert!((result_data[2] - 0.0).abs() < epsilon);
        assert!((result_data[3] - 0.0).abs() < epsilon);
        assert!((result_data[4] - 3.14).abs() < epsilon);
    }

    #[test]
    fn test_cpu_backend_relu_very_small_values() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-0.0001, 0.0001, -0.00001]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data[0], 0.0);
        assert_eq!(result_data[1], 0.0001);
        assert_eq!(result_data[2], 0.0);
    }

    #[test]
    fn test_cpu_backend_relu_large_values() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![1000.0, -1000.0, 5000.0]))));
        
        let result: Box<dyn StorageBackend> = backend.relu(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        
        assert_eq!(result_data, &[1000.0, 0.0, 5000.0]);
    }

    #[test]
    fn test_cpu_backend_tanh_simple() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]))));
        
        let result: Box<dyn StorageBackend> = backend.tanh(&storage).unwrap();
        
        let result_cpu: &CpuStorage = result.as_any()
            .downcast_ref::<CpuStorage>()
            .expect("Result should be CpuStorage");
        
        let result_data: &[f32] = result_cpu.as_slice();
        
        let expected: Vec<f32> = vec![-0.96402758, -0.76159416, 0.0, 0.76159416, 0.96402758];
        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }
    }

    #[test]
    fn test_cpu_backend_tanh_large_values() {
        let backend: CpuBackend = CpuBackend;

        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![1000.0, -1000.0, 5000.0, -5000.0]))));
        let result: Box<dyn StorageBackend> = backend.tanh(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        let expected: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0];
        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }
    }

    #[test]
    fn test_cpu_backend_tanh_small_values() {
        let backend: CpuBackend = CpuBackend;
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-0.0001, 0.0001, -0.00001]))));
        let result: Box<dyn StorageBackend> = backend.tanh(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        let expected: Vec<f32> = vec![-0.0001, 0.0001, -0.00001];
        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }
    }

    #[test]
    fn test_cpu_backend_tanh_zeros() {
        let backend: CpuBackend = CpuBackend;
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![0.0, 0.0, 0.0]))));
        let result: Box<dyn StorageBackend> = backend.tanh(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        assert_eq!(result_data, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cpu_backend_sigmoid_simple() {
        let backend: CpuBackend = CpuBackend;
        
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]))));
        
        let result: Box<dyn StorageBackend> = backend.sigmoid(&storage).unwrap();
        
        let result_cpu: &CpuStorage = result.as_any()
            .downcast_ref::<CpuStorage>()
            .expect("Result should be CpuStorage");
        
        let result_data: &[f32] = result_cpu.as_slice();
        
        let expected: Vec<f32> = vec![0.11920292, 0.26894143, 0.5, 0.73105858, 0.88079708];
        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }
    }

    #[test]
    fn test_cpu_backend_sigmoid_large_values() {
        let backend: CpuBackend = CpuBackend;
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![1000.0, -1000.0, 5000.0, -5000.0]))));
        let result: Box<dyn StorageBackend> = backend.sigmoid(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        let expected: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }
    }

    #[test]
    fn test_cpu_backend_sigmoid_small_values() {
        let backend: CpuBackend = CpuBackend;
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-0.0001, 0.0001, -0.00001]))));
        let result: Box<dyn StorageBackend> = backend.sigmoid(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        let expected: Vec<f32> = vec![0.499975, 0.500025, 0.4999975];
        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }

    }

    #[test]
    fn test_cpu_backend_sigmoid_zeros() {
        let backend: CpuBackend = CpuBackend;
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![0.0, 0.0, 0.0]))));
        let result: Box<dyn StorageBackend> = backend.sigmoid(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        assert_eq!(result_data, &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_cpu_backend_softmax_simple() {
        let backend: CpuBackend = CpuBackend;
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![1.0, 2.0, 3.0]))));
        let result: Box<dyn StorageBackend> = backend.softmax(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        let exp_values: Vec<f32> = vec![1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()];
        let sum_exp_values: f32 = exp_values.iter().sum();
        let expected: Vec<f32> = exp_values.iter().map(|&x| x / sum_exp_values).collect();
        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }
    }

    #[test]
    fn test_cpu_backend_softmax_large_values() {
        let backend: CpuBackend = CpuBackend;
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![1000.0, 1001.0, 1002.0]))));
        let result: Box<dyn StorageBackend> = backend.softmax(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        let exp_values: Vec<f32> = vec![0.0f32.exp(), 1.0f32.exp(), 2.0f32.exp()];
        let sum_exp_values: f32 = exp_values.iter().sum();
        let expected: Vec<f32> = exp_values.iter().map(|&x| x / sum_exp_values).collect();
        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }
    }

    #[test]
    fn test_cpu_backend_softmax_negative_values() {
        let backend: CpuBackend = CpuBackend;
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![-1.0, -2.0, -3.0]))));
        let result: Box<dyn StorageBackend> = backend.softmax(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        let exp_values: Vec<f32> = vec![(-1.0_f32).exp(), (-2.0_f32).exp(), (-3.0_f32).exp()];
        let sum_exp_values: f32 = exp_values.iter().sum();
        let expected: Vec<f32> = exp_values.iter().map(|&x| x / sum_exp_values).collect();

        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }
    }


    #[test]
    fn test_cpu_backend_softmax_zeros() {
        let backend: CpuBackend = CpuBackend;
        let storage: Arc<RwLock<Box<dyn StorageBackend>>> = 
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![0.0, 0.0, 0.0]))));
        let result: Box<dyn StorageBackend> = backend.softmax(&storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();
        let expected: Vec<f32> = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let epsilon: f32 = 1e-6;
        for (res, exp) in result_data.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < epsilon);
        }
    }
}