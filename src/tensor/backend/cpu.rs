use std::sync::{Arc, RwLock};

use crate::error::NeuralNetworkError;
use crate::tensor::{
    backend::Backend,
    storage::{cpu_storage::CpuStorage, storage::StorageBackend},
    Device,
};

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn add(
        &self,
        lhs: &Arc<RwLock<Box<dyn StorageBackend>>>,
        rhs: &Arc<RwLock<Box<dyn StorageBackend>>>,
    ) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        let lhs_storage = lhs.read().expect("Lock poisoned");
        let rhs_storage = rhs.read().expect("Lock poisoned");

        let lhs_cpu = lhs_storage
            .as_any()
            .downcast_ref::<CpuStorage>()
            .expect("CpuBackend expects CPU storage");
        let rhs_cpu = rhs_storage
            .as_any()
            .downcast_ref::<CpuStorage>()
            .expect("CpuBackend expects CPU storage");

        let lhs_data = lhs_cpu.as_slice();
        let rhs_data = rhs_cpu.as_slice();

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Box::new(CpuStorage::from_vec(result)))
    }

    fn mul(
        &self,
        lhs: &Arc<RwLock<Box<dyn StorageBackend>>>,
        rhs: &Arc<RwLock<Box<dyn StorageBackend>>>,
    ) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        let lhs_storage = lhs.read().expect("Lock poisoned");
        let rhs_storage = rhs.read().expect("Lock poisoned");

        let lhs_cpu = lhs_storage
            .as_any()
            .downcast_ref::<CpuStorage>()
            .expect("CpuBackend expects CPU storage");
        let rhs_cpu = rhs_storage
            .as_any()
            .downcast_ref::<CpuStorage>()
            .expect("CpuBackend expects CPU storage");

        let lhs_data = lhs_cpu.as_slice();
        let rhs_data = rhs_cpu.as_slice();

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(a, b)| a * b)
            .collect();

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
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                1.0, 2.0, 3.0, 4.0,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                10.0, 20.0, 30.0, 40.0,
            ]))));

        let result: Box<dyn StorageBackend> = backend.add(&lhs_storage, &rhs_storage).unwrap();

        let result_cpu: &CpuStorage = result
            .as_any()
            .downcast_ref::<CpuStorage>()
            .expect("Result should be CpuStorage");

        let result_data: &[f32] = result_cpu.as_slice();

        assert_eq!(result_data, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_cpu_backend_add_zeros() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                5.0, 10.0, 15.0,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                0.0, 0.0, 0.0,
            ]))));

        let result: Box<dyn StorageBackend> = backend.add(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        assert_eq!(result_data, &[5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_cpu_backend_add_negatives() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                10.0, 5.0, -3.0,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                -5.0, -5.0, 8.0,
            ]))));

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

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
            CpuStorage::from_vec(lhs_data.clone()),
        )));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
            CpuStorage::from_vec(rhs_data.clone()),
        )));

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
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                1.5, 2.7, 3.3,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                0.5, 0.3, 0.7,
            ]))));

        let result: Box<dyn StorageBackend> = backend.add(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        let epsilon: f32 = 1e-6;
        assert!((result_data[0] - 2.0).abs() < epsilon);
        assert!((result_data[1] - 3.0).abs() < epsilon);
        assert!((result_data[2] - 4.0).abs() < epsilon);
    }

    #[test]
    fn test_cpu_backend_mul_simple() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                2.0, 3.0, 4.0, 5.0,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                10.0, 20.0, 30.0, 40.0,
            ]))));

        let result: Box<dyn StorageBackend> = backend.mul(&lhs_storage, &rhs_storage).unwrap();

        let result_cpu: &CpuStorage = result
            .as_any()
            .downcast_ref::<CpuStorage>()
            .expect("Result should be CpuStorage");

        let result_data: &[f32] = result_cpu.as_slice();

        assert_eq!(result_data, &[20.0, 60.0, 120.0, 200.0]);
    }

    #[test]
    fn test_cpu_backend_mul_zeros() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                5.0, 10.0, 15.0,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                0.0, 0.0, 0.0,
            ]))));

        let result: Box<dyn StorageBackend> = backend.mul(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        assert_eq!(result_data, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cpu_backend_mul_ones() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                7.0, 13.0, 21.0,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                1.0, 1.0, 1.0,
            ]))));

        let result: Box<dyn StorageBackend> = backend.mul(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        assert_eq!(result_data, &[7.0, 13.0, 21.0]);
    }

    #[test]
    fn test_cpu_backend_mul_negatives() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                -2.0, 3.0, -4.0,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                5.0, -2.0, -3.0,
            ]))));

        let result: Box<dyn StorageBackend> = backend.mul(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        assert_eq!(result_data, &[-10.0, -6.0, 12.0]);
    }

    #[test]
    fn test_cpu_backend_mul_large() {
        let backend: CpuBackend = CpuBackend;

        let size: usize = 1000;
        let lhs_data: Vec<f32> = (0..size).map(|i: usize| i as f32).collect();
        let rhs_data: Vec<f32> = (0..size).map(|i: usize| 2.0).collect();

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
            CpuStorage::from_vec(lhs_data.clone()),
        )));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
            CpuStorage::from_vec(rhs_data.clone()),
        )));

        let result: Box<dyn StorageBackend> = backend.mul(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        let expected: Vec<f32> = (0..size).map(|i: usize| (i * 2) as f32).collect();
        assert_eq!(result_data, &expected[..]);
    }

    #[test]
    fn test_cpu_backend_mul_single_element() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![7.0]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![6.0]))));

        let result: Box<dyn StorageBackend> = backend.mul(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        assert_eq!(result_data, &[42.0]);
    }

    #[test]
    fn test_cpu_backend_mul_floats() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                1.5, 2.5, 3.5,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                2.0, 4.0, 0.5,
            ]))));

        let result: Box<dyn StorageBackend> = backend.mul(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        let epsilon: f32 = 1e-6;
        assert!((result_data[0] - 3.0).abs() < epsilon);
        assert!((result_data[1] - 10.0).abs() < epsilon);
        assert!((result_data[2] - 1.75).abs() < epsilon);
    }

    #[test]
    fn test_cpu_backend_mul_mixed_signs() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                -1.0, -1.0, 1.0, 1.0,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                -1.0, 1.0, -1.0, 1.0,
            ]))));

        let result: Box<dyn StorageBackend> = backend.mul(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        assert_eq!(result_data, &[1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_cpu_backend_mul_fractional() {
        let backend: CpuBackend = CpuBackend;

        let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                10.0, 100.0, 1000.0,
            ]))));
        let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> =
            Arc::new(RwLock::new(Box::new(CpuStorage::from_vec(vec![
                0.1, 0.01, 0.001,
            ]))));

        let result: Box<dyn StorageBackend> = backend.mul(&lhs_storage, &rhs_storage).unwrap();
        let result_cpu: &CpuStorage = result.as_any().downcast_ref::<CpuStorage>().unwrap();
        let result_data: &[f32] = result_cpu.as_slice();

        let epsilon: f32 = 1e-6;
        assert!((result_data[0] - 1.0).abs() < epsilon);
        assert!((result_data[1] - 1.0).abs() < epsilon);
        assert!((result_data[2] - 1.0).abs() < epsilon);
    }
}

