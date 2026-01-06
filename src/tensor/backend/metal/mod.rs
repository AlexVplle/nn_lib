use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLSize,
};

use crate::{
    error::NeuralNetworkError,
    tensor::{backend::Backend, storage::StorageBackend, Device},
};

pub mod buffer;
pub mod command_semaphore;
pub mod compute_pipeline;
pub mod constant_values;
pub mod device;
pub mod error;
pub mod function;
pub mod function_constant_values;
pub mod library;
pub mod storage;
pub mod value;

use storage::MetalStorage;

pub struct MetalBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalBackend {
    pub fn new() -> Result<Self, NeuralNetworkError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(NeuralNetworkError::AllocationFailed(
            "Metal device not found".to_string(),
        ))?;
        let command_queue =
            device
                .newCommandQueue()
                .ok_or(NeuralNetworkError::AllocationFailed(
                    "Metal command queue creation failed".to_string(),
                ))?;
        let library = Self::compile_library(&device)?;
        Ok(Self {
            device,
            command_queue,
        })
    }

    fn compile_library(
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, NeuralNetworkError> {
        let source = include_str!("kernels/add.metal");
        let source_ns = NSString::from_str(source);
        let options = MTLCompileOptions::new();
        device
            .newLibraryWithSource_options_error(&source_ns, Some(&options))
            .map_err(|err| {
                let error_desc = err.localizedDescription();
                NeuralNetworkError::AllocationFailed(format!(
                    "Metal compilation error: {}",
                    error_desc.to_string()
                ))
            })
    }

    fn create_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, NeuralNetworkError> {
        let name_ns = NSString::from_str(function_name);
        let function = library.newFunctionWithName(&name_ns).ok_or_else(|| {
            NeuralNetworkError::AllocationFailed(format!(
                "Metal function not found: {}",
                function_name
            ))
        })?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| {
                let error_desc = err.localizedDescription();
                NeuralNetworkError::AllocationFailed(format!(
                    "Metal pipeline error: {}",
                    error_desc.to_string()
                ))
            })
    }
}

impl Backend for MetalBackend {
    fn add(
        &self,
        lhs: &std::sync::Arc<std::sync::RwLock<Box<dyn StorageBackend>>>,
        rhs: &std::sync::Arc<std::sync::RwLock<Box<dyn StorageBackend>>>,
    ) -> Result<Box<dyn StorageBackend>, crate::error::NeuralNetworkError> {
        let lhs_storage = lhs.read().expect("Lock poisoned");
        let rhs_storage = rhs.read().expect("Lock poisoned");

        let lhs_metal = lhs_storage
            .as_any()
            .downcast_ref::<MetalStorage>()
            .expect("MetalBackend expects Metal storage");
        let rhs_metal = rhs_storage
            .as_any()
            .downcast_ref::<MetalStorage>()
            .expect("MetalBackend expects Metal storage");

        let len = lhs_metal.len();

        let output = MetalStorage::new(0, len)?;

        let command_buffer =
            self.command_queue
                .commandBuffer()
                .ok_or(NeuralNetworkError::AllocationFailed(
                    "Metal command buffer creation failed".to_string(),
                ))?;
        let encoder =
            command_buffer
                .computeCommandEncoder()
                .ok_or(NeuralNetworkError::AllocationFailed(
                    "Metal encoder creation failed".to_string(),
                ))?;
        unsafe { encoder.setBuffer_offset_atIndex(Some(lhs_metal.buffer()), 0, 0) };
        unsafe { encoder.setBuffer_offset_atIndex(Some(rhs_metal.buffer()), 0, 1) };
        unsafe { encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 2) };
        let thread_group_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: len,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, thread_group_size);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(Box::new(output))
    }

    fn mul(
        &self,
        lhs: &std::sync::Arc<std::sync::RwLock<Box<dyn StorageBackend>>>,
        rhs: &std::sync::Arc<std::sync::RwLock<Box<dyn StorageBackend>>>,
    ) -> Result<Box<dyn StorageBackend>, crate::error::NeuralNetworkError> {
        todo!()
    }

    fn device(&self) -> Device {
        Device::Metal(0)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::sync::{Arc, RwLock};
//
//     #[test]
//     fn test_metal_backend_creation() {
//         let backend = MetalBackend::new();
//         assert!(backend.is_ok(), "Should create MetalBackend successfully");
//     }
//
//     #[test]
//     fn test_metal_backend_device() {
//         let backend = MetalBackend::new().unwrap();
//         assert_eq!(backend.device(), Device::Metal(0));
//     }
//
//     #[test]
//     fn test_metal_backend_add_simple() {
//         let backend = MetalBackend::new().unwrap();
//
//         let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
//         )));
//
//         let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
//         )));
//
//         let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//         let result_data = result.to_cpu().unwrap();
//
//         assert_eq!(result_data, vec![11.0, 22.0, 33.0, 44.0]);
//     }
//
//     #[test]
//     fn test_metal_backend_add_zeros() {
//         let backend = MetalBackend::new().unwrap();
//
//         let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![5.0, 10.0, 15.0]).unwrap(),
//         )));
//
//         let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![0.0, 0.0, 0.0]).unwrap(),
//         )));
//
//         let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//         let result_data = result.to_cpu().unwrap();
//
//         assert_eq!(result_data, vec![5.0, 10.0, 15.0]);
//     }
//
//     #[test]
//     fn test_metal_backend_add_negatives() {
//         let backend = MetalBackend::new().unwrap();
//
//         let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![10.0, 5.0, -3.0]).unwrap(),
//         )));
//
//         let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![-5.0, -5.0, 8.0]).unwrap(),
//         )));
//
//         let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//         let result_data = result.to_cpu().unwrap();
//
//         assert_eq!(result_data, vec![5.0, 0.0, 5.0]);
//     }
//
//     #[test]
//     fn test_metal_backend_add_floats() {
//         let backend = MetalBackend::new().unwrap();
//
//         let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![1.5, 2.7, 3.3]).unwrap(),
//         )));
//
//         let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![0.5, 0.3, 0.7]).unwrap(),
//         )));
//
//         let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//         let result_data = result.to_cpu().unwrap();
//
//         let epsilon = 1e-6;
//         assert!((result_data[0] - 2.0).abs() < epsilon);
//         assert!((result_data[1] - 3.0).abs() < epsilon);
//         assert!((result_data[2] - 4.0).abs() < epsilon);
//     }
//
//     #[test]
//     fn test_metal_backend_add_large() {
//         let backend = MetalBackend::new().unwrap();
//
//         let size = 10_000;
//         let lhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
//         let rhs_data: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
//
//         let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, lhs_data.clone()).unwrap(),
//         )));
//
//         let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, rhs_data.clone()).unwrap(),
//         )));
//
//         let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//         let result_data = result.to_cpu().unwrap();
//
//         let expected: Vec<f32> = (0..size).map(|i| (i * 3) as f32).collect();
//         assert_eq!(result_data, expected);
//     }
//
//     #[test]
//     fn test_metal_backend_add_single_element() {
//         let backend = MetalBackend::new().unwrap();
//
//         let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![42.0]).unwrap(),
//         )));
//
//         let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![8.0]).unwrap(),
//         )));
//
//         let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//         let result_data = result.to_cpu().unwrap();
//
//         assert_eq!(result_data, vec![50.0]);
//     }
//
//     #[test]
//     fn test_metal_backend_add_performance() {
//         let backend = MetalBackend::new().unwrap();
//
//         let size = 1_000_000;
//         let lhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
//         let rhs_data: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
//
//         let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, lhs_data.clone()).unwrap(),
//         )));
//
//         let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, rhs_data.clone()).unwrap(),
//         )));
//
//         let start = std::time::Instant::now();
//         let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//         let duration = start.elapsed();
//
//         println!("Metal add (1M elements): {:?}", duration);
//
//         let result_data = result.to_cpu().unwrap();
//         let expected: Vec<f32> = (0..size).map(|i| (i * 3) as f32).collect();
//
//         assert_eq!(result_data.len(), expected.len());
//         assert_eq!(result_data, expected);
//     }
//
//     #[test]
//     fn test_metal_backend_add_consecutive() {
//         let backend = MetalBackend::new().unwrap();
//
//         let a_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![1.0, 2.0]).unwrap(),
//         )));
//
//         let b_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![10.0, 20.0]).unwrap(),
//         )));
//
//         let c_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![100.0, 200.0]).unwrap(),
//         )));
//
//         // (1+10) = 11, 22
//         let result1 = backend.add(&a_storage, &b_storage).unwrap();
//         let result1_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(result1));
//
//         // (11+100) = 111, (22+200) = 222
//         let result2 = backend.add(&result1_storage, &c_storage).unwrap();
//         let final_data = result2.to_cpu().unwrap();
//
//         assert_eq!(final_data, vec![111.0, 222.0]);
//     }
//
//     #[test]
//     fn test_metal_backend_add_same_input() {
//         let backend = MetalBackend::new().unwrap();
//
//         let storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![5.0, 10.0, 15.0]).unwrap(),
//         )));
//
//         let result = backend.add(&storage, &storage).unwrap();
//         let result_data = result.to_cpu().unwrap();
//
//         assert_eq!(result_data, vec![10.0, 20.0, 30.0]);
//     }
//
//     #[test]
//     fn test_metal_backend_add_multiple_operations() {
//         let backend = MetalBackend::new().unwrap();
//
//         for i in 0..5 {
//             let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(
//                 Box::new(MetalStorage::from_vec(0, vec![i as f32; 100]).unwrap()),
//             ));
//
//             let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(
//                 Box::new(MetalStorage::from_vec(0, vec![1.0; 100]).unwrap()),
//             ));
//
//             let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//             let result_data = result.to_cpu().unwrap();
//
//             assert_eq!(result_data, vec![i as f32 + 1.0; 100]);
//         }
//     }
//
//     #[test]
//     fn test_metal_backend_add_different_sizes() {
//         let backend = MetalBackend::new().unwrap();
//
//         let sizes = vec![1, 10, 100, 1000, 10000];
//
//         for size in sizes {
//             let lhs_data: Vec<f32> = vec![1.0; size];
//             let rhs_data: Vec<f32> = vec![2.0; size];
//
//             let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(
//                 Box::new(MetalStorage::from_vec(0, lhs_data).unwrap()),
//             ));
//
//             let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(
//                 Box::new(MetalStorage::from_vec(0, rhs_data).unwrap()),
//             ));
//
//             let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//             let result_data = result.to_cpu().unwrap();
//
//             assert_eq!(result_data, vec![3.0; size]);
//         }
//     }
//
//     #[test]
//     fn test_metal_backend_add_precision() {
//         let backend = MetalBackend::new().unwrap();
//
//         let lhs_data = vec![
//             std::f32::consts::PI,
//             std::f32::consts::E,
//             std::f32::consts::SQRT_2,
//         ];
//         let rhs_data = vec![
//             std::f32::consts::TAU,
//             std::f32::consts::LN_2,
//             std::f32::consts::LN_10,
//         ];
//
//         let lhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, lhs_data.clone()).unwrap(),
//         )));
//
//         let rhs_storage: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, rhs_data.clone()).unwrap(),
//         )));
//
//         let result = backend.add(&lhs_storage, &rhs_storage).unwrap();
//         let result_data = result.to_cpu().unwrap();
//
//         let epsilon = 1e-6;
//         for i in 0..lhs_data.len() {
//             assert!((result_data[i] - (lhs_data[i] + rhs_data[i])).abs() < epsilon);
//         }
//     }
//
//     #[test]
//     fn test_metal_backend_reuse() {
//         let backend = MetalBackend::new().unwrap();
//
//         let storage1: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![1.0, 2.0]).unwrap(),
//         )));
//
//         let storage2: Arc<RwLock<Box<dyn StorageBackend>>> = Arc::new(RwLock::new(Box::new(
//             MetalStorage::from_vec(0, vec![3.0, 4.0]).unwrap(),
//         )));
//
//         let result1 = backend.add(&storage1, &storage2).unwrap();
//         assert_eq!(result1.to_cpu().unwrap(), vec![4.0, 6.0]);
//         let result2 = backend.add(&storage1, &storage2).unwrap();
//         assert_eq!(result2.to_cpu().unwrap(), vec![4.0, 6.0]);
//     }
//
//     #[test]
//     fn test_metal_backend_library_compilation() {
//         let device = MTLCreateSystemDefaultDevice().unwrap();
//         let library = MetalBackend::compile_library(&device);
//
//         assert!(library.is_ok(), "Should compile library successfully");
//     }
//
//     #[test]
//     fn test_metal_backend_pipeline_creation() {
//         let device = MTLCreateSystemDefaultDevice().unwrap();
//         let library = MetalBackend::compile_library(&device).unwrap();
//         let pipeline = MetalBackend::create_pipeline(&device, &library, "add");
//
//         assert!(pipeline.is_ok(), "Should create pipeline successfully");
//     }
//
//     #[test]
//     fn test_metal_backend_invalid_function_name() {
//         let device = MTLCreateSystemDefaultDevice().unwrap();
//         let library = MetalBackend::compile_library(&device).unwrap();
//         let pipeline = MetalBackend::create_pipeline(&device, &library, "nonexistent_function");
//
//         assert!(pipeline.is_err(), "Should fail for nonexistent function");
//         match pipeline {
//             Err(MetalError::MetalFunctionNotFound(name)) => {
//                 assert_eq!(name, "nonexistent_function");
//             }
//             _ => panic!("Expected MetalFunctionNotFound error"),
//         }
//     }
// }
