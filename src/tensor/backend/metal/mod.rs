use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLSize,
};

use crate::{
    error::NeuralNetworkError,
    tensor::{
        backend::Backend,
        storage::{metal_storage::MetalStorage, storage::StorageBackend},
        Device,
    },
};

pub struct MetalBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    add_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalBackend {
    pub fn new() -> Result<Self, NeuralNetworkError> {
        let device =
            MTLCreateSystemDefaultDevice().ok_or(NeuralNetworkError::MetalDeviceNotFound)?;
        let command_queue = device
            .newCommandQueue()
            .ok_or(NeuralNetworkError::MetalCommandQueueCreationFailed)?;
        let library = Self::compile_library(&device)?;
        let add_pipeline = Self::create_pipeline(&device, &library, "add")?;
        Ok(Self {
            device,
            command_queue,
            add_pipeline,
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
                NeuralNetworkError::MetalCompilationError(error_desc.to_string())
            })
    }

    fn create_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, NeuralNetworkError> {
        let name_ns = NSString::from_str(function_name);
        let function = library
            .newFunctionWithName(&name_ns)
            .ok_or_else(|| NeuralNetworkError::MetalFunctionNotFound(function_name.to_string()))?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| {
                let error_desc = err.localizedDescription();
                NeuralNetworkError::MetalPipelineError(error_desc.to_string())
            })
    }
}

impl Backend for MetalBackend {
    fn add(
        &self,
        lhs: &std::sync::Arc<
            std::sync::RwLock<Box<dyn crate::tensor::storage::storage::StorageBackend>>,
        >,
        rhs: &std::sync::Arc<
            std::sync::RwLock<Box<dyn crate::tensor::storage::storage::StorageBackend>>,
        >,
    ) -> Result<Box<dyn crate::tensor::storage::storage::StorageBackend>, NeuralNetworkError> {
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

        let command_buffer = self
            .command_queue
            .commandBuffer()
            .ok_or(NeuralNetworkError::MetalCommandBufferCreationFailed)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(NeuralNetworkError::MetalEncoderCreationFailed)?;
        encoder.setComputePipelineState(&self.add_pipeline);
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
    ) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        todo!()
    }

    fn device(&self) -> Device {
        Device::Metal(0)
    }
}
