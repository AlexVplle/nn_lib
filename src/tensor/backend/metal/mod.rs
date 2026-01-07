use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};

use crate::{error::NeuralNetworkError, tensor::backend::metal::device::Device};

pub mod buffer;
pub mod command_buffer;
pub mod command_semaphore;
pub mod commands;
pub mod compute_command_encoder;
pub mod compute_pipeline;
pub mod constant_values;
pub mod device;
pub mod error;
pub mod function;
pub mod function_constant_values;
pub mod library;
pub mod storage;
pub mod value;

pub struct MetalDevice {
    device: Device,
}

impl MetalDevice {
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
        Ok(Self { device })
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
