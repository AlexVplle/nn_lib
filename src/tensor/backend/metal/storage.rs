use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

use crate::error::NeuralNetworkError;

use std::ptr::NonNull;

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
        let device = MTLCreateSystemDefaultDevice().ok_or(NeuralNetworkError::AllocationFailed(
            "Metal device not found".to_string(),
        ))?;
        let byte_size = (size * std::mem::size_of::<f32>()) as NSUInteger;
        let buffer = device
            .newBufferWithLength_options(byte_size, MTLResourceOptions::StorageModeShared)
            .ok_or(NeuralNetworkError::AllocationFailed(
                "Metal buffer creation failed".to_string(),
            ))?;
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
        let device = MTLCreateSystemDefaultDevice().ok_or(NeuralNetworkError::AllocationFailed(
            "Metal device not found".to_string(),
        ))?;
        let byte_size = (len * std::mem::size_of::<f32>()) as NSUInteger;
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).ok_or(
            NeuralNetworkError::AllocationFailed("Null pointer".to_string()),
        )?;
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                ptr,
                byte_size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(NeuralNetworkError::AllocationFailed(
            "Metal buffer creation failed".to_string(),
        ))?;
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
