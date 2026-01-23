use std::{ffi::c_void, ptr, sync::Arc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandEncoder, MTLComputeCommandEncoder, MTLResourceUsage, MTLSize};

use crate::tensor::backend::metal::{
    buffer::{Buffer, MetalResource},
    command_semaphore::{CommandSemaphore, CommandStatus},
    compute_pipeline::ComputePipeline,
};

pub struct ComputeCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
}

impl AsRef<ComputeCommandEncoder> for ComputeCommandEncoder {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self
    }
}

impl ComputeCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> Self {
        Self { raw, semaphore }
    }

    pub fn signal_encoded_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe {
            self.raw.setThreadgroupMemoryLength_atIndex(length, index);
        }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup);
    }

    pub fn dispatch_thread_groups(
        &self,
        threads_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.raw
            .dispatchThreadgroups_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup);
    }

    pub fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b: &Buffer| b.as_ref()), offset, index);
        }
    }

    pub fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const c_void) {
        let pointer = ptr::NonNull::new(bytes as *mut c_void).unwrap();
        unsafe {
            self.raw.setBytes_length_atIndex(pointer, length, index);
        }
    }

    pub fn set_bytes<T>(&self, index: usize, data: &T) {
        let size = core::mem::size_of::<T>();
        let pointer = ptr::NonNull::new(data as *const T as *mut c_void).unwrap();
        unsafe {
            self.raw.setBytes_length_atIndex(pointer, size, index);
        }
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    pub fn use_resource<'a>(
        &self,
        resource: impl Into<&'a MetalResource>,
        resource_usage: MTLResourceUsage,
    ) {
        self.raw.useResource_usage(resource.into(), resource_usage);
    }

    pub fn end_encoding(&self) {
        self.raw.endEncoding();
        self.signal_encoded_ended();
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)));
    }
}

impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        self.end_encoding();
    }
}
