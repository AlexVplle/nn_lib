use std::sync::Arc;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::tensor::backend::metal::command_semaphore::{CommandSemaphore, CommandStatus};

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

    pub fn dispatch_thread(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
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
}
