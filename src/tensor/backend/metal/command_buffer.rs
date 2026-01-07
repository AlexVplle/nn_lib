use std::sync::Arc;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLCommandBuffer;

use crate::tensor::backend::metal::{
    command_semaphore::CommandSemaphore, compute_command_encoder::ComputeCommandEncoder,
};

pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    semaphore: Arc<CommandSemaphore>,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> Self {
        Self { raw, semaphore }
    }

    pub fn compute_command_encoder(&self) -> ComputeCommandEncoder {
        self.as_ref()
            .computeCommandEncoder()
            .map(|raw| ComputeCommandEncoder::new(raw, Arc::clone(&self.semaphore)))
            .unwrap()
    }
}

impl AsRef<ProtocolObject<dyn MTLCommandBuffer>> for CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }
}
