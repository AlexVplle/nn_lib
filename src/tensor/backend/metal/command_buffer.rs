use std::{borrow::Cow, ffi::CStr, sync::Arc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};

use crate::tensor::backend::metal::{
    command_semaphore::CommandSemaphore,
    encoder::{BlitCommandEncoder, ComputeCommandEncoder},
};

#[derive(Debug)]
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

    pub fn blit_command_encoder(&self) -> BlitCommandEncoder {
        self.as_ref()
            .blitCommandEncoder()
            .map(|raw| BlitCommandEncoder::new(raw, Arc::clone(&self.semaphore)))
            .unwrap()
    }

    pub fn commit(&self) {
        self.raw.commit()
    }

    pub fn enqueue(&self) {
        self.raw.enqueue()
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().setLabel(Some(&NSString::from_str(label)));
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        self.raw.status()
    }

    pub fn error(&self) -> Option<Cow<'_, str>> {
        unsafe {
            self.raw.error().map(|error| {
                let description = error.localizedDescription();
                let c_str = CStr::from_ptr(description.UTF8String());
                c_str.to_string_lossy()
            })
        }
    }

    pub fn wait_until_completed(&self) {
        self.raw.waitUntilCompleted();
    }
}

impl AsRef<ProtocolObject<dyn MTLCommandBuffer>> for CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }
}
