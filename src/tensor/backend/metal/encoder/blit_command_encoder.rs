use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandEncoder};
use std::sync::Arc;

use crate::tensor::backend::metal::{
    buffer::Buffer,
    command_semaphore::{CommandSemaphore, CommandStatus},
};

pub struct BlitCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
}

impl BlitCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> Self {
        BlitCommandEncoder { raw, semaphore }
    }

    pub fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn end_encoding(&self) {
        self.raw.endEncoding();
        self.signal_encoding_ended();
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }

    pub fn copy_from_buffer(
        &self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer.as_ref(),
                    src_offset,
                    dst_buffer.as_ref(),
                    dst_offset,
                    size,
                );
        }
    }

    pub fn fill_buffer(&self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.raw.fillBuffer_range_value(
            buffer.as_ref(),
            NSRange {
                location: range.0,
                length: range.1,
            },
            value,
        );
    }
}

impl AsRef<BlitCommandEncoder> for BlitCommandEncoder {
    fn as_ref(&self) -> &BlitCommandEncoder {
        self
    }
}
