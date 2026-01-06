use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Metal device not found")]
    MetalDeviceNotFound,

    #[error("Metal command queue creation failed")]
    MetalCommandQueueCreationFailed,

    #[error("Metal compilation error: {0}")]
    MetalCompilationError(String),

    #[error("Metal function not found: {0}")]
    MetalFunctionNotFound(String),

    #[error("Metal pipeline error: {0}")]
    MetalPipelineError(String),

    #[error("Metal buffer creation failed")]
    MetalBufferCreationFailed,

    #[error("Metal command buffer creation failed")]
    MetalCommandBufferCreationFailed,

    #[error("Metal encoder creation failed")]
    MetalEncoderCreationFailed,

    #[error("Null pointer")]
    NullPointer,
}
