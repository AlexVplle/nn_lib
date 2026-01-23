use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Metal device not found")]
    DeviceNotFound,

    #[error("Metal device creation failed: no system default device available")]
    DeviceCreationFailed,

    #[error("Metal command queue creation failed")]
    CommandQueueCreationFailed,

    #[error("Metal command buffer creation failed")]
    CommandBufferCreationFailed,

    #[error("Metal command buffer pool is empty")]
    CommandBufferPoolEmpty,

    #[error("Metal command buffer execution error: {0}")]
    CommandBufferExecutionError(String),

    #[error("Metal library compilation error: {0}")]
    LibraryCompilationError(String),

    #[error("Metal function not found: {0}")]
    FunctionNotFound(String),

    #[error("Metal compute pipeline creation failed: {0}")]
    PipelineCreationError(String),

    #[error("Metal buffer allocation failed: {0}")]
    BufferAllocationFailed(String),

    #[error("Metal buffer creation failed: requested size {0} bytes")]
    BufferCreationFailed(usize),

    #[error("Metal buffer with data creation failed")]
    BufferWithDataCreationFailed,

    #[error("Metal compute command encoder creation failed")]
    ComputeEncoderCreationFailed,

    #[error("Metal blit command encoder creation failed")]
    BlitEncoderCreationFailed,

    #[error("Metal lock acquisition failed: {0}")]
    LockError(String),

    #[error("Metal buffer lock failed for: {0}")]
    BufferLockFailed(String),

    #[error("Metal command lock failed: {0}")]
    CommandLockFailed(String),

    #[error("Null pointer encountered in Metal operation")]
    NullPointer,

    #[error("Metal kernel not found: {0}")]
    KernelNotFound(String),

    #[error("Metal invalid buffer size: requested {requested}, maximum {maximum}")]
    InvalidBufferSize { requested: usize, maximum: usize },
}

impl<T> From<std::sync::PoisonError<T>> for MetalError {
    fn from(error: std::sync::PoisonError<T>) -> Self {
        MetalError::LockError(error.to_string())
    }
}
