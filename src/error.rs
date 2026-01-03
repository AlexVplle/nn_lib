use ndarray::ShapeError;
use thiserror::Error;

use crate::tensor::Device;

#[derive(Debug, Error)]
pub enum NeuralNetworkError {
    #[error("Missing a last activation layer before the output")]
    MissingActivationLayer,

    #[error(
        "Invalid output activation layer,
        see CostFunction::output_dependant for detailed explanation"
    )]
    WrongOutputActivationLayer,

    #[error("Access to stored input of the layer before stored happened")]
    IllegalInputAccess,

    #[error("Error reshaping array: {0}")]
    ReshapeError(#[from] ShapeError),

    #[error("Dimension don't match")]
    DimensionMismatch,

    #[error("The storage cannot be cloned")]
    NotCloned,

    #[error("The tensor is not contiguous")]
    NotContiguous,

    #[error("Shape given ({shape_given:?}) is incompatible with tensor shape ({tensor_shape:?})")]
    IncompatibleShape {
        shape_given: Vec<usize>,
        tensor_shape: Vec<usize>,
    },

    #[error("Dimension given ({got}) is not a good dimension in the tensor ({max_dimension})")]
    InvalidDimension { got: usize, max_dimension: usize },

    #[error("Range given is out of bounds")]
    OutOfBounds,

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Device mismatch: expected {expected:?}, got {got:?}")]
    DeviceMismatch { expected: Device, got: Device },

    #[error("CUDA error: {0}")]
    CudaError(String),

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

    #[error("Feature not yet implemented: {0}")]
    NotImplemented(&'static str),

    #[error("Lock error")]
    LockError,
}
