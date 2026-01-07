use ndarray::ShapeError;
use thiserror::Error;

// use crate::tensor::Device;

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

    #[error("Empty tensor is not allowed")]
    EmptyTensorNotAllowed,

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

    // #[error("Device mismatch: expected {expected:?}, got {got:?}")]
    // DeviceMismatch { expected: Device, got: Device },
    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Feature not yet implemented: {0}")]
    NotImplemented(&'static str),

    #[error("Lock error")]
    LockError,
}
