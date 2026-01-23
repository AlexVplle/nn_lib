use thiserror::Error;

use crate::tensor::Device;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("The tensor is not contiguous")]
    NotContiguous,

    #[error("Empty tensor is not allowed")]
    EmptyTensorNotAllowed,

    #[error("Shape given ({shape_given:?}) is incompatible with tensor shape ({tensor_shape:?})")]
    IncompatibleShape {
        shape_given: Vec<usize>,
        tensor_shape: Vec<usize>,
    },

    #[error("Dimension given ({got}) is not a valid dimension in the tensor (max: {max_dimension})")]
    InvalidDimension { got: usize, max_dimension: usize },

    #[error("Range given is out of bounds")]
    OutOfBounds,

    #[error("Device mismatch: first {first:?}, second {second:?}")]
    DeviceMismatch { first: Device, second: Device },

    #[error("Dimension mismatch")]
    DimensionMismatch,

    #[error("The storage cannot be cloned")]
    StorageNotCloned,

    #[error("Lock error")]
    LockError,

    #[error("Metal backend error: {0}")]
    MetalError(#[from] crate::tensor::backend::metal::error::MetalError),
}
