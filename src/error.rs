use ndarray::ShapeError;
use thiserror::Error;

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

    #[error("Dimension mismatch in training data")]
    DimensionMismatch,

    #[error("Tensor error: {0}")]
    TensorError(#[from] crate::tensor::TensorError),

    #[error("{0}")]
    Other(String),
}
