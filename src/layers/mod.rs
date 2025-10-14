use ndarray::{ArrayD, ShapeError};
use std::any::Any;
use thiserror::Error;

// Re-export all layer types
pub mod activation;
pub mod convolutional;
pub mod dense;
pub mod pooling;
pub mod reshape;

pub use activation::ActivationLayer;
pub use convolutional::ConvolutionalLayer;
pub use dense::DenseLayer;
pub use pooling::MaxPoolingLayer;
pub use reshape::ReshapeLayer;

/// The `Layer` trait need to be implemented by any nn layer
//
/// a layer is defined as input nodes x and output nodes y, and have two main functions,
/// `feed_forward()` and `propagate_backward()`
///
/// Layer implementations in this library support batch processing, (i.e. processing more than one
/// data point at once).
/// The convention chosen in the layer implementations is (n, features) where n is the number of
/// sample in the batch
pub trait Layer {
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError>;

    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError>;

    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, LayerError>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait Trainable {
    fn get_parameters(&self) -> Vec<ArrayD<f64>>;

    fn get_parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>>;

    fn get_gradients(&self) -> Vec<ArrayD<f64>>;
}

#[derive(Error, Debug)]
pub enum LayerError {
    #[error("Access to stored input of the layer before stored happened")]
    IllegalInputAccess,

    #[error("Error reshaping array: {0}")]
    ReshapeError(#[from] ShapeError),

    #[error("Dimension don't match")]
    DimensionMismatch,
}
