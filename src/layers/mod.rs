use std::any::Any;

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

use crate::error::NeuralNetworkError;
use crate::tensor::Tensor;

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
    fn feed_forward_save(&mut self, input: &Tensor) -> Result<Tensor, NeuralNetworkError>;

    fn feed_forward(&self, input: &Tensor) -> Result<Tensor, NeuralNetworkError>;

    fn propagate_backward(
        &mut self,
        output_gradient: &Tensor,
    ) -> Result<Tensor, NeuralNetworkError>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait Trainable {
    fn get_parameters(&self) -> Vec<Tensor>;

    fn get_parameters_mut(&mut self) -> Vec<&mut Tensor>;

    fn get_gradients(&self) -> Vec<Tensor>;
}
