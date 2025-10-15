use ndarray::ArrayD;
use std::any::Any;

use crate::activation::Activation;

use super::{Layer, LayerError};

/// The `ActivationLayer` apply a activation function to it's input node to yield the output nodes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ActivationLayer {
    pub activation: Activation,
    pub input: Option<ArrayD<f64>>,
}

impl ActivationLayer {
    pub fn from(activation: Activation) -> Self {
        Self {
            activation,
            input: None,
        }
    }
}

impl Layer for ActivationLayer {
    /// Return a matrices (shape (n, i)) with the activation function applied to a batch
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        self.input = Some(input.clone());
        self.feed_forward(input)
    }

    /// Return a matrices (shape (n, i)) with the activation function applied to a batch
    /// while storing the input for later use in backpropagation process
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        Ok(self.activation.apply(input))
    }

    /// Return the input gradient (shape (n, i)) of this `ActivationLayer` by processing the output gradient.
    /// # Arguments
    /// * `output_gradient` shape (n, j)
    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, LayerError> {
        let input_gradient = match self.input.as_ref() {
            Some(input) => Ok(output_gradient * self.activation.apply_derivative(input)),
            None => Err(LayerError::IllegalInputAccess),
        };
        input_gradient
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
