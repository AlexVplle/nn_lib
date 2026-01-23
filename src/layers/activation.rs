use std::any::Any;

use crate::activation::Activation;
use crate::tensor::Tensor;

use super::Layer;
use crate::error::NeuralNetworkError;

/// The `ActivationLayer` apply a activation function to it's input node to yield the output nodes.
#[derive(Debug, Clone)]
pub struct ActivationLayer {
    pub activation: Activation,
    pub input: Option<Tensor>,
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
    fn feed_forward_save(&mut self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        self.input = Some(input.clone());
        self.feed_forward(input)
    }

    /// Return a matrices (shape (n, i)) with the activation function applied to a batch
    /// while storing the input for later use in backpropagation process
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward(&self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        Ok(self.activation.apply_tensor(input)?)
    }

    /// Return the input gradient (shape (n, i)) of this `ActivationLayer` by processing the output gradient.
    /// # Arguments
    /// * `output_gradient` shape (n, j)
    fn propagate_backward(
        &mut self,
        output_gradient: &Tensor,
    ) -> Result<Tensor, NeuralNetworkError> {
        // Pour Softmax + CrossEntropy, le gradient est déjà calculé correctement
        // dans la fonction de coût, donc on le passe tel quel
        if self.activation == Activation::Softmax {
            return Ok(output_gradient.clone());
        }

        match self.input.as_ref() {
            Some(input) => {
                let derivative = self.activation.apply_derivative_tensor(input)?;
                Ok(output_gradient.clone() * derivative)
            }
            None => Err(NeuralNetworkError::IllegalInputAccess),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
