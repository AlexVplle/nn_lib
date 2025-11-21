use candle_core::Tensor;
use std::any::Any;

use crate::initialization::InitializerType;

use super::{Layer, LayerError, Trainable};

#[derive(Debug, Clone)]
pub struct DenseLayer {
    weights: Tensor,
    bias: Tensor,
    last_batch_input: Option<Tensor>,
    // store those for optimizer access (from the trait Trainable)
    weights_gradient: Option<Tensor>,
    biases_gradient: Option<Tensor>,
    input_size: usize,
    output_size: usize,
}

impl DenseLayer {
    /// Create a new `DenseLayer` filling it with random value. see `InitializerType` for
    /// initialization parameters
    pub fn new(input_size: usize, output_size: usize, init: InitializerType) -> Self {
        Self {
            weights: init.initialize(input_size, output_size, &[input_size, output_size]),
            bias: init.initialize(input_size, output_size, &[output_size]),
            last_batch_input: None,
            weights_gradient: None,
            biases_gradient: None,
            input_size,
            output_size,
        }
    }
}

impl Layer for DenseLayer {
    /// Return the output matrices of this `DenseLayer` (shape (n, j)), while storing the input matrices
    /// (shape (n, i))
    ///
    /// where **n** is the number of samples, **j** is the layer output size and **i** is the layer
    /// input size.
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward_save(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.last_batch_input = Some(input.clone());
        self.feed_forward(input)
    }

    /// Return the output matrices of this `DenseLayer` (shape (n, j))
    ///
    /// where **n** is the number of samples, **j** is the layer output size.
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let batch_size = input.dims()[0];
        let input_2d = input.reshape(&[batch_size, self.input_size])?;
        let weight_2d = self.weights.reshape(&[self.input_size, self.output_size])?;

        let output = input_2d.matmul(&weight_2d)?;
        let output_with_bias = output.broadcast_add(&self.bias)?;

        Ok(output_with_bias)
    }

    /// Return the input gradient vector (shape (n, i)), by processing the output gradient vector
    /// (shape (n, j)).
    ///
    /// This function also compute and store the current batch weights and biases gradient in the layer.
    ///
    /// # Arguments
    /// * `input` - (shape (n, i))
    /// * `output_gradient` - (shape (n, j))
    fn propagate_backward(
        &mut self,
        output_gradient: &Tensor,
    ) -> Result<Tensor, LayerError> {
        let input_gradient = match self.last_batch_input.as_ref() {
            Some(input) => {
                let batch_size = output_gradient.dims()[0];
                let output_grad_2d = output_gradient.reshape(&[batch_size, self.output_size])?;

                let input_2d = input.reshape(&[batch_size, self.input_size])?;

                let weight_2d = self.weights.reshape(&[self.input_size, self.output_size])?;

                // mean relative to the batch
                let weights_gradient = (input_2d.t()?.matmul(&output_grad_2d)? / (batch_size as f64))?;
                let biases_gradient = (output_grad_2d.sum(0)? / (batch_size as f64))?;

                self.weights_gradient = Some(weights_gradient);
                self.biases_gradient = Some(biases_gradient);

                let weight_2d_t = weight_2d.t()?;
                Ok(output_grad_2d.matmul(&weight_2d_t)?)
            }
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

impl Trainable for DenseLayer {
    fn get_parameters(&self) -> Vec<Tensor> {
        vec![
            self.weights.clone(),
            self.bias.clone(),
        ]
    }

    fn get_parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn get_gradients(&self) -> Vec<Tensor> {
        vec![
            self.weights_gradient
                .as_ref()
                .expect("Illegal access to unset weights gradient")
                .clone(),
            self.biases_gradient
                .as_ref()
                .expect("Illegal access to unset biases gradient")
                .clone(),
        ]
    }
}
