use ndarray::ArrayD;
use std::any::Any;

use crate::initialization::InitializerType;

use crate::error::NeuralNetworkError;
use super::{Layer, Trainable};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DenseLayer {
    weights: ArrayD<f64>,
    bias: ArrayD<f64>,
    last_batch_input: Option<ArrayD<f64>>,
    // store those for optimizer access (from the trait Trainable)
    weights_gradient: Option<ArrayD<f64>>,
    biases_gradient: Option<ArrayD<f64>>,
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
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NeuralNetworkError> {
        // TODO find a without clone method, like in place mutation
        self.last_batch_input = Some(input.clone());
        self.feed_forward(input)
    }

    /// Return the output matrices of this `DenseLayer` (shape (n, j))
    ///
    /// where **n** is the number of samples, **j** is the layer output size.
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NeuralNetworkError> {
        let batch_size = input.shape()[0];
        let input_2d = input.view().into_shape((batch_size, self.input_size))?;
        let weight_2d = self
            .weights
            .view()
            .into_shape((self.input_size, self.output_size))?;

        Ok((input_2d.dot(&weight_2d) + &self.bias).into_dyn())
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
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, NeuralNetworkError> {
        let input_gradient = match self.last_batch_input.as_ref() {
            Some(input) => {
                let batch_size = output_gradient.shape()[0];
                let output_grad_2d = output_gradient
                    .view()
                    .into_shape((batch_size, self.output_size))?;

                let input_2d = input.view().into_shape((batch_size, self.input_size))?;

                let weight_2d = self
                    .weights
                    .view()
                    .into_shape((self.input_size, self.output_size))?;

                // mean relative to the batch
                let weights_gradient = input_2d.t().dot(&output_grad_2d) / batch_size as f64;
                let biases_gradient = output_grad_2d.sum_axis(ndarray::Axis(0)) / batch_size as f64;

                self.weights_gradient = Some(weights_gradient.to_owned().into_dyn());
                self.biases_gradient = Some(biases_gradient.into_dyn());

                Ok((output_grad_2d.dot(&weight_2d.t())).into_dyn())
            }
            None => Err(NeuralNetworkError::IllegalInputAccess),
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
    fn get_parameters(&self) -> Vec<ArrayD<f64>> {
        vec![
            self.weights.clone().into_dyn(),
            self.bias.clone().into_dyn(),
        ]
    }

    fn get_parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn get_gradients(&self) -> Vec<ArrayD<f64>> {
        vec![
            self.weights_gradient
                .as_ref()
                .expect("Illegal access to unset weights gradient")
                .clone()
                .into_dyn(),
            self.biases_gradient
                .as_ref()
                .expect("Illegal access to unset biases gradient")
                .clone()
                .into_dyn(),
        ]
    }
}
