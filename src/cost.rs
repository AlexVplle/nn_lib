use crate::error::NeuralNetworkError;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize)]
pub enum CostFunction {
    #[default]
    CrossEntropy,
    BinaryCrossEntropy,
    Mse,
}

impl CostFunction {
    /// This crate don't use any kind of auto diff mechanism,
    /// thus, for function like BinaryCrossEntropy and CrossEntropy that need clamped output,
    /// we assume Sigmoid and Softmax respectively as the output activation layer.
    /// the gradient calculation is done with those activation function in mind.
    /// Those function are called 'Output dependant' to contrast with function like Mse, of which
    /// the derivative can be easily calculated with respect to any output layer, because it
    /// doesn't need clamped output.
    pub fn is_output_dependant(&self) -> bool {
        match self {
            Self::BinaryCrossEntropy | Self::CrossEntropy => true,
            Self::Mse => false,
        }
    }

    /// Compute the mean cost of the neural network with respect to a batch output and observed values
    ///
    /// # Arguments
    /// * `output` - Network output tensor, shape: [batch_size, n_classes]
    /// * `observed` - One-hot encoded labels tensor, shape: [batch_size, n_classes]
    ///
    /// # Returns
    /// * `Result<f64, NeuralNetworkError>` - Mean cost over the batch
    pub fn cost(&self, output: &Tensor, observed: &Tensor) -> Result<f64, NeuralNetworkError> {
        let output_arr: ArrayD<f64> = output.clone().into();
        let observed_arr: ArrayD<f64> = observed.clone().into();

        let epsilon = 1e-7;
        let clipped_output = output_arr.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));

        let result = match self {
            Self::CrossEntropy => {
                observed_arr
                    .axis_iter(Axis(0))
                    .enumerate()
                    .map(|(i, observed_row)| {
                        let correct_class = observed_row.iter().position(|&x| x == 1.0).unwrap();
                        -f64::ln(clipped_output[[i, correct_class]])
                    })
                    .sum::<f64>()
                    / output_arr.shape()[0] as f64
            }
            Self::BinaryCrossEntropy => {
                let losses = &observed_arr * &clipped_output.mapv(f64::ln)
                    + &(1.0 - &observed_arr) * &((1.0 - &clipped_output).mapv(f64::ln));
                -losses.mean().unwrap()
            }
            Self::Mse => {
                let diff = &output_arr - &observed_arr;
                diff.mapv(|x| x.powi(2)).mean().unwrap()
            }
        };

        Ok(result)
    }

    /// Return the gradient of cost function with respect to output
    ///
    /// Note that this simple, from 'almost' scratch library don't use auto-differentiation
    /// thus `BinaryCrossEntropy` calculation assume a Sigmoid activation as the last layer.
    /// `CrossEntropy` calculation assume a Softmax activation as the last layer.
    ///
    /// # Arguments
    /// * `output` - Network output tensor, shape: [batch_size, n_classes]
    /// * `observed` - One-hot encoded labels tensor, shape: [batch_size, n_classes]
    ///
    /// # Returns
    /// * `Result<Tensor, NeuralNetworkError>` - Gradient tensor, same shape as input
    ///
    /// Note that CrossEntropy and BinaryCrossEntropy assume one hot encoded vector for the
    /// observed values in multi-class classification.
    pub fn cost_output_gradient(
        &self,
        output: &Tensor,
        observed: &Tensor,
    ) -> Result<Tensor, NeuralNetworkError> {
        let output_arr: ArrayD<f64> = output.clone().into();
        let observed_arr: ArrayD<f64> = observed.clone().into();

        let gradient_arr = match self {
            Self::CrossEntropy => &output_arr - &observed_arr,
            Self::BinaryCrossEntropy => &output_arr - &observed_arr,
            Self::Mse => {
                let batch_size: usize = output_arr.shape()[0];
                2.0 * (&output_arr - &observed_arr) / batch_size as f64
            }
        };

        let data: Vec<f32> = gradient_arr.iter().map(|&x| x as f32).collect();
        let shape = gradient_arr.shape().to_vec();
        Ok(Tensor::new(data, shape, output.device().clone())?)
    }
}
