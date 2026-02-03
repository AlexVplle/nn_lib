use log::error;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use crate::error::NeuralNetworkError;
use crate::tensor::Tensor;

fn check_nan(array: &ArrayD<f64>, operation: &str) {
    if array.iter().any(|&x| x.is_nan()) {
        error!("NaN detected after {} operation", operation);
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Default, Copy, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Activation {
    #[default]
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
}

impl Activation {
    /// Apply the activation function derivative to each element of a multidimensional array
    /// used for backpropagation via ArrayD (temporary until autograd is implemented)
    /// # Arguments
    /// * `input` - a multidimensional array;
    fn apply_derivative(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        let result = match self {
            Self::ReLU => input.mapv(|e| if e > 0f64 { 1f64 } else { 0f64 }),
            Self::Tanh => input.mapv(|e| 1f64 - e.tanh().powi(2)),
            Self::Sigmoid => input.mapv(|e| {
                let sig = 1.0 / (1.0 + f64::exp(-e));
                sig * (1.0 - sig)
            }),
            Self::Softmax => unimplemented!("We don't use the softmax jacobian matrix in practice"),
        };
        check_nan(&result, &format!("{:?}_derivative", self));
        result
    }

    pub fn apply_tensor(&self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        match self {
            Self::ReLU => Ok(input.relu()?),
            Self::Tanh => Ok(input.tanh()?),
            Self::Sigmoid => Ok(input.sigmoid()?),
            Self::Softmax => Ok(input.softmax()?),
        }
    }

    pub fn apply_derivative_tensor(&self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        let data = input.to_vec()?;
        let shape = input.shape().to_vec();
        let device = input.device().clone();

        let array = ArrayD::from_shape_vec(shape.clone(), data.iter().map(|&x| x as f64).collect())
            .map_err(|_e| {
                NeuralNetworkError::TensorError(crate::tensor::TensorError::IncompatibleShape {
                    shape_given: shape.clone(),
                    tensor_shape: vec![data.len()],
                })
            })?;

        let result_array = self.apply_derivative(&array);
        let result_data: Vec<f32> = result_array.iter().map(|&x| x as f32).collect();

        Ok(Tensor::new(result_data, shape, device)?)
    }
}
