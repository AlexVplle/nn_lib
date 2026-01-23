use std::any::Any;

use crate::error::NeuralNetworkError;
use crate::tensor::Tensor;
use super::Layer;

#[derive(Debug, Clone)]
pub struct ReshapeLayer {
    input: Option<Tensor>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl ReshapeLayer {
    pub fn new(input_shape: &[usize], output_shape: &[usize]) -> Result<Self, NeuralNetworkError> {
        let input_elements: usize = input_shape.iter().product();
        let output_elements: usize = output_shape.iter().product();
        if input_elements != output_elements {
            return Err(NeuralNetworkError::TensorError(
                crate::tensor::TensorError::IncompatibleShape {
                    shape_given: output_shape.to_vec(),
                    tensor_shape: input_shape.to_vec(),
                },
            ));
        }
        Ok(Self {
            input: None,
            input_shape: input_shape.to_vec(),
            output_shape: output_shape.to_vec(),
        })
    }
}

impl Layer for ReshapeLayer {
    fn feed_forward_save(&mut self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        self.input = Some(input.clone());
        self.feed_forward(input)
    }

    fn feed_forward(&self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        let batch_size: usize = input.shape()[0];
        let mut shape: Vec<usize> = Vec::with_capacity(self.output_shape.len() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(&self.output_shape);

        let data = input.to_vec()?;
        let device = input.device().clone();
        Ok(Tensor::new(data, shape, device)?)
    }

    fn propagate_backward(
        &mut self,
        output_gradient: &Tensor,
    ) -> Result<Tensor, NeuralNetworkError> {
        let batch_size: usize = output_gradient.shape()[0];
        let mut shape: Vec<usize> = Vec::with_capacity(self.input_shape.len() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(&self.input_shape);

        let data = output_gradient.to_vec()?;
        let device = output_gradient.device().clone();
        Ok(Tensor::new(data, shape, device)?)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
