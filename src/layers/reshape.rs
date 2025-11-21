use candle_core::Tensor;
use std::any::Any;

use super::{Layer, LayerError};

#[derive(Debug, Clone)]
pub struct ReshapeLayer {
    input: Option<Tensor>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl ReshapeLayer {
    pub fn new(input_shape: &[usize], output_shape: &[usize]) -> Result<Self, LayerError> {
        let input_elements: usize = input_shape.iter().product();
        let output_elements: usize = output_shape.iter().product();
        if input_elements != output_elements {
            return Err(LayerError::DimensionMismatch);
        }
        Ok(Self {
            input: None,
            input_shape: input_shape.to_vec(),
            output_shape: output_shape.to_vec(),
        })
    }
}

impl Layer for ReshapeLayer {
    fn feed_forward_save(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.input = Some(input.clone());
        self.feed_forward(input)
    }

    fn feed_forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let batch_size = input.dims()[0];
        let mut shape: Vec<usize> = Vec::with_capacity(self.output_shape.len() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(&self.output_shape);

        let input_elements: usize = input.dims().iter().product();
        let target_elements: usize = shape.iter().product();
        if input_elements != target_elements {
            return Err(LayerError::DimensionMismatch);
        }
        Ok(input.reshape(shape.as_slice())?)
    }

    fn propagate_backward(
        &mut self,
        output_gradient: &Tensor,
    ) -> Result<Tensor, LayerError> {
        let batch_size = output_gradient.dims()[0];
        let mut shape: Vec<usize> = Vec::with_capacity(self.input_shape.len() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(&self.input_shape);

        let grad_elements: usize = output_gradient.dims().iter().product();
        let target_elements: usize = shape.iter().product();
        if grad_elements != target_elements {
            return Err(LayerError::DimensionMismatch);
        }
        Ok(output_gradient.reshape(shape.as_slice())?)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
