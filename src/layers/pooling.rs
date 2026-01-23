use std::any::Any;

use crate::error::NeuralNetworkError;
use crate::tensor::Tensor;
use super::Layer;

#[derive(Debug, Clone)]
pub struct MaxPoolingLayer {
    input_size: (usize, usize, usize),
    output_size: (usize, usize, usize),
    pool_size: (usize, usize),
}

impl MaxPoolingLayer {
    pub fn new(input_size: (usize, usize, usize), pool_size: (usize, usize)) -> Self {
        let (input_height, input_width, input_channel): (usize, usize, usize) = input_size;
        let (pool_height, pool_width): (usize, usize) = pool_size;
        let output_size: (usize, usize, usize) = (
            input_height / pool_height,
            input_width / pool_width,
            input_channel,
        );
        Self {
            input_size,
            output_size,
            pool_size,
        }
    }
}

impl Layer for MaxPoolingLayer {
    fn feed_forward_save(&mut self, _input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        unimplemented!("MaxPoolingLayer not yet implemented with Tensor")
    }

    fn feed_forward(&self, _input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        unimplemented!("MaxPoolingLayer not yet implemented with Tensor")
    }

    fn propagate_backward(
        &mut self,
        _output_gradient: &Tensor,
    ) -> Result<Tensor, NeuralNetworkError> {
        unimplemented!("MaxPoolingLayer not yet implemented with Tensor")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
