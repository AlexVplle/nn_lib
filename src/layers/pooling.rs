use candle_core::Tensor;
use std::any::Any;

use super::{Layer, LayerError};

// TODO: Implement MaxPoolingLayer with Candle tensors
#[derive(Debug, Clone)]
pub struct MaxPoolingLayer {
    input: Option<Tensor>,
    input_size: (usize, usize, usize),
    output_size: (usize, usize, usize),
    pool_size: (usize, usize),
}

impl MaxPoolingLayer {
    pub fn new(input_size: (usize, usize, usize), pool_size: (usize, usize)) -> Self {
        let (input_height, input_width, channels) = input_size;
        let (pool_height, pool_width) = pool_size;

        let output_size = (
            input_height / pool_height,
            input_width / pool_width,
            channels,
        );

        Self {
            input: None,
            input_size,
            output_size,
            pool_size,
        }
    }
}

impl Layer for MaxPoolingLayer {
    fn feed_forward_save(&mut self, _input: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!("MaxPoolingLayer not yet implemented with Candle tensors")
    }

    fn feed_forward(&self, _input: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!("MaxPoolingLayer not yet implemented with Candle tensors")
    }

    fn propagate_backward(&mut self, _output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!("MaxPoolingLayer not yet implemented with Candle tensors")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
