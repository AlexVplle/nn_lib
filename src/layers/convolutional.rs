use std::any::Any;

use crate::initialization::InitializerType;
use crate::tensor::Tensor;

use crate::error::NeuralNetworkError;
use super::{Layer, Trainable};

#[derive(Debug, Clone)]
pub struct ConvolutionalLayer {
    input_size: (usize, usize, usize),
    output_size: (usize, usize, usize),
    kernels_size: (usize, usize, usize, usize),
}

impl ConvolutionalLayer {
    pub fn new(
        input_size: (usize, usize, usize),
        kernel_size: (usize, usize),
        number_of_kernel: usize,
        _init: InitializerType,
    ) -> Self {
        let (kernel_height, kernel_width): (usize, usize) = kernel_size;
        let (input_height, input_width, input_channel): (usize, usize, usize) = input_size;

        let output_size: (usize, usize, usize) = (
            input_height - kernel_height + 1,
            input_width - kernel_width + 1,
            number_of_kernel,
        );

        Self {
            input_size,
            output_size,
            kernels_size: (kernel_height, kernel_width, input_channel, number_of_kernel),
        }
    }
}

impl Layer for ConvolutionalLayer {
    fn feed_forward_save(&mut self, _input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        unimplemented!("ConvolutionalLayer not yet implemented with Tensor")
    }

    fn feed_forward(&self, _input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        unimplemented!("ConvolutionalLayer not yet implemented with Tensor")
    }

    fn propagate_backward(
        &mut self,
        _output_gradient: &Tensor,
    ) -> Result<Tensor, NeuralNetworkError> {
        unimplemented!("ConvolutionalLayer not yet implemented with Tensor")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Trainable for ConvolutionalLayer {
    fn get_parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn get_parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn get_gradients(&self) -> Vec<Tensor> {
        vec![]
    }
}
