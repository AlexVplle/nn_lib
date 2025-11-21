use candle_core::Tensor;
use std::any::Any;

use crate::initialization::InitializerType;

use super::{Layer, LayerError, Trainable};

// TODO: Implement ConvolutionalLayer with Candle tensors
#[derive(Debug, Clone)]
pub struct ConvolutionalLayer {
    kernels: Tensor,
    bias: Tensor,
    input: Option<Tensor>,
    kernel_gradient: Option<Tensor>,
    bias_gradient: Option<Tensor>,

    input_size: (usize, usize, usize),
    output_size: (usize, usize, usize),
    kernels_size: (usize, usize, usize, usize),
}

impl ConvolutionalLayer {
    pub fn new(
        input_size: (usize, usize, usize),
        kernel_size: (usize, usize),
        number_of_kernel: usize,
        init: InitializerType,
    ) -> Self {
        let (kernel_height, kernel_width) = kernel_size;
        let (input_height, input_width, input_channel) = input_size;

        let output_size = (
            input_height - kernel_height + 1,
            input_width - kernel_width + 1,
            number_of_kernel,
        );
        let (output_height, output_width, output_channel) = output_size;

        Self {
            kernels: init.initialize(
                input_height * input_width * input_channel,
                output_height * output_width * output_channel,
                &[kernel_height, kernel_width, input_channel, number_of_kernel],
            ),
            bias: init.initialize(
                input_height * input_width * input_channel,
                output_height * output_width * output_channel,
                &[number_of_kernel],
            ),
            input: None,
            kernel_gradient: None,
            bias_gradient: None,
            input_size,
            output_size,
            kernels_size: (kernel_height, kernel_width, input_channel, number_of_kernel),
        }
    }
}

impl Layer for ConvolutionalLayer {
    fn feed_forward_save(&mut self, _input: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!("ConvolutionalLayer not yet implemented with Candle tensors")
    }

    fn feed_forward(&self, _input: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!("ConvolutionalLayer not yet implemented with Candle tensors")
    }

    fn propagate_backward(&mut self, _output_gradient: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!("ConvolutionalLayer not yet implemented with Candle tensors")
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
        vec![self.kernels.clone(), self.bias.clone()]
    }

    fn get_parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.kernels, &mut self.bias]
    }

    fn get_gradients(&self) -> Vec<Tensor> {
        vec![
            self.kernel_gradient
                .as_ref()
                .expect("Illegal access to unset kernel gradient")
                .clone(),
            self.bias_gradient
                .as_ref()
                .expect("Illegal access to unset bias gradient")
                .clone(),
        ]
    }
}
