use ndarray::{linalg, Array2, ArrayD, Axis, IxDyn};
use std::any::Any;

use crate::initialization::InitializerType;

use crate::error::NeuralNetworkError;
use super::{Layer, Trainable};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConvolutionalLayer {
    kernels: ArrayD<f64>,
    bias: ArrayD<f64>,
    input: Option<ArrayD<f64>>,
    kernel_gradient: Option<ArrayD<f64>>,
    bias_gradient: Option<ArrayD<f64>>,

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
        let (kernel_height, kernel_width): (usize, usize) = kernel_size;
        let (input_height, input_width, input_channel): (usize, usize, usize) = input_size;

        let output_size: (usize, usize, usize) = (
            input_height - kernel_height + 1,
            input_width - kernel_width + 1,
            number_of_kernel,
        );
        let (output_height, output_width, output_channel): (usize, usize, usize) = output_size;

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

    fn flip_kernels(&self) -> ArrayD<f64> {
        // Ensure the kernels array is 4D
        assert_eq!(self.kernels.ndim(), 4);

        let kernel_h = self.kernels.shape()[0];
        let kernel_w = self.kernels.shape()[1];
        let kernel_d = self.kernels.shape()[2];
        let num_kernels = self.kernels.shape()[3];

        let mut flipped_kernels =
            ArrayD::zeros(IxDyn(&[kernel_h, kernel_w, kernel_d, num_kernels]));

        for ky in 0..kernel_h {
            for kx in 0..kernel_w {
                for c in 0..kernel_d {
                    for nk in 0..num_kernels {
                        let flipped_ky = kernel_h - 1 - ky;
                        let flipped_kx = kernel_w - 1 - kx;
                        flipped_kernels[[flipped_ky, flipped_kx, c, nk]] =
                            self.kernels[[ky, kx, c, nk]];
                    }
                }
            }
        }
        flipped_kernels
    }

    fn im2col(&self, input: ArrayD<f64>) -> Array2<f64> {
        assert_eq!(input.ndim(), 4);

        let batch_size = input.shape()[0];
        let (input_h, input_w, input_channels) = self.input_size;
        let (kernel_h, kernel_w, kernel_d, _num_kernels) = self.kernels_size;
        let (output_h, output_w, _output_channels) = self.output_size;

        assert_eq!(input.shape()[1], input_h);
        assert_eq!(input.shape()[2], input_w);
        assert_eq!(input.shape()[3], input_channels);
        assert_eq!(kernel_d, input_channels); // kernel_depth should match input_channels

        let output_size = output_h * output_w * batch_size;
        let kernel_size = kernel_h * kernel_w * kernel_d;

        let mut output = Array2::zeros((output_size, kernel_size));

        for b in 0..batch_size {
            for y in 0..output_h {
                for x in 0..output_w {
                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            for c in 0..kernel_d {
                                let in_y = y + ky;
                                let in_x = x + kx;
                                let output_row = b * output_h * output_w + y * output_w + x;
                                let output_col = ky * kernel_w * kernel_d + kx * kernel_d + c;
                                output[[output_row, output_col]] = input[[b, in_y, in_x, c]];
                            }
                        }
                    }
                }
            }
        }

        output
    }

    fn im2col_full(&self, output: ArrayD<f64>) -> Array2<f64> {
        assert_eq!(output.ndim(), 4);

        let batch_size = output.shape()[0];
        let (input_h, input_w, input_channels) = self.input_size;
        let (kernel_h, kernel_w, kernel_d, num_kernels) = self.kernels_size;
        let (output_h, output_w, output_channels) = self.output_size;

        assert_eq!(output.shape()[1], output_h);
        assert_eq!(output.shape()[2], output_w);
        assert_eq!(output.shape()[3], output_channels);
        assert_eq!(num_kernels, output_channels);

        // Calculate padding
        let pad_h = kernel_h - 1;
        let pad_w = kernel_w - 1;

        // Calculate the total output size as the product of output height, output width, and batch size
        let input_size = input_h * input_w * batch_size;
        let kernel_size = kernel_h * kernel_w * num_kernels;

        // Initialize the output matrix with zeros, with shape (output_size, kernel_size)
        let mut result = Array2::zeros((input_size, kernel_size));

        // Pad the input tensor
        let mut padded_input: ArrayD<f64> = ArrayD::zeros(IxDyn(&[
            batch_size,
            output_h + 2 * pad_h,
            output_w + 2 * pad_w,
            output_channels,
        ]));
        for b in 0..batch_size {
            for c in 0..output_channels {
                for y in 0..output_h {
                    for x in 0..output_w {
                        padded_input[[b, y + pad_h, x + pad_w, c]] = output[[b, y, x, c]];
                    }
                }
            }
        }

        // Iterate over each element in the batch
        for b in 0..batch_size {
            for y in 0..input_h {
                for x in 0..input_w {
                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            for c in 0..num_kernels {
                                let in_y = y + ky;
                                let in_x = x + kx;
                                let input_row = b * input_h * input_w + y * input_w + x;
                                let input_col = ky * kernel_w * num_kernels + kx * num_kernels + c;
                                result[[input_row, input_col]] = padded_input[[b, in_y, in_x, c]];
                            }
                        }
                    }
                }
            }
        }

        result
    }

    fn convolve(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        let col = self.im2col(input.clone());
        let (kernel_h, kernel_w, kernel_d, num_kernels) = self.kernels_size;
        let (output_h, output_w, output_channels) = self.output_size;
        let batch_size = input.shape()[0];

        let kernel_size = kernel_h * kernel_w * kernel_d;

        let kernels_reshaped = self
            .kernels
            .clone()
            .into_shape((num_kernels, kernel_size))
            .unwrap();

        let col_reshaped = col
            .into_shape((batch_size * output_h * output_w, kernel_size))
            .unwrap();

        let mut result = Array2::zeros((batch_size * output_h * output_w, output_channels));

        linalg::general_mat_mul(1.0, &col_reshaped, &kernels_reshaped.t(), 0.0, &mut result);

        result
            .into_shape(IxDyn(&[batch_size, output_h, output_w, output_channels]))
            .unwrap()
    }

    fn convolve_full(&self, output: &ArrayD<f64>) -> ArrayD<f64> {
        let col = self.im2col_full(output.clone());
        let (kernel_h, kernel_w, kernel_d, num_kernels) = self.kernels_size;
        let (input_h, input_w, input_channels) = self.input_size;
        let batch_size = output.shape()[0];

        let kernel_size = kernel_h * kernel_w * num_kernels;

        let kernels_reshaped = self
            .flip_kernels()
            .clone()
            .into_shape((kernel_d, kernel_size))
            .unwrap();

        let col_reshaped = col
            .into_shape((batch_size * input_h * input_w, kernel_size))
            .unwrap();

        let mut result = Array2::zeros((batch_size * input_h * input_w, input_channels));

        linalg::general_mat_mul(1.0, &col_reshaped, &kernels_reshaped.t(), 0.0, &mut result);

        result
            .into_shape(IxDyn(&[batch_size, input_h, input_w, input_channels]))
            .unwrap()
    }
}

impl Layer for ConvolutionalLayer {
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NeuralNetworkError> {
        self.input = Some(input.clone());
        self.feed_forward(input)
    }

    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NeuralNetworkError> {
        let output = self.convolve(&input.clone());
        Ok(output)
    }

    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, NeuralNetworkError> {
        let input = self
            .input
            .as_ref()
            .expect("Input not set. Call feed_forward first.");

        let (kernel_h, kernel_w, kernel_d, num_kernels) = self.kernels_size;
        let (output_h, output_w, output_channels) = self.output_size;
        let batch_size = input.shape()[0];

        let mut col_input = self.im2col(input.clone());

        let kernel_size = kernel_h * kernel_w * kernel_d;
        let output_gradient_flat = output_gradient
            .clone()
            .into_shape((batch_size * output_h * output_w, output_channels))
            .unwrap();

        // Calculate the gradient with respect to the input (dL/dX) using the convolve function with flipped kernels
        let d_input = self.convolve_full(output_gradient);

        col_input = col_input
            .into_shape((batch_size * output_h * output_w, kernel_size))
            .unwrap();

        // Calculate the gradient with respect to the filters (dL/dW)
        let mut d_kernels = Array2::zeros((num_kernels, kernel_size));
        linalg::general_mat_mul(
            1.0,
            &output_gradient_flat.t(),
            &col_input,
            0.0,
            &mut d_kernels,
        );
        let d_kernels = d_kernels
            .into_shape(IxDyn(&[kernel_h, kernel_w, kernel_d, num_kernels]))
            .unwrap();
        self.kernel_gradient = Some(d_kernels);

        // Calculate the gradient with respect to the biases (dL/db)
        let d_biases = output_gradient
            .sum_axis(Axis(0))
            .sum_axis(Axis(0))
            .sum_axis(Axis(0));
        self.bias_gradient = Some(d_biases);

        Ok(d_input)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Trainable for ConvolutionalLayer {
    fn get_parameters(&self) -> Vec<ArrayD<f64>> {
        vec![
            self.kernels.clone().into_dyn(),
            self.bias.clone().into_dyn(),
        ]
    }

    fn get_parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![&mut self.kernels, &mut self.bias]
    }

    fn get_gradients(&self) -> Vec<ArrayD<f64>> {
        vec![
            self.kernel_gradient
                .as_ref()
                .expect("Illegal access to unset weights gradient")
                .clone()
                .into_dyn(),
            self.bias_gradient
                .as_ref()
                .expect("Illegal access to unset biases gradient")
                .clone()
                .into_dyn(),
        ]
    }
}
