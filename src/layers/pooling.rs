use ndarray::{s, ArrayD, IxDyn};
use std::any::Any;

use crate::error::NeuralNetworkError;
use super::Layer;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MaxPoolingLayer {
    input: Option<ArrayD<f64>>,
    max_indices: Option<ArrayD<usize>>,
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
            input: None,
            max_indices: None,
            input_size,
            output_size,
            pool_size,
        }
    }

    fn find_max_indices(&mut self) {
        let input = self
            .input
            .as_ref()
            .expect("Input not set. Call feed_forward first.");

        let batch_size: usize = input.shape()[0];
        let (pool_height, pool_width): (usize, usize) = self.pool_size;
        let (output_height, output_width, output_channel) = self.output_size;

        let mut max_indices: ArrayD<usize> = ArrayD::zeros(IxDyn(&[
            batch_size,
            output_height,
            output_width,
            output_channel,
        ]));
        for batch_index in 0..batch_size {
            for channel in 0..output_channel {
                for y in 0..output_height {
                    for x in 0..output_width {
                        let height_start: usize = y * pool_height;
                        let width_start: usize = x * pool_width;
                        let window = input.slice(s![
                            batch_index,
                            height_start..height_start + pool_height,
                            width_start..width_start + pool_width,
                            channel
                        ]);
                        let (max_index, _) = window.indexed_iter().fold(
                            (0, f64::MIN),
                            |(max_idx, max_value), (idx, &val)| {
                                if val > max_value {
                                    (idx.0 * window.ncols() + idx.1, val)
                                } else {
                                    (max_idx, max_value)
                                }
                            },
                        );
                        max_indices[[batch_index, y, x, channel]] = max_index;
                    }
                }
            }
        }
        self.max_indices = Some(max_indices);
    }
}

impl Layer for MaxPoolingLayer {
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NeuralNetworkError> {
        self.input = Some(input.clone());
        self.find_max_indices();
        self.feed_forward(input)
    }

    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, NeuralNetworkError> {
        let batch_size: usize = input.shape()[0];
        let (pool_height, pool_width): (usize, usize) = self.pool_size;
        let (output_height, output_width, output_channel) = self.output_size;

        let max_indices = self
            .max_indices
            .as_ref()
            .expect("Max_indices not set. Call feed_forward first.");

        let mut output: ArrayD<f64> = ArrayD::zeros(IxDyn(&[
            batch_size,
            output_height,
            output_width,
            output_channel,
        ]));

        for batch_index in 0..batch_size {
            for channel in 0..output_channel {
                for y in 0..output_height {
                    for x in 0..output_width {
                        let index = max_indices[[batch_index, y, x, channel]];
                        let height_start = y * pool_height;
                        let width_start = x * pool_width;
                        let dy = index / pool_height;
                        let dx = index % pool_width;
                        output[[batch_index, y, x, channel]] =
                            input[[batch_index, height_start + dy, width_start + dx, channel]];
                    }
                }
            }
        }

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

        let max_indices = self
            .max_indices
            .as_ref()
            .expect("Max_indices not set. Call feed_forward first");

        let batch_size: usize = input.shape()[0];
        let (input_height, input_width, input_channel): (usize, usize, usize) = self.input_size;
        let (output_height, output_width, output_channel) = self.output_size;
        let (pool_height, pool_width): (usize, usize) = self.pool_size;

        let mut input_gradient: ArrayD<f64> = ArrayD::zeros(IxDyn(&[
            batch_size,
            input_height,
            input_width,
            input_channel,
        ]));

        for batch_index in 0..batch_size {
            for channel in 0..output_channel {
                for y in 0..output_height {
                    for x in 0..output_width {
                        let index = max_indices[[batch_index, y, x, channel]];
                        let height_start = y * pool_height;
                        let width_start = x * pool_width;
                        let dy = index / pool_height;
                        let dx = index % pool_width;
                        input_gradient
                            [[batch_index, height_start + dy, width_start + dx, channel]] +=
                            output_gradient[[batch_index, y, x, channel]];
                    }
                }
            }
        }
        Ok(input_gradient)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
