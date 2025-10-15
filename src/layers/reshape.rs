use ndarray::{ArrayD, Dimension, IxDyn, ShapeError};
use std::any::Any;

use super::{Layer, LayerError};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReshapeLayer {
    input: Option<ArrayD<f64>>,
    input_shape: IxDyn,
    output_shape: IxDyn,
}

impl ReshapeLayer {
    pub fn new(input_shape: &[usize], output_shape: &[usize]) -> Result<Self, LayerError> {
        let input_elements: usize = input_shape.iter().product();
        let output_elements: usize = output_shape.iter().product();
        if input_elements != output_elements {
            return Err(LayerError::ReshapeError(ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            )));
        }
        Ok(Self {
            input: None,
            input_shape: IxDyn(input_shape),
            output_shape: IxDyn(output_shape),
        })
    }
}

impl Layer for ReshapeLayer {
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        self.input = Some(input.clone());
        self.feed_forward(input)
    }

    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        let batch_size: usize = input.shape()[0];
        let mut shape: Vec<usize> = Vec::with_capacity(self.output_shape.ndim() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(self.output_shape.as_array_view().as_slice().unwrap());

        if input.shape().iter().product::<usize>() != shape.iter().product() {
            return Err(LayerError::ReshapeError(ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            )));
        }
        Ok(input.clone().into_shape(shape).unwrap())
    }

    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, LayerError> {
        let batch_size: usize = output_gradient.shape()[0];
        let mut shape: Vec<usize> = Vec::with_capacity(self.output_shape.ndim() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(self.input_shape.as_array_view().as_slice().unwrap());
        if output_gradient.shape().iter().product::<usize>() != shape.iter().product() {
            return Err(LayerError::ReshapeError(ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            )));
        }
        Ok(output_gradient.clone().into_shape(shape).unwrap())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
