use std::any::Any;

use crate::initialization::InitializerType;
use crate::tensor::Tensor;

use super::{Layer, Trainable};
use crate::error::NeuralNetworkError;

#[derive(Debug, Clone)]
pub struct DenseLayer {
    weights: Tensor,
    bias: Tensor,
    last_batch_input: Option<Tensor>,
    weights_gradient: Option<Tensor>,
    biases_gradient: Option<Tensor>,
    input_size: usize,
    output_size: usize,
}

impl DenseLayer {
    /// Create a new `DenseLayer` filling it with random value. see `InitializerType` for
    /// initialization parameters
    pub fn new(
        input_size: usize,
        output_size: usize,
        init: InitializerType,
        device: crate::tensor::Device,
    ) -> Result<Self, NeuralNetworkError> {
        Ok(Self {
            weights: init.initialize(
                input_size,
                output_size,
                &[input_size, output_size],
                device.clone(),
            )?,
            bias: init.initialize(input_size, output_size, &[output_size], device)?,
            last_batch_input: None,
            weights_gradient: None,
            biases_gradient: None,
            input_size,
            output_size,
        })
    }

    pub fn set_weights(&mut self, weights: Tensor) {
        self.weights = weights;
    }

    pub fn set_bias(&mut self, bias: Tensor) {
        self.bias = bias;
    }
}

impl Layer for DenseLayer {
    /// Return output matrices of this `DenseLayer` (shape (n, j)), while storing the input matrices
    /// (shape (n, i))
    ///
    /// where **n** is the number of samples, **j** is layer output size and **i** is layer
    /// input size.
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward_save(&mut self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        self.last_batch_input = Some(input.clone());
        self.feed_forward(input)
    }

    /// Return output matrices of this `DenseLayer` (shape (n, j))
    ///
    /// where **n** is the number of samples, **j** is layer output size.
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward(&self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        let output = input.matmul(&self.weights)?;
        // Add bias to each row (broadcast)
        add_bias_to_output(&output, &self.bias)
    }

    /// Return the input gradient vector (shape (n, i)), by processing the output gradient vector
    /// (shape (n, j)).
    ///
    /// This function also computes and stores the current batch weights and biases gradient in the layer.
    ///
    /// # Arguments
    /// * `output_gradient` - (shape (n, j))
    fn propagate_backward(
        &mut self,
        output_gradient: &Tensor,
    ) -> Result<Tensor, NeuralNetworkError> {
        match self.last_batch_input.as_ref() {
            Some(input) => {
                let input_t = input.transpose()?;
                let weights_t = self.weights.transpose()?;

                let weights_gradient = input_t.matmul(output_gradient)?;
                self.weights_gradient = Some(weights_gradient);

                let biases_gradient = output_gradient.sum_axis(0)?;
                self.biases_gradient = Some(biases_gradient);

                let input_gradient = output_gradient.matmul(&weights_t)?;
                Ok(input_gradient)
            }
            None => Err(NeuralNetworkError::IllegalInputAccess),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn add_bias_to_output(output: &Tensor, bias: &Tensor) -> Result<Tensor, NeuralNetworkError> {
    let batch_size = output.shape()[0];
    let output_vec = output.to_vec()?;
    let bias_vec = bias.to_vec()?;

    let mut result = Vec::with_capacity(output_vec.len());
    let output_size = bias_vec.len();

    for i in 0..batch_size {
        let start_idx = i * output_size;

        for j in 0..output_size {
            let output_val = output_vec[start_idx + j];
            let bias_val = bias_vec[j];
            result.push(output_val + bias_val);
        }
    }

    let output_shape = vec![batch_size, output_size];
    Ok(crate::tensor::Tensor::new(
        result,
        output_shape,
        bias.device().clone(),
    )?)
}

impl Trainable for DenseLayer {
    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }

    fn get_parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn get_gradients(&self) -> Vec<Tensor> {
        vec![
            self.weights_gradient
                .as_ref()
                .expect("Illegal access to unset weights gradient")
                .clone(),
            self.biases_gradient
                .as_ref()
                .expect("Illegal access to unset biases gradient")
                .clone(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    #[test]
    fn test_dense_layer_new() {
        let input_size: usize = 10;
        let output_size: usize = 5;

        let layer: DenseLayer = DenseLayer::new(
            input_size,
            output_size,
            InitializerType::He,
            Device::CPU,
        )
        .unwrap();

        let params: Vec<Tensor> = layer.get_parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[input_size, output_size]);
        assert_eq!(params[1].shape(), &[output_size]);
    }

    #[test]
    fn test_dense_layer_feed_forward() {
        let weights_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weights: Tensor = Tensor::new(weights_data, vec![2, 2], Device::CPU).unwrap();

        let bias_data: Vec<f32> = vec![0.5, 1.0];
        let bias: Tensor = Tensor::new(bias_data, vec![2], Device::CPU).unwrap();

        let layer: DenseLayer = DenseLayer {
            weights,
            bias,
            last_batch_input: None,
            weights_gradient: None,
            biases_gradient: None,
            input_size: 2,
            output_size: 2,
        };

        let input: Tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2], Device::CPU).unwrap();
        let output: Tensor = layer.feed_forward(&input).unwrap();
        let result: Vec<f32> = output.to_vec().unwrap();

        assert_eq!(result, vec![7.5, 11.0]);
    }

    #[test]
    fn test_dense_layer_feed_forward_save() {
        let weights_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weights: Tensor = Tensor::new(weights_data, vec![2, 2], Device::CPU).unwrap();

        let bias_data: Vec<f32> = vec![0.5, 1.0];
        let bias: Tensor = Tensor::new(bias_data, vec![2], Device::CPU).unwrap();

        let mut layer: DenseLayer = DenseLayer {
            weights,
            bias,
            last_batch_input: None,
            weights_gradient: None,
            biases_gradient: None,
            input_size: 2,
            output_size: 2,
        };

        let input: Tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2], Device::CPU).unwrap();
        let output: Tensor = layer.feed_forward_save(&input).unwrap();

        assert!(layer.last_batch_input.is_some());
        let saved_input: Vec<f32> = layer.last_batch_input.unwrap().to_vec().unwrap();
        assert_eq!(saved_input, vec![1.0, 2.0]);

        let result: Vec<f32> = output.to_vec().unwrap();
        assert_eq!(result, vec![7.5, 11.0]);
    }

    #[test]
    fn test_add_bias_to_output() {
        let output: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let bias: Tensor = Tensor::new(vec![0.5, 1.0], vec![2], Device::CPU).unwrap();

        let result_tensor: Tensor = add_bias_to_output(&output, &bias).unwrap();
        let result: Vec<f32> = result_tensor.to_vec().unwrap();

        assert_eq!(result, vec![1.5, 3.0, 3.5, 5.0]);
    }

    #[test]
    fn test_dense_layer_get_parameters() {
        let layer: DenseLayer = DenseLayer::new(3, 2, InitializerType::He, Device::CPU).unwrap();
        let params: Vec<Tensor> = layer.get_parameters();

        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[3, 2]);
        assert_eq!(params[1].shape(), &[2]);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_dense_layer_new_metal() {
        let device: Result<Device, _> = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device: Device = device.unwrap();

        let input_size: usize = 10;
        let output_size: usize = 5;

        let layer: DenseLayer =
            DenseLayer::new(input_size, output_size, InitializerType::He, device).unwrap();

        let params: Vec<Tensor> = layer.get_parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[input_size, output_size]);
        assert_eq!(params[1].shape(), &[output_size]);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_dense_layer_feed_forward_metal() {
        let device: Result<Device, _> = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device: Device = device.unwrap();

        let weights_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weights: Tensor = Tensor::new(weights_data, vec![2, 2], device.clone()).unwrap();

        let bias_data: Vec<f32> = vec![0.5, 1.0];
        let bias: Tensor = Tensor::new(bias_data, vec![2], device.clone()).unwrap();

        let layer: DenseLayer = DenseLayer {
            weights,
            bias,
            last_batch_input: None,
            weights_gradient: None,
            biases_gradient: None,
            input_size: 2,
            output_size: 2,
        };

        let input: Tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2], device).unwrap();
        let output: Tensor = layer.feed_forward(&input).unwrap();
        let result: Vec<f32> = output.to_vec().unwrap();

        assert_eq!(result, vec![7.5, 11.0]);
    }
}
