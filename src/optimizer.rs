use crate::layers::Trainable;

pub trait Optimizer: Sync + Send {
    fn get_learning_rate(&self) -> f64;
    fn step(&mut self, layer: &mut dyn Trainable);
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
pub struct GradientDescent {
    learning_rate: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        GradientDescent { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn step(&mut self, layer: &mut dyn Trainable) {
        let gradients = layer.get_gradients();
        let mut parameters = layer.get_parameters_mut();

        for (param, gradient) in parameters.iter_mut().zip(gradients.iter()) {
            let scaled_gradient = gradient
                .mul_scalar(self.learning_rate as f32)
                .expect("Failed to scale gradient");
            **param = param
                .sub(&scaled_gradient)
                .expect("Failed to update parameter");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialization::InitializerType;
    use crate::layers::dense::DenseLayer;
    use crate::layers::Layer;
    use crate::tensor::{Device, Tensor};

    #[test]
    fn test_gradient_descent_step() {
        let mut layer: DenseLayer =
            DenseLayer::new(2, 2, InitializerType::He, Device::CPU).unwrap();

        let input: Tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2], Device::CPU).unwrap();
        let _output: Tensor = layer.feed_forward_save(&input).unwrap();

        let output_gradient: Tensor = Tensor::new(vec![0.1, 0.2], vec![1, 2], Device::CPU).unwrap();
        let _input_gradient: Tensor = layer.propagate_backward(&output_gradient).unwrap();

        let params_before: Vec<Tensor> = layer.get_parameters();
        let weights_before: Vec<f32> = params_before[0].to_vec().unwrap();

        let mut optimizer: GradientDescent = GradientDescent::new(0.01);
        optimizer.step(&mut layer);

        let params_after: Vec<Tensor> = layer.get_parameters();
        let weights_after: Vec<f32> = params_after[0].to_vec().unwrap();

        for (before, after) in weights_before.iter().zip(weights_after.iter()) {
            assert_ne!(before, after);
        }
    }

    #[test]
    fn test_gradient_descent_learning_rate() {
        let optimizer: GradientDescent = GradientDescent::new(0.1);
        assert_eq!(optimizer.get_learning_rate(), 0.1);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_gradient_descent_step_metal() {
        let device: Result<Device, _> = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device: Device = device.unwrap();

        let mut layer: DenseLayer =
            DenseLayer::new(2, 2, InitializerType::He, device.clone()).unwrap();

        let input: Tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2], device.clone()).unwrap();
        let _output: Tensor = layer.feed_forward_save(&input).unwrap();

        let output_gradient: Tensor = Tensor::new(vec![0.1, 0.2], vec![1, 2], device).unwrap();
        let _input_gradient: Tensor = layer.propagate_backward(&output_gradient).unwrap();

        let params_before: Vec<Tensor> = layer.get_parameters();
        let weights_before: Vec<f32> = params_before[0].to_vec().unwrap();

        let mut optimizer: GradientDescent = GradientDescent::new(0.01);
        optimizer.step(&mut layer);

        let params_after: Vec<Tensor> = layer.get_parameters();
        let weights_after: Vec<f32> = params_after[0].to_vec().unwrap();

        for (before, after) in weights_before.iter().zip(weights_after.iter()) {
            assert_ne!(before, after);
        }
    }
}
