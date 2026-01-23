use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use rand::thread_rng;

use crate::tensor::{Device, Tensor};

pub enum InitializerType {
    He,
    RandomNormal(f64, f64),
    GlorotUniform,
}

impl InitializerType {
    pub fn initialize(
        &self,
        fan_in: usize,
        fan_out: usize,
        shape: &[usize],
        device: Device,
    ) -> Result<Tensor, crate::error::NeuralNetworkError> {
        let size: usize = shape.iter().product();
        let mut rng = thread_rng();

        let data: Vec<f32> = match self {
            InitializerType::He => {
                let std_dev = (2.0 / fan_in as f64).sqrt();
                let normal = Normal::new(0.0, std_dev).expect("Can't create normal distribution");
                (0..size).map(|_| normal.sample(&mut rng) as f32).collect()
            }
            InitializerType::RandomNormal(mean, std_dev) => {
                let normal =
                    Normal::new(*mean, *std_dev).expect("Can't create normal distribution");
                (0..size).map(|_| normal.sample(&mut rng) as f32).collect()
            }
            InitializerType::GlorotUniform => {
                let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
                let uniform = Uniform::new(-limit, limit);
                (0..size).map(|_| uniform.sample(&mut rng) as f32).collect()
            }
        };

        Ok(Tensor::new(data, shape.to_vec(), device)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(data: &[f32]) -> f32 {
        data.iter().sum::<f32>() / data.len() as f32
    }

    fn std_dev(data: &[f32]) -> f32 {
        let m: f32 = mean(data);
        let variance: f32 = data.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    }

    #[test]
    fn test_he_initialization() {
        let fan_in: usize = 100;
        let fan_out: usize = 50;
        let shape: Vec<usize> = vec![fan_in, fan_out];

        let tensor: Tensor = InitializerType::He
            .initialize(fan_in, fan_out, &shape, Device::CPU)
            .unwrap();

        let data: Vec<f32> = tensor.to_vec().unwrap();
        assert_eq!(data.len(), fan_in * fan_out);

        let m: f32 = mean(&data);
        let std: f32 = std_dev(&data);
        let expected_std: f32 = (2.0 / fan_in as f32).sqrt();

        assert!(m.abs() < 0.1);
        assert!((std - expected_std).abs() < 0.1);
    }

    #[test]
    fn test_random_normal_initialization() {
        let fan_in: usize = 100;
        let fan_out: usize = 50;
        let shape: Vec<usize> = vec![fan_in, fan_out];
        let mean_val: f64 = 2.0;
        let std_val: f64 = 0.5;

        let tensor: Tensor = InitializerType::RandomNormal(mean_val, std_val)
            .initialize(fan_in, fan_out, &shape, Device::CPU)
            .unwrap();

        let data: Vec<f32> = tensor.to_vec().unwrap();
        assert_eq!(data.len(), fan_in * fan_out);

        let m: f32 = mean(&data);
        let std: f32 = std_dev(&data);

        assert!((m - mean_val as f32).abs() < 0.1);
        assert!((std - std_val as f32).abs() < 0.1);
    }

    #[test]
    fn test_glorot_uniform_initialization() {
        let fan_in: usize = 100;
        let fan_out: usize = 50;
        let shape: Vec<usize> = vec![fan_in, fan_out];

        let tensor: Tensor = InitializerType::GlorotUniform
            .initialize(fan_in, fan_out, &shape, Device::CPU)
            .unwrap();

        let data: Vec<f32> = tensor.to_vec().unwrap();
        assert_eq!(data.len(), fan_in * fan_out);

        let limit: f32 = (6.0 / (fan_in + fan_out) as f32).sqrt();

        for &val in &data {
            assert!(val >= -limit && val <= limit);
        }

        let m: f32 = mean(&data);
        assert!(m.abs() < 0.1);
    }

    #[test]
    fn test_initialization_shape() {
        let shapes: Vec<Vec<usize>> = vec![vec![10, 20], vec![5, 5, 5], vec![100]];

        for shape in shapes {
            let tensor: Tensor = InitializerType::He
                .initialize(10, 20, &shape, Device::CPU)
                .unwrap();

            assert_eq!(tensor.shape(), shape.as_slice());
            let expected_size: usize = shape.iter().product();
            assert_eq!(tensor.to_vec().unwrap().len(), expected_size);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_he_initialization_metal() {
        let device: Result<Device, _> = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device: Device = device.unwrap();

        let fan_in: usize = 100;
        let fan_out: usize = 50;
        let shape: Vec<usize> = vec![fan_in, fan_out];

        let tensor: Tensor = InitializerType::He
            .initialize(fan_in, fan_out, &shape, device)
            .unwrap();

        let data: Vec<f32> = tensor.to_vec().unwrap();
        assert_eq!(data.len(), fan_in * fan_out);

        let m: f32 = mean(&data);
        let std: f32 = std_dev(&data);
        let expected_std: f32 = (2.0 / fan_in as f32).sqrt();

        assert!(m.abs() < 0.1);
        assert!((std - expected_std).abs() < 0.1);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_glorot_uniform_initialization_metal() {
        let device: Result<Device, _> = Device::new_metal(0);
        if device.is_err() {
            return;
        }
        let device: Device = device.unwrap();

        let fan_in: usize = 100;
        let fan_out: usize = 50;
        let shape: Vec<usize> = vec![fan_in, fan_out];

        let tensor: Tensor = InitializerType::GlorotUniform
            .initialize(fan_in, fan_out, &shape, device)
            .unwrap();

        let data: Vec<f32> = tensor.to_vec().unwrap();
        assert_eq!(data.len(), fan_in * fan_out);

        let limit: f32 = (6.0 / (fan_in + fan_out) as f32).sqrt();

        for &val in &data {
            assert!(val >= -limit && val <= limit);
        }

        let m: f32 = mean(&data);
        assert!(m.abs() < 0.1);
    }
}
