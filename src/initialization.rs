use candle_core::{Tensor, Device};

pub enum InitializerType {
    He,
    RandomNormal(f64, f64),
    GlorotUniform,
}

impl InitializerType {
    /// Return a new multidimensional tensor initialized according to the `InitializerType`
    ///
    /// # Arguments
    /// * `fan_in` - The number of input in the layer
    /// * `fan_out` The number of output in the layer
    /// * `shape` - output tensor shape
    pub fn initialize(&self, fan_in: usize, fan_out: usize, shape: &[usize], device: &Device) -> Tensor {
        match self {
            InitializerType::He => {
                let std_dev = (2.0 / fan_in as f64).sqrt();
                Tensor::randn(0.0, std_dev, shape, device).expect("Failed to create He initialized tensor")
            }
            InitializerType::RandomNormal(mean, std_dev) => {
                Tensor::randn(*mean, *std_dev, shape, device).expect("Failed to create normal initialized tensor")
            }
            InitializerType::GlorotUniform => {
                let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
                let uniform = Tensor::rand(-limit, limit, shape, device).expect("Failed to create Glorot initialized tensor");
                uniform
            }
        }
    }
}
