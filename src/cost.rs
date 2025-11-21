use candle_core::Tensor;

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum CostFunction {
    #[default]
    CrossEntropy,
    BinaryCrossEntropy,
    Mse,
}

impl CostFunction {
    /// This crate don't use any kind of auto diff mechanism,
    /// thus, for function like BinaryCrossEntropy and CrossEntropy that need clamped output,
    /// we assume Sigmoid and Softmax respectively as the output activation layer.
    /// the gradient calculation is done with those activation function in mind.
    /// Those function are called 'Output dependant' to contrast with function like Mse, of which
    /// the derivative can be easily calculated with respect to any output layer, because it
    /// doesn't need clamped output.
    pub fn is_output_dependant(&self) -> bool {
        match self {
            Self::BinaryCrossEntropy | Self::CrossEntropy => true,
            Self::Mse => false,
        }
    }

    /// Compute the mean cost of the neural network with respect to a batch `output` and `observed`
    /// # Arguments
    /// * `output` - a batch matrices (shape (n, j)) of output of the network
    /// * `observed` - a one hotted encoded vector of observed values
    pub fn cost(&self, output: &Tensor, observed: &Tensor) -> f64 {
        let epsilon = 1e-7;
        let clipped_output = output.clamp(epsilon, 1.0 - epsilon).expect("Failed to clamp output");

        match self {
            Self::CrossEntropy => {
                let log_output = clipped_output.log().expect("Failed to compute log");
                let losses = (observed * log_output).expect("Failed to multiply");
                let sum_losses = losses.sum_all().expect("Failed to sum").to_scalar::<f64>().expect("Failed to convert to scalar");
                let batch_size = output.dims()[0] as f64;
                -sum_losses / batch_size
            }
            Self::BinaryCrossEntropy => {
                let log_output = clipped_output.log().expect("Failed to compute log");
                let one_minus_output = (1.0 - &clipped_output).expect("Failed to compute 1-output");
                let log_one_minus = one_minus_output.log().expect("Failed to compute log");
                let one_minus_obs = (1.0 - observed).expect("Failed to compute 1-observed");

                let term1 = (observed * log_output).expect("Failed to multiply");
                let term2 = (one_minus_obs * log_one_minus).expect("Failed to multiply");
                let losses = (term1 + term2).expect("Failed to add");
                let mean_loss = losses.mean_all().expect("Failed to compute mean").to_scalar::<f64>().expect("Failed to convert to scalar");
                -mean_loss
            }
            Self::Mse => {
                let diff = (output - observed).expect("Failed to compute diff");
                let squared = diff.sqr().expect("Failed to square");
                squared.mean_all().expect("Failed to compute mean").to_scalar::<f64>().expect("Failed to convert to scalar")
            }
        }
    }

    /// Return the gradient of cost function with respect to `output`
    /// Note that this simple, from 'almost' scratch library don't use auto-differentiation
    /// thus `BinaryCrossEntropy` calculation assume a Sigmoid activation as the layer.
    /// `CrossEntropy` calculation assume a Softmax activation as the last
    /// layer
    /// # Arguments
    /// * `output` - a batch matrices of neural network output (shape (n, j))
    /// * `observed` - a batch matrices of observed values (shape (n, j))
    ///
    /// Note that CrossEntropy and BinaryCrossEntropy assume one hot encoded vector for the
    /// observed vector if the is multi-class.
    pub fn cost_output_gradient(
        &self,
        output: &Tensor,
        observed: &Tensor,
    ) -> Tensor {
        match self {
            Self::CrossEntropy => (output - observed).expect("Failed to compute gradient"),
            Self::BinaryCrossEntropy => (output - observed).expect("Failed to compute gradient"),
            Self::Mse => {
                let batch_size = output.dims()[0] as f64;
                let diff = (output - observed).expect("Failed to compute diff");
                ((diff * 2.0).expect("Failed to multiply") / batch_size).expect("Failed to divide")
            }
        }
    }
}
