use log::error;
use candle_core::Tensor;

fn check_nan(tensor: &Tensor, operation: &str) {
    if let Ok(data) = tensor.flatten_all().and_then(|t| t.to_vec1::<f64>()) {
        if data.iter().any(|&x| x.is_nan()) {
            error!("NaN detected after {} operation", operation);
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Default, Copy, PartialOrd, Ord, Hash)]
pub enum Activation {
    #[default]
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
}

impl Activation {
    /// Apply the activation function to each element of a multidimensional tensor
    /// dimensions doesn't matter as the transformation is applied element wise
    /// except for the softmax function, the softmax will be computed onto each batch independently
    /// if the tensor is of shape (n, i) with **n** the number of batch and **i** the size of the
    /// vector, the function will return a tensor of same shape, with softmax function computed
    /// for every element in the outermost dimension.
    /// # Arguments
    /// * `input` - a multidimensional tensor;
    pub fn apply(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let result = match self {
            Self::ReLU => {
                let zeros = input.zeros_like()?;
                input.maximum(&zeros)?
            },
            Self::Tanh => input.tanh()?,
            Self::Sigmoid => {
                let neg_input = input.neg()?;
                let exp_neg = neg_input.exp()?;
                let one_plus_exp = (exp_neg + 1.0)?;
                one_plus_exp.recip()?
            },
            Self::Softmax => {
                // For numerical stability, subtract max from each row
                let max_logits = input.max_keepdim(1)?;
                let shifted = input.broadcast_sub(&max_logits)?;
                let exps = shifted.exp()?;
                let sum_exps = exps.sum_keepdim(1)?;
                // Add small epsilon for numerical stability
                let sum_exps_stable = (sum_exps + 1e-10)?;
                exps.broadcast_div(&sum_exps_stable)?
            }
        };
        check_nan(&result, &format!("{:?}", self));
        Ok(result)
    }

    /// Apply the activation function derivative to each element of a multidimensional tensor
    /// not that the dimensions doesn't matter as the transformation is applied element wise.
    /// # Arguments
    /// * `input` - a multidimensional tensor;
    pub fn apply_derivative(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let result = match self {
            Self::ReLU => {
                let zeros = input.zeros_like()?;
                input.gt(&zeros)?.to_dtype(candle_core::DType::F64)?
            },
            Self::Tanh => {
                let tanh_val = input.tanh()?;
                let tanh_sq = tanh_val.sqr()?;
                (1.0 - tanh_sq)?
            },
            Self::Sigmoid => {
                let sigmoid_output = self.apply(input)?;
                let one_minus_sigmoid = (1.0 - &sigmoid_output)?;
                (&sigmoid_output * one_minus_sigmoid)?
            }
            Self::Softmax => unimplemented!("We don't use the softmax jacobian matrix in practice"),
        };
        check_nan(&result, &format!("{:?}", self));
        Ok(result)
    }
}
