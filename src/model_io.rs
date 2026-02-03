use crate::{
    activation::Activation,
    cost::CostFunction,
    error::NeuralNetworkError,
    layers::{ActivationLayer, DenseLayer, Trainable},
    metrics::{Metrics, MulticlassMetricType},
    sequential::{Sequential, SequentialBuilder},
    tensor::{Device, Tensor},
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub enum LayerConfig {
    Dense {
        input_size: usize,
        output_size: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    },
    Activation {
        activation: Activation,
    },
}

#[derive(Serialize, Deserialize)]
pub struct ModelState {
    pub layers: Vec<LayerConfig>,
    pub cost_function: CostFunction,
}

impl Sequential {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), NeuralNetworkError> {
        let mut layer_configs = Vec::new();

        for layer in &self.layers {
            if let Some(dense) = layer.as_any().downcast_ref::<DenseLayer>() {
                let params = dense.get_parameters();
                let weights = params[0].to_vec()?;
                let bias = params[1].to_vec()?;
                let weights_shape = params[0].shape();
                let input_size = weights_shape[0];
                let output_size = weights_shape[1];

                layer_configs.push(LayerConfig::Dense {
                    input_size,
                    output_size,
                    weights,
                    bias,
                });
            } else if let Some(act) = layer.as_any().downcast_ref::<ActivationLayer>() {
                layer_configs.push(LayerConfig::Activation {
                    activation: act.activation(),
                });
            }
        }

        let model_state = ModelState {
            layers: layer_configs,
            cost_function: self.cost_function.clone(),
        };

        let file = File::create(path)
            .map_err(|e| NeuralNetworkError::Other(format!("Failed to create file: {}", e)))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &model_state)
            .map_err(|e| NeuralNetworkError::Other(format!("Failed to serialize model: {}", e)))?;

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(
        path: P,
        device: Device,
    ) -> Result<Sequential, NeuralNetworkError> {
        let file = File::open(path)
            .map_err(|e| NeuralNetworkError::Other(format!("Failed to open file: {}", e)))?;
        let reader = BufReader::new(file);
        let model_state: ModelState = serde_json::from_reader(reader)
            .map_err(|e| NeuralNetworkError::Other(format!("Failed to deserialize model: {}", e)))?;

        let mut builder = SequentialBuilder::new();

        for layer_config in model_state.layers {
            match layer_config {
                LayerConfig::Dense {
                    input_size,
                    output_size,
                    weights,
                    bias,
                } => {
                    let weights_tensor = Tensor::new(
                        weights,
                        vec![input_size, output_size],
                        device.clone(),
                    )?;
                    let bias_tensor = Tensor::new(bias, vec![output_size], device.clone())?;

                    let mut dense_layer = DenseLayer::new(
                        input_size,
                        output_size,
                        crate::initialization::InitializerType::He,
                        device.clone(),
                    )?;
                    dense_layer.set_weights(weights_tensor);
                    dense_layer.set_bias(bias_tensor);

                    builder = builder.push(dense_layer);
                }
                LayerConfig::Activation { activation } => {
                    builder = builder.push(ActivationLayer::from(activation));
                }
            }
        }

        let metrics = Metrics::multiclass_classification(&vec![MulticlassMetricType::Accuracy]);

        let model = builder
            .with_metrics(metrics)
            .compile(
                crate::optimizer::GradientDescent::new(0.01),
                model_state.cost_function,
            )?;

        Ok(model)
    }
}
