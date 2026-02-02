use crate::error::NeuralNetworkError;
use crate::tensor::Tensor;
use ndarray::ArrayD;

use super::{MulticlassClassifierMetrics, MulticlassMetricType};

#[derive(Clone, PartialEq, Debug)]
pub enum Metrics {
    MulticlassClassification(MulticlassClassifierMetrics),
}

impl Metrics {
    pub fn multiclass_classification(metric_types: &Vec<MulticlassMetricType>) -> Self {
        Metrics::MulticlassClassification(MulticlassClassifierMetrics::from(metric_types))
    }

    /// Accumulate metrics from tensor predictions and labels
    ///
    /// # Arguments
    /// * `predictions` - Model predictions tensor, shape: [batch_size, n_classes]
    /// * `observed` - One-hot encoded labels tensor, shape: [batch_size, n_classes]
    pub fn accumulate(
        &mut self,
        predictions: &Tensor,
        observed: &Tensor,
    ) -> Result<(), NeuralNetworkError> {
        use crate::tensor::Device;
        let pred_cpu = predictions.to_device(Device::CPU)?;
        let obs_cpu = observed.to_device(Device::CPU)?;

        let pred_arr: ArrayD<f64> = pred_cpu.into();
        let obs_arr: ArrayD<f64> = obs_cpu.into();

        self.accumulate_internal(&pred_arr, &obs_arr);
        Ok(())
    }

    fn accumulate_internal(&mut self, predictions: &ArrayD<f64>, observed: &ArrayD<f64>) {
        match self {
            Metrics::MulticlassClassification(metrics) => metrics.accumulate(predictions, observed),
        }
    }

    pub fn finalize(&mut self) {
        match self {
            Metrics::MulticlassClassification(metrics) => metrics.finalize(),
        }
    }

    pub fn get_metric(&self, metric_type: MulticlassMetricType) -> Option<f64> {
        match self {
            Metrics::MulticlassClassification(metrics) => metrics.get_metric(metric_type),
        }
    }
}
