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

    pub fn accumulate(&mut self, predictions: &ArrayD<f64>, observed: &ArrayD<f64>) {
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
