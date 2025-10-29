use std::collections::HashMap;

use ndarray::{ArrayD, Axis};
use ndarray_stats::QuantileExt;

#[derive(Clone, PartialEq, Debug, Default)]
pub struct History {
    pub history: Vec<Benchmark>,
}

impl History {
    pub fn new() -> Self {
        Self { history: vec![] }
    }

    pub fn get_loss_time_series(&self) -> Vec<f64> {
        self.history.iter().map(|h| h.loss).collect::<Vec<_>>()
    }

    pub fn get_metric_time_series(&self, metrics_type: MetricsType) -> Option<Vec<f64>> {
        self.history
            .iter()
            .map(|h| h.metrics.get_metric(metrics_type))
            .collect::<Option<Vec<_>>>()
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Benchmark {
    pub metrics: Metrics,
    pub loss: f64,
}

impl Benchmark {
    pub fn new(metrics: &Vec<MetricsType>) -> Self {
        Self {
            metrics: Metrics::from(metrics),
            loss: 0f64,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default, PartialOrd, Ord)]
pub enum MetricsType {
    #[default]
    Accuracy,
    Recall,
    Precision,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Metrics {
    pub metrics: HashMap<MetricsType, f64>,
}

impl Metrics {
    fn from(metrics: &Vec<MetricsType>) -> Self {
        let mut map = HashMap::new();
        for el in metrics {
            map.insert(*el, 0f64);
        }
        Self { metrics: map }
    }

    pub fn get_all(&self) -> &HashMap<MetricsType, f64> {
        &self.metrics
    }

    pub fn get_metric(&self, metric: MetricsType) -> Option<f64> {
        if let Some(metric) = self.metrics.get(&metric) {
            return Some(*metric);
        }
        None
    }

    /// Accumulate metrics for a given batch
    /// # Arguments
    /// * `predictions` a batched probability distribution of shape (n, i)
    /// * `true_labels` a batched observed values of shape (n, i)
    pub fn accumulate(&mut self, predictions: &ArrayD<f64>, observed: &ArrayD<f64>) {
        for (metric_type, value) in self.metrics.iter_mut() {
            match metric_type {
                MetricsType::Accuracy => {
                    let pred_classes = predictions.map_axis(Axis(1), |prob| prob.argmax().unwrap());

                    let true_classes =
                        observed.map_axis(Axis(1), |one_hot| one_hot.argmax().unwrap());

                    let correct_preds = pred_classes
                        .iter()
                        .zip(true_classes.iter())
                        .filter(|&(pred, true_label)| pred == true_label)
                        .count();

                    let accuracy = correct_preds as f64 / predictions.shape()[0] as f64;
                    *value += accuracy;
                }
                MetricsType::Recall => {
                    todo!()
                }
                MetricsType::Precision => {
                    let pred_classes = predictions.map_axis(Axis(1), |prob| prob.argmax().unwrap());
                    let true_classes = observed.map_axis(Axis(1), |one_hot| one_hot.argmax().unwrap());

                    // Get number of classes from the shape of predictions
                    let num_classes = predictions.shape()[1];
                    
                    let mut class_precisions = Vec::new();
                    
                    // Calculate precision for each class
                    for class_idx in 0..num_classes {
                        let mut true_positives = 0;
                        let mut false_positives = 0;
                        
                        for (pred, true_label) in pred_classes.iter().zip(true_classes.iter()) {
                            if *pred == class_idx {
                                if *true_label == class_idx {
                                    true_positives += 1;
                                } else {
                                    false_positives += 1;
                                }
                            }
                        }
                        
                        // Calculate precision for this class
                        let total_predicted = true_positives + false_positives;
                        if total_predicted > 0 {
                            let precision = true_positives as f64 / total_predicted as f64;
                            class_precisions.push(precision);
                        }
                    }
                    
                    // Compute macro-averaged precision
                    let precision = if !class_precisions.is_empty() {
                        class_precisions.iter().sum::<f64>() / class_precisions.len() as f64
                    } else {
                        0.0
                    };
                    
                    *value += precision;
                }
            }
        }
    }

    pub fn mean(&mut self, metric_type: MetricsType, number_of_batch: usize) {
        if let Some(m) = self.metrics.get_mut(&metric_type) {
            *m /= number_of_batch as f64;
        }
    }

    pub fn mean_all(&mut self, number_of_batch: usize) {
        for (&_, value) in self.metrics.iter_mut() {
            *value /= number_of_batch as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_precision_perfect_predictions() {
        // Test case: Perfect predictions (all correct)
        let mut metrics = Metrics::from(&vec![MetricsType::Precision]);
        
        // Create predictions and observed values with 3 classes
        // Predictions: class 0, 1, 2, 0, 1, 2 (one-hot encoded)
        let predictions = arr2(&[
            [1.0, 0.0, 0.0],  // predicted class 0
            [0.0, 1.0, 0.0],  // predicted class 1
            [0.0, 0.0, 1.0],  // predicted class 2
            [1.0, 0.0, 0.0],  // predicted class 0
            [0.0, 1.0, 0.0],  // predicted class 1
            [0.0, 0.0, 1.0],  // predicted class 2
        ]).into_dyn();
        
        // True labels match predictions perfectly
        let observed = arr2(&[
            [1.0, 0.0, 0.0],  // true class 0
            [0.0, 1.0, 0.0],  // true class 1
            [0.0, 0.0, 1.0],  // true class 2
            [1.0, 0.0, 0.0],  // true class 0
            [0.0, 1.0, 0.0],  // true class 1
            [0.0, 0.0, 1.0],  // true class 2
        ]).into_dyn();
        
        metrics.accumulate(&predictions, &observed);
        
        let precision = metrics.get_metric(MetricsType::Precision).unwrap();
        assert!((precision - 1.0).abs() < 1e-10, "Expected precision 1.0, got {}", precision);
    }

    #[test]
    fn test_precision_with_errors() {
        // Test case: Some misclassifications
        let mut metrics = Metrics::from(&vec![MetricsType::Precision]);
        
        // Create predictions and observed values with 3 classes
        let predictions = arr2(&[
            [1.0, 0.0, 0.0],  // predicted class 0
            [1.0, 0.0, 0.0],  // predicted class 0 (wrong, should be 1)
            [0.0, 0.0, 1.0],  // predicted class 2 (wrong, should be 1)
            [1.0, 0.0, 0.0],  // predicted class 0
        ]).into_dyn();
        
        let observed = arr2(&[
            [1.0, 0.0, 0.0],  // true class 0
            [0.0, 1.0, 0.0],  // true class 1
            [0.0, 1.0, 0.0],  // true class 1
            [1.0, 0.0, 0.0],  // true class 0
        ]).into_dyn();
        
        metrics.accumulate(&predictions, &observed);
        
        // Class 0: 2 TP, 1 FP -> precision = 2/3
        // Class 1: 0 TP, 0 FP -> not counted (no predictions)
        // Class 2: 0 TP, 1 FP -> precision = 0/1 = 0
        // Macro average: (2/3 + 0) / 2 = 1/3 ≈ 0.333...
        let precision = metrics.get_metric(MetricsType::Precision).unwrap();
        let expected = 1.0 / 3.0;
        assert!((precision - expected).abs() < 1e-10, "Expected precision {}, got {}", expected, precision);
    }

    #[test]
    fn test_precision_binary_classification() {
        // Test case: Binary classification
        let mut metrics = Metrics::from(&vec![MetricsType::Precision]);
        
        // 2 classes: positive (0) and negative (1)
        let predictions = arr2(&[
            [1.0, 0.0],  // predicted positive
            [1.0, 0.0],  // predicted positive
            [0.0, 1.0],  // predicted negative
            [1.0, 0.0],  // predicted positive
        ]).into_dyn();
        
        let observed = arr2(&[
            [1.0, 0.0],  // true positive
            [0.0, 1.0],  // true negative (FP for positive)
            [0.0, 1.0],  // true negative
            [1.0, 0.0],  // true positive
        ]).into_dyn();
        
        metrics.accumulate(&predictions, &observed);
        
        // Class 0 (positive): 2 TP, 1 FP -> precision = 2/3
        // Class 1 (negative): 1 TP, 0 FP -> precision = 1/1 = 1
        // Macro average: (2/3 + 1) / 2 = 5/6 ≈ 0.833...
        let precision = metrics.get_metric(MetricsType::Precision).unwrap();
        let expected = 5.0 / 6.0;
        assert!((precision - expected).abs() < 1e-10, "Expected precision {}, got {}", expected, precision);
    }
}
