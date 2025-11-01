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
        self.history
            .iter()
            .map(|h: &Benchmark| h.loss)
            .collect::<Vec<_>>()
    }

    pub fn get_metric_time_series(&self, metrics_type: MetricsType) -> Option<Vec<f64>> {
        self.history
            .iter()
            .map(|h: &Benchmark| h.metrics.get_metric(metrics_type))
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
    Specificity,
    TypeIError,
    TypeIIError,
    F1Score,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Metrics {
    pub metrics: HashMap<MetricsType, f64>,
}

impl Metrics {
    fn from(metrics: &Vec<MetricsType>) -> Self {
        let mut map: HashMap<MetricsType, f64> = HashMap::new();
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
    /// * `observed` a batched observed values of shape (n, i)
    pub fn accumulate(&mut self, predictions: &ArrayD<f64>, observed: &ArrayD<f64>) {
        let confusion_matrix: ConfusionMatrix = ConfusionMatrix::from(predictions, observed);
        for (metric_type, value) in self.metrics.iter_mut() {
            match metric_type {
                MetricsType::Accuracy => {
                    let total_correct: usize = confusion_matrix.true_positives.iter().sum();
                    let n_samples: usize = predictions.len_of(Axis(0));
                    let accuracy: f64 = total_correct as f64 / n_samples as f64;
                    *value += accuracy;
                }

                MetricsType::Recall => {
                    let true_positive: usize = confusion_matrix.true_positives.iter().sum();
                    let false_negative: usize = confusion_matrix.false_negatives.iter().sum();
                    let denominator: f64 = (true_positive + false_negative) as f64;
                    let recall: f64 = if denominator > 0.0 {
                        true_positive as f64 / denominator
                    } else {
                        0.0
                    };
                    *value += recall;
                }

                MetricsType::Precision => {
                    let true_positive: usize = confusion_matrix.true_positives.iter().sum();
                    let false_positive: usize = confusion_matrix.false_positives.iter().sum();
                    let denominator: f64 = (true_positive + false_positive) as f64;
                    let precision: f64 = if denominator > 0.0 {
                        true_positive as f64 / denominator
                    } else {
                        0.0
                    };
                    *value += precision;
                }

                MetricsType::Specificity => {
                    let true_negative: usize = confusion_matrix.true_negatives.iter().sum();
                    let false_positive: usize = confusion_matrix.false_positives.iter().sum();
                    let denominator: f64 = (true_negative + false_positive) as f64;
                    let specificity: f64 = if denominator > 0.0 {
                        true_negative as f64 / denominator
                    } else {
                        0.0
                    };
                    *value += specificity;
                }

                MetricsType::TypeIError => {
                    let false_positive: usize = confusion_matrix.false_positives.iter().sum();
                    let true_negative: usize = confusion_matrix.true_negatives.iter().sum();
                    let denominator: f64 = (false_positive + true_negative) as f64;
                    let type_1_error: f64 = if denominator > 0.0 {
                        false_positive as f64 / denominator
                    } else {
                        0.0
                    };
                    *value += type_1_error;
                }

                MetricsType::TypeIIError => {
                    let false_negative: usize = confusion_matrix.false_negatives.iter().sum();
                    let true_positive: usize = confusion_matrix.true_positives.iter().sum();
                    let denominator: f64 = (false_negative + true_positive) as f64;
                    let type_2_error: f64 = if denominator > 0.0 {
                        false_negative as f64 / denominator
                    } else {
                        0.0
                    };
                    *value += type_2_error;
                }

                MetricsType::F1Score => {
                    let true_positive: usize = confusion_matrix.true_positives.iter().sum();
                    let false_positive: usize = confusion_matrix.false_positives.iter().sum();
                    let false_negative: usize = confusion_matrix.false_negatives.iter().sum();
                    let denominator: f64 =
                        (2 * true_positive + false_positive + false_negative) as f64;
                    let f1_score: f64 = if denominator > 0.0 {
                        2.0 * true_positive as f64 / denominator
                    } else {
                        0.0
                    };
                    *value += f1_score;
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

#[derive(Clone, Eq, PartialEq, Hash, Debug, Default, PartialOrd, Ord)]
pub struct ConfusionMatrix {
    pub matrix: Vec<Vec<usize>>,

    pub true_positives: Vec<usize>,
    pub true_negatives: Vec<usize>,
    pub false_positives: Vec<usize>,
    pub false_negatives: Vec<usize>,
}

impl ConfusionMatrix {
    /// # Arguments
    /// * `predictions` a batched probability distribution of shape (n, i)
    /// * `observed` a batched observed values of shape (n, i)
    fn from(predictions: &ArrayD<f64>, observed: &ArrayD<f64>) -> Self {
        let number_of_class: usize = predictions.len_of(Axis(1));
        let mut matrix: Vec<Vec<usize>> = vec![vec![0; number_of_class]; number_of_class];

        let predicted_classes: ArrayD<usize> =
            predictions.map_axis(Axis(1), |probabilities| probabilities.argmax().unwrap());
        let true_classes: ArrayD<usize> =
            observed.map_axis(Axis(1), |one_hot| one_hot.argmax().unwrap());

        for (&prediction, &true_label) in predicted_classes.iter().zip(true_classes.iter()) {
            matrix[true_label][prediction] += 1;
        }

        let true_positives: Vec<usize> =
            (0..number_of_class).map(|i: usize| matrix[i][i]).collect();

        let false_positives: Vec<usize> = (0..number_of_class)
            .map(|predicted_class: usize| {
                (0..number_of_class)
                    .filter(|&true_class| true_class != predicted_class)
                    .map(|true_class: usize| matrix[true_class][predicted_class])
                    .sum()
            })
            .collect();

        let false_negatives: Vec<usize> = (0..number_of_class)
            .map(|true_class: usize| {
                (0..number_of_class)
                    .filter(|&pred_class| pred_class != true_class)
                    .map(|pred_class: usize| matrix[true_class][pred_class])
                    .sum()
            })
            .collect();

        let n_samples: usize = predictions.len_of(Axis(0));
        let true_negatives: Vec<usize> = (0..number_of_class)
            .map(|class: usize| {
                n_samples - true_positives[class] - false_positives[class] - false_negatives[class]
            })
            .collect();

        Self {
            matrix,
            true_positives,
            true_negatives,
            false_positives,
            false_negatives,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_history_new() {
        let history: History = History::new();
        assert_eq!(history.history.len(), 0);
    }

    #[test]
    fn test_history_get_loss_time_series() {
        let mut history: History = History::new();

        let mut benchmark1: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark1.loss = 0.5;

        let mut benchmark2: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark2.loss = 0.3;

        let mut benchmark3: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark3.loss = 0.1;

        history.history.push(benchmark1);
        history.history.push(benchmark2);
        history.history.push(benchmark3);

        let loss_series = history.get_loss_time_series();
        assert_eq!(loss_series, vec![0.5, 0.3, 0.1]);
    }

    #[test]
    fn test_history_get_metric_time_series() {
        let mut history: History = History::new();

        let mut benchmark1: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark1
            .metrics
            .metrics
            .insert(MetricsType::Accuracy, 0.85);

        let mut benchmark2: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark2
            .metrics
            .metrics
            .insert(MetricsType::Accuracy, 0.90);

        let mut benchmark3: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark3
            .metrics
            .metrics
            .insert(MetricsType::Accuracy, 0.95);

        history.history.push(benchmark1);
        history.history.push(benchmark2);
        history.history.push(benchmark3);

        let metric_series: Option<Vec<f64>> = history.get_metric_time_series(MetricsType::Accuracy);
        assert_eq!(metric_series, Some(vec![0.85, 0.90, 0.95]));
    }

    #[test]
    fn test_history_get_metric_time_series_missing_metric() {
        let mut history: History = History::new();

        let benchmark: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        history.history.push(benchmark);

        let metric_series: Option<Vec<f64>> = history.get_metric_time_series(MetricsType::Recall);
        assert_eq!(metric_series, None);
    }

    #[test]
    fn test_benchmark_new() {
        let metrics_types: Vec<MetricsType> = vec![MetricsType::Accuracy, MetricsType::Precision];
        let benchmark: Benchmark = Benchmark::new(&metrics_types);

        assert_eq!(benchmark.loss, 0.0);
        assert_eq!(benchmark.metrics.metrics.len(), 2);
        assert_eq!(
            benchmark.metrics.get_metric(MetricsType::Accuracy),
            Some(0.0)
        );
        assert_eq!(
            benchmark.metrics.get_metric(MetricsType::Precision),
            Some(0.0)
        );
    }

    #[test]
    fn test_metrics_from() {
        let metrics_types: Vec<MetricsType> = vec![MetricsType::Accuracy, MetricsType::Recall];
        let metrics: Metrics = Metrics::from(&metrics_types);

        assert_eq!(metrics.metrics.len(), 2);
        assert!(metrics.metrics.contains_key(&MetricsType::Accuracy));
        assert!(metrics.metrics.contains_key(&MetricsType::Recall));
        assert_eq!(metrics.metrics.get(&MetricsType::Accuracy), Some(&0.0));
        assert_eq!(metrics.metrics.get(&MetricsType::Recall), Some(&0.0));
    }

    #[test]
    fn test_metrics_get_metric() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);
        metrics.metrics.insert(MetricsType::Accuracy, 0.85);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(0.85));
        assert_eq!(metrics.get_metric(MetricsType::Recall), None);
    }

    #[test]
    fn test_metrics_get_all() {
        let metrics_types: Vec<MetricsType> = vec![MetricsType::Accuracy, MetricsType::Precision];
        let mut metrics: Metrics = Metrics::from(&metrics_types);
        metrics.metrics.insert(MetricsType::Accuracy, 0.90);
        metrics.metrics.insert(MetricsType::Precision, 0.88);

        let all_metrics: &HashMap<MetricsType, f64> = metrics.get_all();
        assert_eq!(all_metrics.len(), 2);
        assert_eq!(all_metrics.get(&MetricsType::Accuracy), Some(&0.90));
        assert_eq!(all_metrics.get(&MetricsType::Precision), Some(&0.88));
    }

    #[test]
    fn test_metrics_accumulate_accuracy_perfect() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);

        // Perfect predictions: all match
        // Shape: (batch_size=3, num_classes=4)
        let predictions: ArrayD<f64> = arr2(&[
            [0.1, 0.2, 0.6, 0.1],   // argmax = 2
            [0.8, 0.1, 0.05, 0.05], // argmax = 0
            [0.1, 0.1, 0.1, 0.7],   // argmax = 3
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [0.0, 0.0, 1.0, 0.0], // true class = 2
            [1.0, 0.0, 0.0, 0.0], // true class = 0
            [0.0, 0.0, 0.0, 1.0], // true class = 3
        ])
        .into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(1.0));
    }

    #[test]
    fn test_metrics_accumulate_accuracy_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);

        // Partial predictions: 2 out of 3 correct
        let predictions: ArrayD<f64> = arr2(&[
            [0.1, 0.2, 0.6, 0.1],   // argmax = 2 (correct)
            [0.8, 0.1, 0.05, 0.05], // argmax = 0 (wrong, should be 1)
            [0.1, 0.1, 0.1, 0.7],   // argmax = 3 (correct)
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [0.0, 0.0, 1.0, 0.0], // true class = 2
            [0.0, 1.0, 0.0, 0.0], // true class = 1
            [0.0, 0.0, 0.0, 1.0], // true class = 3
        ])
        .into_dyn();

        metrics.accumulate(&predictions, &observed);

        // 2 out of 3 correct, accuracy should be 2/3
        let accuracy: f64 = metrics.get_metric(MetricsType::Accuracy).unwrap();
        assert!((accuracy - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_accuracy_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);

        // All predictions wrong
        let predictions: ArrayD<f64> = arr2(&[
            [0.1, 0.2, 0.6, 0.1],   // argmax = 2 (wrong)
            [0.8, 0.1, 0.05, 0.05], // argmax = 0 (wrong)
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [1.0, 0.0, 0.0, 0.0], // true class = 0
            [0.0, 1.0, 0.0, 0.0], // true class = 1
        ])
        .into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(0.0));
    }

    #[test]
    fn test_metrics_accumulate_multiple_batches() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);

        // First batch: 100% accuracy
        let predictions1: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed1: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions1, &observed1);
        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(1.0));

        // Second batch: 50% accuracy
        let predictions2: ArrayD<f64> = arr2(&[
            [0.9, 0.1], // correct
            [0.9, 0.1], // wrong
        ])
        .into_dyn();

        let observed2: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions2, &observed2);

        // Accumulated: 1.0 + 0.5 = 1.5
        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(1.5));
    }

    #[test]
    fn test_metrics_mean_single_metric() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);
        metrics.metrics.insert(MetricsType::Accuracy, 3.0);

        metrics.mean(MetricsType::Accuracy, 4);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(0.75));
    }

    #[test]
    fn test_metrics_mean_nonexistent_metric() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);
        metrics.metrics.insert(MetricsType::Accuracy, 3.0);

        // Calling mean on a metric that doesn't exist shouldn't panic
        metrics.mean(MetricsType::Recall, 4);

        // Accuracy should remain unchanged
        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(3.0));
    }

    #[test]
    fn test_metrics_mean_all() {
        let mut metrics: Metrics =
            Metrics::from(&vec![MetricsType::Accuracy, MetricsType::Precision]);
        metrics.metrics.insert(MetricsType::Accuracy, 6.0);
        metrics.metrics.insert(MetricsType::Precision, 8.0);

        metrics.mean_all(2);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(3.0));
        assert_eq!(metrics.get_metric(MetricsType::Precision), Some(4.0));
    }

    #[test]
    fn test_full_workflow() {
        let mut history: History = History::new();

        // Epoch 1
        let mut benchmark1: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark1.loss = 0.8;

        // Simulate 2 batches
        let preds1: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();
        let obs1: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
        benchmark1.metrics.accumulate(&preds1, &obs1); // accuracy = 1.0

        let preds2: ArrayD<f64> = arr2(&[[0.8, 0.2], [0.6, 0.4]]).into_dyn();
        let obs2: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
        benchmark1.metrics.accumulate(&preds2, &obs2); // accuracy = 0.5

        benchmark1.metrics.mean(MetricsType::Accuracy, 2); // (1.0 + 0.5) / 2 = 0.75
        history.history.push(benchmark1);

        // Epoch 2
        let mut benchmark2: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark2.loss = 0.4;

        let preds3: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();
        let obs3: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
        benchmark2.metrics.accumulate(&preds3, &obs3); // accuracy = 1.0

        benchmark2.metrics.mean(MetricsType::Accuracy, 1);
        history.history.push(benchmark2);

        // Verify history
        let loss_series: Vec<f64> = history.get_loss_time_series();
        assert_eq!(loss_series, vec![0.8, 0.4]);

        let accuracy_series: Option<Vec<f64>> =
            history.get_metric_time_series(MetricsType::Accuracy);
        assert_eq!(accuracy_series, Some(vec![0.75, 1.0]));
    }

    #[test]
    fn test_confusion_matrix_perfect_predictions() {
        let predictions: ArrayD<f64> = arr2(&[
            [0.9, 0.05, 0.05], // Predicted: class 0
            [0.1, 0.8, 0.1],   // Predicted: class 1
            [0.05, 0.1, 0.85], // Predicted: class 2
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [1.0, 0.0, 0.0], // True: class 0
            [0.0, 1.0, 0.0], // True: class 1
            [0.0, 0.0, 1.0], // True: class 2
        ])
        .into_dyn();

        let confusion_matrix: ConfusionMatrix = ConfusionMatrix::from(&predictions, &observed);

        assert_eq!(confusion_matrix.true_positives, vec![1, 1, 1]);
        assert_eq!(confusion_matrix.true_negatives, vec![2, 2, 2]);
        assert_eq!(confusion_matrix.false_positives, vec![0, 0, 0]);
        assert_eq!(confusion_matrix.false_negatives, vec![0, 0, 0]);
    }

    #[test]
    fn test_confusion_matrix_all_wrong() {
        let predictions: ArrayD<f64> = arr2(&[
            [0.9, 0.05, 0.05], // Predicted: class 0,
            [0.1, 0.8, 0.1],   // Predicted: class 1,
            [0.05, 0.1, 0.85], // Predicted: class 2,
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [0.0, 1.0, 0.0], // True: class 1
            [0.0, 0.0, 1.0], // True: class 2
            [1.0, 0.0, 0.0], // True: class 0
        ])
        .into_dyn();

        let confusion_matrix: ConfusionMatrix = ConfusionMatrix::from(&predictions, &observed);

        assert_eq!(confusion_matrix.true_positives, vec![0, 0, 0]);

        assert_eq!(confusion_matrix.false_positives, vec![1, 1, 1]);
        assert_eq!(confusion_matrix.false_negatives, vec![1, 1, 1]);

        assert_eq!(confusion_matrix.true_negatives, vec![1, 1, 1]);
    }

    #[test]
    fn test_confusion_matrix_mixed_predictions() {
        let predictions: ArrayD<f64> = arr2(&[
            [0.9, 0.05, 0.05], // Predicted: class 0, True: class 0 ✓
            [0.8, 0.1, 0.1],   // Predicted: class 0, True: class 1 ✗
            [0.1, 0.8, 0.1],   // Predicted: class 1, True: class 1 ✓
            [0.1, 0.1, 0.8],   // Predicted: class 2, True: class 2 ✓
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [1.0, 0.0, 0.0], // True: class 0
            [0.0, 1.0, 0.0], // True: class 1
            [0.0, 1.0, 0.0], // True: class 1
            [0.0, 0.0, 1.0], // True: class 2
        ])
        .into_dyn();

        let confusion_matrix: ConfusionMatrix = ConfusionMatrix::from(&predictions, &observed);

        assert_eq!(confusion_matrix.true_positives, vec![1, 1, 1]);
        assert_eq!(confusion_matrix.false_positives, vec![1, 0, 0]);
        assert_eq!(confusion_matrix.false_negatives, vec![0, 1, 0]);
        assert_eq!(confusion_matrix.true_negatives, vec![2, 2, 3]);
    }

    #[test]
    fn test_confusion_matrix_binary_classification() {
        let predictions: ArrayD<f64> =
            arr2(&[[0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]]).into_dyn();

        let observed: ArrayD<f64> =
            arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]).into_dyn();

        let confusion_matrix: ConfusionMatrix = ConfusionMatrix::from(&predictions, &observed);

        assert_eq!(confusion_matrix.true_positives, vec![1, 1]);
        assert_eq!(confusion_matrix.false_positives, vec![1, 1]);
        assert_eq!(confusion_matrix.false_negatives, vec![1, 1]);
        assert_eq!(confusion_matrix.true_negatives, vec![1, 1]);
    }

    #[test]
    fn test_confusion_matrix_consistency() {
        let predictions: ArrayD<f64> = arr2(&[
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
            [0.8, 0.1, 0.1],
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        .into_dyn();

        let confusion_matrix: ConfusionMatrix = ConfusionMatrix::from(&predictions, &observed);

        let n_samples: usize = 4;

        for i in 0..3 {
            let total = confusion_matrix.true_positives[i]
                + confusion_matrix.true_negatives[i]
                + confusion_matrix.false_positives[i]
                + confusion_matrix.false_negatives[i];
            assert_eq!(
                total, n_samples,
                "Class {}: TP + TN + FP + FN should equal n_samples",
                i
            );
        }
    }

    #[test]
    fn test_metrics_accumulate_recall_perfect() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Recall]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::Recall), Some(1.0));
    }

    #[test]
    fn test_metrics_accumulate_recall_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Recall]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let recall: f64 = metrics.get_metric(MetricsType::Recall).unwrap();
        assert!((recall - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_recall_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Recall]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::Recall), Some(0.0));
    }

    #[test]
    fn test_metrics_accumulate_precision_perfect() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Precision]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::Precision), Some(1.0));
    }

    #[test]
    fn test_metrics_accumulate_precision_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Precision]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let precision: f64 = metrics.get_metric(MetricsType::Precision).unwrap();
        assert!((precision - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_precision_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Precision]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::Precision), Some(0.0));
    }

    #[test]
    fn test_metrics_accumulate_specificity_perfect() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Specificity]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::Specificity), Some(1.0));
    }

    #[test]
    fn test_metrics_accumulate_specificity_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Specificity]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let specificity: f64 = metrics.get_metric(MetricsType::Specificity).unwrap();
        assert!((specificity - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_type_i_error_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::TypeIError]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::TypeIError), Some(0.0));
    }

    #[test]
    fn test_metrics_accumulate_type_i_error_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::TypeIError]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let type_i_error: f64 = metrics.get_metric(MetricsType::TypeIError).unwrap();
        assert!((type_i_error - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_type_i_error_complement_specificity() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::TypeIError, MetricsType::Specificity]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let type_i_error: f64 = metrics.get_metric(MetricsType::TypeIError).unwrap();
        let specificity: f64 = metrics.get_metric(MetricsType::Specificity).unwrap();

        assert!((type_i_error + specificity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_type_ii_error_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::TypeIIError]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::TypeIIError), Some(0.0));
    }

    #[test]
    fn test_metrics_accumulate_type_ii_error_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::TypeIIError]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let type_ii_error: f64 = metrics.get_metric(MetricsType::TypeIIError).unwrap();
        assert!((type_ii_error - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_type_ii_error_complement_recall() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::TypeIIError, MetricsType::Recall]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let type_ii_error: f64 = metrics.get_metric(MetricsType::TypeIIError).unwrap();
        let recall: f64 = metrics.get_metric(MetricsType::Recall).unwrap();

        assert!((type_ii_error + recall - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_f1_score_perfect() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::F1Score]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::F1Score), Some(1.0));
    }

    #[test]
    fn test_metrics_accumulate_f1_score_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::F1Score]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let f1_score: f64 = metrics.get_metric(MetricsType::F1Score).unwrap();
        assert!((f1_score - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_f1_score_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::F1Score]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::F1Score), Some(0.0));
    }

    #[test]
    fn test_metrics_accumulate_f1_score_harmonic_mean() {
        let mut metrics: Metrics = Metrics::from(&vec![
            MetricsType::F1Score,
            MetricsType::Precision,
            MetricsType::Recall,
        ]);

        let predictions: ArrayD<f64> = arr2(&[
            [0.9, 0.1, 0.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.1, 0.1, 0.8],
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        .into_dyn();

        metrics.accumulate(&predictions, &observed);

        let precision: f64 = metrics.get_metric(MetricsType::Precision).unwrap();
        let recall: f64 = metrics.get_metric(MetricsType::Recall).unwrap();
        let f1_score: f64 = metrics.get_metric(MetricsType::F1Score).unwrap();

        let expected_f1 = 2.0 * (precision * recall) / (precision + recall);
        assert!((f1_score - expected_f1).abs() < 1e-10);
    }
}
