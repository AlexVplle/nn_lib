use ndarray::{ArrayD, Axis};
use std::collections::HashMap;

use super::{ConfusionMatrix, MetricsType};

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Metrics {
    pub metrics: HashMap<MetricsType, f64>,
}

impl Metrics {
    pub(crate) fn from(metrics: &Vec<MetricsType>) -> Self {
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
                    let accuracy: f64 = if n_samples > 0 {
                        total_correct as f64 / n_samples as f64
                    } else {
                        0.0
                    };
                    *value += accuracy;
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

                MetricsType::MacroRecall => {
                    let recall: f64 = (0..confusion_matrix.number_of_class)
                        .map(|class_i: usize| {
                            let denominator: f64 = (confusion_matrix.true_positives[class_i]
                                + confusion_matrix.false_negatives[class_i])
                                as f64;
                            if denominator > 0.0 {
                                confusion_matrix.true_positives[class_i] as f64 / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>()
                        / confusion_matrix.number_of_class as f64;
                    *value += recall;
                }

                MetricsType::MacroPrecision => {
                    let precision: f64 = (0..confusion_matrix.number_of_class)
                        .map(|class_i: usize| {
                            let denominator: f64 = (confusion_matrix.true_positives[class_i]
                                + confusion_matrix.false_positives[class_i])
                                as f64;
                            if denominator > 0.0 {
                                confusion_matrix.true_positives[class_i] as f64 / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>()
                        / confusion_matrix.number_of_class as f64;
                    *value += precision;
                }

                MetricsType::MacroF1Score => {
                    let f1_score: f64 = (0..confusion_matrix.number_of_class)
                        .map(|class_i: usize| {
                            let denominator: f64 = (2 * confusion_matrix.true_positives[class_i]
                                + confusion_matrix.false_positives[class_i]
                                + confusion_matrix.false_negatives[class_i])
                                as f64;
                            if denominator > 0.0 {
                                2.0 * confusion_matrix.true_positives[class_i] as f64 / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>()
                        / confusion_matrix.number_of_class as f64;
                    *value += f1_score;
                }

                MetricsType::WeightedRecall => {
                    let recall: f64 = (0..confusion_matrix.number_of_class)
                        .map(|class_i: usize| {
                            let denominator: f64 = (confusion_matrix.true_positives[class_i]
                                + confusion_matrix.false_negatives[class_i])
                                as f64;
                            if denominator > 0.0 {
                                confusion_matrix.weight_classes[class_i]
                                    * confusion_matrix.true_positives[class_i] as f64
                                    / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    *value += recall;
                }

                MetricsType::WeightedPrecision => {
                    let precision: f64 = (0..confusion_matrix.number_of_class)
                        .map(|class_i: usize| {
                            let denominator: f64 = (confusion_matrix.true_positives[class_i]
                                + confusion_matrix.false_positives[class_i])
                                as f64;
                            if denominator > 0.0 {
                                confusion_matrix.weight_classes[class_i]
                                    * confusion_matrix.true_positives[class_i] as f64
                                    / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    *value += precision;
                }

                MetricsType::WeightedF1Score => {
                    let f1_score: f64 = (0..confusion_matrix.number_of_class)
                        .map(|class_i: usize| {
                            let denominator: f64 = (2 * confusion_matrix.true_positives[class_i]
                                + confusion_matrix.false_positives[class_i]
                                + confusion_matrix.false_negatives[class_i])
                                as f64;
                            if denominator > 0.0 {
                                confusion_matrix.weight_classes[class_i]
                                    * 2.0
                                    * confusion_matrix.true_positives[class_i] as f64
                                    / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    *value += f1_score;
                }
            }
        }
    }

    pub fn mean(&mut self, metric_type: MetricsType, number_of_batch: usize) {
        if number_of_batch > 0 {
            if let Some(m) = self.metrics.get_mut(&metric_type) {
                *m /= number_of_batch as f64;
            }
        }
    }

    pub fn mean_all(&mut self, number_of_batch: usize) {
        if number_of_batch > 0 {
            for (&_, value) in self.metrics.iter_mut() {
                *value /= number_of_batch as f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{Benchmark, History};
    use ndarray::arr2;

    #[test]
    fn test_metrics_from() {
        let metrics_types: Vec<MetricsType> = vec![MetricsType::Accuracy, MetricsType::MacroRecall];
        let metrics: Metrics = Metrics::from(&metrics_types);

        assert_eq!(metrics.metrics.len(), 2);
        assert!(metrics.metrics.contains_key(&MetricsType::Accuracy));
        assert!(metrics.metrics.contains_key(&MetricsType::MacroRecall));
        assert_eq!(metrics.metrics.get(&MetricsType::Accuracy), Some(&0.0));
        assert_eq!(metrics.metrics.get(&MetricsType::MacroRecall), Some(&0.0));
    }

    #[test]
    fn test_metrics_get_metric() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);
        metrics.metrics.insert(MetricsType::Accuracy, 0.85);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(0.85));
        assert_eq!(metrics.get_metric(MetricsType::MacroRecall), None);
    }

    #[test]
    fn test_metrics_get_all() {
        let metrics_types: Vec<MetricsType> =
            vec![MetricsType::Accuracy, MetricsType::MacroPrecision];
        let mut metrics: Metrics = Metrics::from(&metrics_types);
        metrics.metrics.insert(MetricsType::Accuracy, 0.90);
        metrics.metrics.insert(MetricsType::MacroPrecision, 0.88);

        let all_metrics: &HashMap<MetricsType, f64> = metrics.get_all();
        assert_eq!(all_metrics.len(), 2);
        assert_eq!(all_metrics.get(&MetricsType::Accuracy), Some(&0.90));
        assert_eq!(all_metrics.get(&MetricsType::MacroPrecision), Some(&0.88));
    }

    #[test]
    fn test_metrics_accumulate_accuracy_perfect() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);

        let predictions: ArrayD<f64> = arr2(&[
            [0.1, 0.2, 0.6, 0.1],
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.1, 0.1, 0.7],
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(1.0));
    }

    #[test]
    fn test_metrics_accumulate_accuracy_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);

        let predictions: ArrayD<f64> = arr2(&[
            [0.1, 0.2, 0.6, 0.1],
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.1, 0.1, 0.7],
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .into_dyn();

        metrics.accumulate(&predictions, &observed);

        let accuracy: f64 = metrics.get_metric(MetricsType::Accuracy).unwrap();
        assert!((accuracy - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_accuracy_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);

        let predictions: ArrayD<f64> =
            arr2(&[[0.1, 0.2, 0.6, 0.1], [0.8, 0.1, 0.05, 0.05]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(0.0));
    }

    #[test]
    fn test_metrics_accumulate_multiple_batches() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);

        let predictions1: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed1: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions1, &observed1);
        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(1.0));

        let predictions2: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed2: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions2, &observed2);

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

        metrics.mean(MetricsType::MacroRecall, 4);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(3.0));
    }

    #[test]
    fn test_metrics_mean_all() {
        let mut metrics: Metrics =
            Metrics::from(&vec![MetricsType::Accuracy, MetricsType::MacroPrecision]);
        metrics.metrics.insert(MetricsType::Accuracy, 6.0);
        metrics.metrics.insert(MetricsType::MacroPrecision, 8.0);

        metrics.mean_all(2);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(3.0));
        assert_eq!(metrics.get_metric(MetricsType::MacroPrecision), Some(4.0));
    }

    #[test]
    fn test_weighted_metrics_imbalanced_with_errors() {
        let mut metrics: Metrics =
            Metrics::from(&vec![MetricsType::MacroRecall, MetricsType::WeightedRecall]);

        let predictions: ArrayD<f64> =
            arr2(&[[0.1, 0.9], [0.1, 0.9], [0.2, 0.8], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> =
            arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let macro_recall: f64 = metrics.get_metric(MetricsType::MacroRecall).unwrap();
        let weighted_recall: f64 = metrics.get_metric(MetricsType::WeightedRecall).unwrap();

        assert!((macro_recall - 0.5).abs() < 1e-10);

        assert!((weighted_recall - 0.75).abs() < 1e-10);

        assert!(weighted_recall > macro_recall);
    }

    #[test]
    fn test_weighted_vs_macro_metrics() {
        let mut metrics: Metrics = Metrics::from(&vec![
            MetricsType::MacroRecall,
            MetricsType::WeightedRecall,
            MetricsType::MacroPrecision,
            MetricsType::WeightedPrecision,
        ]);

        let predictions: ArrayD<f64> =
            arr2(&[[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> =
            arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let macro_recall: f64 = metrics.get_metric(MetricsType::MacroRecall).unwrap();
        let weighted_recall: f64 = metrics.get_metric(MetricsType::WeightedRecall).unwrap();
        let macro_precision: f64 = metrics.get_metric(MetricsType::MacroPrecision).unwrap();
        let weighted_precision: f64 = metrics.get_metric(MetricsType::WeightedPrecision).unwrap();

        assert!((macro_recall - 1.0).abs() < 1e-10);
        assert!((weighted_recall - 1.0).abs() < 1e-10);
        assert!((macro_precision - 1.0).abs() < 1e-10);
        assert!((weighted_precision - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_mean_all_zero_batches() {
        let mut metrics: Metrics = Metrics::from(&vec![
            MetricsType::Accuracy,
            MetricsType::MacroPrecision,
            MetricsType::MacroRecall,
            MetricsType::MacroF1Score,
        ]);

        metrics.metrics.insert(MetricsType::Accuracy, 2.5);
        metrics.metrics.insert(MetricsType::MacroPrecision, 1.8);
        metrics.metrics.insert(MetricsType::MacroRecall, 1.5);
        metrics.metrics.insert(MetricsType::MacroF1Score, 1.6);

        metrics.mean_all(0);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(2.5));
        assert_eq!(metrics.get_metric(MetricsType::MacroPrecision), Some(1.8));
        assert_eq!(metrics.get_metric(MetricsType::MacroRecall), Some(1.5));
        assert_eq!(metrics.get_metric(MetricsType::MacroF1Score), Some(1.6));
    }

    #[test]
    fn test_metrics_mean_single_metric_zero_batches() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::Accuracy]);
        metrics.metrics.insert(MetricsType::Accuracy, 3.0);

        metrics.mean(MetricsType::Accuracy, 0);

        assert_eq!(metrics.get_metric(MetricsType::Accuracy), Some(3.0));
    }

    #[test]
    fn test_metrics_accumulate_recall_perfect() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::MacroRecall]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::MacroRecall), Some(1.0));
    }

    #[test]
    fn test_metrics_accumulate_recall_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::MacroRecall]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let recall: f64 = metrics.get_metric(MetricsType::MacroRecall).unwrap();
        assert!((recall - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_recall_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::MacroRecall]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::MacroRecall), Some(0.0));
    }

    #[test]
    fn test_metrics_accumulate_precision_perfect() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::MacroPrecision]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::MacroPrecision), Some(1.0));
    }

    #[test]
    fn test_metrics_accumulate_precision_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::MacroPrecision]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        let precision: f64 = metrics.get_metric(MetricsType::MacroPrecision).unwrap();
        assert!((precision - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_precision_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::MacroPrecision]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::MacroPrecision), Some(0.0));
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
        let mut metrics: Metrics =
            Metrics::from(&vec![MetricsType::TypeIError, MetricsType::Specificity]);

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
    fn test_metrics_accumulate_f1_score_perfect() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::MacroF1Score]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::MacroF1Score), Some(1.0));
    }

    #[test]
    fn test_metrics_accumulate_f1_score_partial() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::MacroF1Score]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        // Macro-averaging:
        // Class 0: TP=1, FP=2, FN=0 -> F1=2*1/(2*1+2+0)=2/4=0.5
        // Class 1: TP=0, FP=0, FN=2 -> F1=0/2=0.0
        // Macro-F1 = (0.5 + 0.0) / 2 = 0.25
        let f1_score: f64 = metrics.get_metric(MetricsType::MacroF1Score).unwrap();
        assert!((f1_score - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_accumulate_f1_score_zero() {
        let mut metrics: Metrics = Metrics::from(&vec![MetricsType::MacroF1Score]);

        let predictions: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.9, 0.1]]).into_dyn();

        let observed: ArrayD<f64> = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();

        metrics.accumulate(&predictions, &observed);

        assert_eq!(metrics.get_metric(MetricsType::MacroF1Score), Some(0.0));
    }

    #[test]
    fn test_full_workflow() {
        let mut history: History = History::new();

        let mut benchmark1: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark1.loss = 0.8;

        let preds1: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();
        let obs1: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
        benchmark1.metrics.accumulate(&preds1, &obs1);

        let preds2: ArrayD<f64> = arr2(&[[0.8, 0.2], [0.6, 0.4]]).into_dyn();
        let obs2: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
        benchmark1.metrics.accumulate(&preds2, &obs2);

        benchmark1.metrics.mean(MetricsType::Accuracy, 2);
        history.history.push(benchmark1);

        let mut benchmark2: Benchmark = Benchmark::new(&vec![MetricsType::Accuracy]);
        benchmark2.loss = 0.4;

        let preds3: ArrayD<f64> = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();
        let obs3: ArrayD<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
        benchmark2.metrics.accumulate(&preds3, &obs3);

        benchmark2.metrics.mean(MetricsType::Accuracy, 1);
        history.history.push(benchmark2);

        let loss_series: Vec<f64> = history.get_loss_time_series();
        assert_eq!(loss_series, vec![0.8, 0.4]);

        let accuracy_series: Option<Vec<f64>> =
            history.get_metric_time_series(MetricsType::Accuracy);
        assert_eq!(accuracy_series, Some(vec![0.75, 1.0]));
    }
}
