use ndarray::{ArrayD, Axis};
use std::collections::HashMap;

use super::ConfusionMatrix;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MulticlassMetricType {
    Accuracy,
    Specificity,
    TypeIError,
    TypeIIError,
    MacroRecall,
    MacroPrecision,
    MacroF1Score,
    WeightedRecall,
    WeightedPrecision,
    WeightedF1Score,
}

#[derive(Clone, PartialEq, Debug)]
pub struct MulticlassClassifierMetrics {
    pub metrics: HashMap<MulticlassMetricType, f64>,
    total_samples: usize,
    accumulated_confusion_matrix: ConfusionMatrix,
}

impl MulticlassClassifierMetrics {
    pub fn from(metric_types: &Vec<MulticlassMetricType>) -> Self {
        let mut map: HashMap<MulticlassMetricType, f64> = HashMap::new();
        for metric_type in metric_types {
            map.insert(*metric_type, 0f64);
        }
        Self {
            metrics: map,
            total_samples: 0,
            accumulated_confusion_matrix: ConfusionMatrix::default(),
        }
    }

    pub fn get_metric(&self, metric_type: MulticlassMetricType) -> Option<f64> {
        self.metrics.get(&metric_type).copied()
    }

    pub fn accumulate(&mut self, predictions: &ArrayD<f64>, observed: &ArrayD<f64>) {
        let batch_size: usize = predictions.len_of(Axis(0));
        self.total_samples += batch_size;

        let confusion_matrix: ConfusionMatrix = ConfusionMatrix::from(predictions, observed);
        self.accumulated_confusion_matrix.add(&confusion_matrix);
    }

    pub fn finalize(&mut self) {
        let confusion_matrix: &ConfusionMatrix = &self.accumulated_confusion_matrix;
        let number_of_classes: usize = confusion_matrix.number_of_class;

        for (metric_type, value) in self.metrics.iter_mut() {
            *value = match metric_type {
                MulticlassMetricType::Accuracy => {
                    let total_correct: usize = confusion_matrix.true_positives.iter().sum();
                    if self.total_samples > 0 {
                        total_correct as f64 / self.total_samples as f64
                    } else {
                        0.0
                    }
                }

                MulticlassMetricType::Specificity => {
                    let true_negatives: usize = confusion_matrix.true_negatives.iter().sum();
                    let false_positives: usize = confusion_matrix.false_positives.iter().sum();
                    let denominator: f64 = (true_negatives + false_positives) as f64;
                    if denominator > 0.0 {
                        true_negatives as f64 / denominator
                    } else {
                        0.0
                    }
                }

                MulticlassMetricType::TypeIError => {
                    let false_positives: usize = confusion_matrix.false_positives.iter().sum();
                    let true_negatives: usize = confusion_matrix.true_negatives.iter().sum();
                    let denominator: f64 = (false_positives + true_negatives) as f64;
                    if denominator > 0.0 {
                        false_positives as f64 / denominator
                    } else {
                        0.0
                    }
                }

                MulticlassMetricType::TypeIIError => {
                    let false_negatives: usize = confusion_matrix.false_negatives.iter().sum();
                    let true_positives: usize = confusion_matrix.true_positives.iter().sum();
                    let denominator: f64 = (false_negatives + true_positives) as f64;
                    if denominator > 0.0 {
                        false_negatives as f64 / denominator
                    } else {
                        0.0
                    }
                }

                MulticlassMetricType::MacroRecall => {
                    let recall: f64 = (0..number_of_classes)
                        .map(|class_index: usize| {
                            let denominator: f64 = (confusion_matrix.true_positives[class_index]
                                + confusion_matrix.false_negatives[class_index])
                                as f64;
                            if denominator > 0.0 {
                                confusion_matrix.true_positives[class_index] as f64 / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>()
                        / number_of_classes as f64;
                    recall
                }

                MulticlassMetricType::MacroPrecision => {
                    let precision: f64 = (0..number_of_classes)
                        .map(|class_index: usize| {
                            let denominator: f64 = (confusion_matrix.true_positives[class_index]
                                + confusion_matrix.false_positives[class_index])
                                as f64;
                            if denominator > 0.0 {
                                confusion_matrix.true_positives[class_index] as f64 / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>()
                        / number_of_classes as f64;
                    precision
                }

                MulticlassMetricType::MacroF1Score => {
                    let f1_score: f64 = (0..number_of_classes)
                        .map(|class_index: usize| {
                            let denominator: f64 =
                                (2 * confusion_matrix.true_positives[class_index]
                                    + confusion_matrix.false_positives[class_index]
                                    + confusion_matrix.false_negatives[class_index])
                                    as f64;
                            if denominator > 0.0 {
                                2.0 * confusion_matrix.true_positives[class_index] as f64
                                    / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>()
                        / number_of_classes as f64;
                    f1_score
                }

                MulticlassMetricType::WeightedRecall => {
                    let recall: f64 = (0..number_of_classes)
                        .map(|class_index: usize| {
                            let denominator: f64 = (confusion_matrix.true_positives[class_index]
                                + confusion_matrix.false_negatives[class_index])
                                as f64;
                            if denominator > 0.0 {
                                confusion_matrix.weight_classes[class_index]
                                    * confusion_matrix.true_positives[class_index] as f64
                                    / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    recall
                }

                MulticlassMetricType::WeightedPrecision => {
                    let precision: f64 = (0..number_of_classes)
                        .map(|class_index: usize| {
                            let denominator: f64 = (confusion_matrix.true_positives[class_index]
                                + confusion_matrix.false_positives[class_index])
                                as f64;
                            if denominator > 0.0 {
                                confusion_matrix.weight_classes[class_index]
                                    * confusion_matrix.true_positives[class_index] as f64
                                    / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    precision
                }

                MulticlassMetricType::WeightedF1Score => {
                    let f1_score: f64 = (0..number_of_classes)
                        .map(|class_index: usize| {
                            let denominator: f64 =
                                (2 * confusion_matrix.true_positives[class_index]
                                    + confusion_matrix.false_positives[class_index]
                                    + confusion_matrix.false_negatives[class_index])
                                    as f64;
                            if denominator > 0.0 {
                                confusion_matrix.weight_classes[class_index]
                                    * 2.0
                                    * confusion_matrix.true_positives[class_index] as f64
                                    / denominator
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>();
                    f1_score
                }
            };
        }
    }
}
