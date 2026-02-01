use ndarray::{ArrayD, Axis};
use ndarray_stats::QuantileExt;

#[derive(Clone, PartialEq, Debug, Default, PartialOrd)]
pub struct ConfusionMatrix {
    pub matrix: Vec<Vec<usize>>,
    pub number_of_class: usize,
    pub weight_classes: Vec<f64>,

    pub true_positives: Vec<usize>,
    pub true_negatives: Vec<usize>,
    pub false_positives: Vec<usize>,
    pub false_negatives: Vec<usize>,
}

impl ConfusionMatrix {
    /// # Arguments
    /// * `predictions` a batched probability distribution of shape (n, i)
    /// * `observed` a batched observed values of shape (n, i)
    pub(crate) fn from(predictions: &ArrayD<f64>, observed: &ArrayD<f64>) -> Self {
        let number_of_class: usize = predictions.len_of(Axis(1));
        let n_samples: usize = predictions.len_of(Axis(0));

        let mut matrix: Vec<Vec<usize>> = vec![vec![0; number_of_class]; number_of_class];

        let predicted_classes: ArrayD<usize> =
            predictions.map_axis(Axis(1), |probabilities| {
                match probabilities.argmax() {
                    Ok(idx) => idx,
                    Err(e) => {
                        eprintln!("argmax failed on probabilities: {:?}", probabilities);
                        eprintln!("Error: {:?}", e);
                        eprintln!("Has NaN: {}", probabilities.iter().any(|x| x.is_nan()));
                        eprintln!("Has Inf: {}", probabilities.iter().any(|x| x.is_infinite()));
                        panic!("argmax failed: {:?}", e);
                    }
                }
            });
        let true_classes: ArrayD<usize> =
            observed.map_axis(Axis(1), |one_hot| one_hot.argmax().unwrap());

        let mut weight_classes: Vec<f64> = vec![0.0; number_of_class];
        for &class_i in &true_classes {
            weight_classes[class_i] += 1.0;
        }
        weight_classes = weight_classes
            .into_iter()
            .map(|weight: f64| weight / n_samples as f64)
            .collect();

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

        let true_negatives: Vec<usize> = (0..number_of_class)
            .map(|class: usize| {
                n_samples - true_positives[class] - false_positives[class] - false_negatives[class]
            })
            .collect();

        Self {
            matrix,
            number_of_class,
            weight_classes,

            true_positives,
            true_negatives,
            false_positives,
            false_negatives,
        }
    }

    pub fn add(&mut self, other: &ConfusionMatrix) {
        if self.number_of_class == 0 {
            *self = other.clone();
            return;
        }

        assert_eq!(
            self.number_of_class, other.number_of_class,
            "Cannot add confusion matrices with different number of classes"
        );

        for class_index in 0..self.number_of_class {
            self.true_positives[class_index] += other.true_positives[class_index];
            self.true_negatives[class_index] += other.true_negatives[class_index];
            self.false_positives[class_index] += other.false_positives[class_index];
            self.false_negatives[class_index] += other.false_negatives[class_index];

            for predicted_class in 0..self.number_of_class {
                self.matrix[class_index][predicted_class] +=
                    other.matrix[class_index][predicted_class];
            }
        }

        let total_samples: usize = self
            .true_positives
            .iter()
            .zip(self.false_negatives.iter())
            .map(|(true_positive, false_negative)| true_positive + false_negative)
            .sum();

        if total_samples > 0 {
            for class_index in 0..self.number_of_class {
                let class_samples: usize =
                    self.true_positives[class_index] + self.false_negatives[class_index];
                self.weight_classes[class_index] = class_samples as f64 / total_samples as f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

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
    fn test_confusion_matrix_weight_classes() {
        let predictions: ArrayD<f64> = arr2(&[
            [0.9, 0.1], // class 0
            [0.8, 0.2], // class 0
            [0.1, 0.9], // class 1
            [0.2, 0.8], // class 1
            [0.3, 0.7], // class 1
        ])
        .into_dyn();

        let observed: ArrayD<f64> = arr2(&[
            [1.0, 0.0], // class 0
            [1.0, 0.0], // class 0
            [0.0, 1.0], // class 1
            [0.0, 1.0], // class 1
            [0.0, 1.0], // class 1
        ])
        .into_dyn();

        let confusion_matrix: ConfusionMatrix = ConfusionMatrix::from(&predictions, &observed);

        assert!((confusion_matrix.weight_classes[0] - 0.4).abs() < 1e-10);
        assert!((confusion_matrix.weight_classes[1] - 0.6).abs() < 1e-10);

        let sum_weights: f64 = confusion_matrix.weight_classes.iter().sum();
        assert!((sum_weights - 1.0).abs() < 1e-10);
    }
}
