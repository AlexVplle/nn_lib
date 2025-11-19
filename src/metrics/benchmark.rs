use super::Metrics;

#[derive(Clone, PartialEq, Debug)]
pub struct Benchmark {
    pub metrics: Metrics,
    pub loss: f64,
}

impl Benchmark {
    pub fn new(metrics: Metrics) -> Self {
        Self {
            metrics,
            loss: 0f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::MulticlassMetricType;

    #[test]
    fn test_new_benchmark() {
        let metric_types: Vec<MulticlassMetricType> = vec![
            MulticlassMetricType::Accuracy,
            MulticlassMetricType::MacroPrecision,
            MulticlassMetricType::MacroRecall,
        ];
        let metrics: Metrics = Metrics::multiclass_classification(&metric_types);
        let metrics_clone: Metrics = metrics.clone();
        let benchmark: Benchmark = Benchmark::new(metrics);

        assert_eq!(benchmark.loss, 0.0);
        assert_eq!(benchmark.metrics, metrics_clone);
    }
}
