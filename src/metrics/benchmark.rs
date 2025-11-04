use super::{Metrics, MetricsType};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_new() {
        let metrics_types: Vec<MetricsType> =
            vec![MetricsType::Accuracy, MetricsType::MacroPrecision];
        let benchmark: Benchmark = Benchmark::new(&metrics_types);

        assert_eq!(benchmark.loss, 0.0);
        assert_eq!(benchmark.metrics.metrics.len(), 2);
        assert_eq!(
            benchmark.metrics.get_metric(MetricsType::Accuracy),
            Some(0.0)
        );
        assert_eq!(
            benchmark.metrics.get_metric(MetricsType::MacroPrecision),
            Some(0.0)
        );
    }
}
