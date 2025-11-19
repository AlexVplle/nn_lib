use super::{Benchmark, MulticlassMetricType};

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

    pub fn get_metric_time_series(
        &self,
        metric_type: MulticlassMetricType,
    ) -> Option<Vec<f64>> {
        self.history
            .iter()
            .map(|h: &Benchmark| h.metrics.get_metric(metric_type))
            .collect::<Option<Vec<_>>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metrics;

    #[test]
    fn test_new_history() {
        let history: History = History::new();

        assert_eq!(history.history.len(), 0);
    }

    #[test]
    fn test_get_loss_time_series_empty() {
        let history: History = History::new();
        let loss_series: Vec<f64> = history.get_loss_time_series();

        assert_eq!(loss_series.len(), 0);
    }

    #[test]
    fn test_get_loss_time_series_with_data() {
        let mut history: History = History::new();

        let metric_types: Vec<MulticlassMetricType> = vec![MulticlassMetricType::Accuracy];
        let metrics: Metrics = Metrics::multiclass_classification(&metric_types);

        let mut benchmark1: Benchmark = Benchmark::new(metrics.clone());
        benchmark1.loss = 0.5;

        let mut benchmark2: Benchmark = Benchmark::new(metrics.clone());
        benchmark2.loss = 0.3;

        let mut benchmark3: Benchmark = Benchmark::new(metrics.clone());
        benchmark3.loss = 0.1;

        history.history.push(benchmark1);
        history.history.push(benchmark2);
        history.history.push(benchmark3);

        let loss_series: Vec<f64> = history.get_loss_time_series();

        assert_eq!(loss_series, vec![0.5, 0.3, 0.1]);
    }

    #[test]
    fn test_get_metric_time_series_empty() {
        let history: History = History::new();
        let metric_series: Option<Vec<f64>> =
            history.get_metric_time_series(MulticlassMetricType::Accuracy);

        assert_eq!(metric_series, Some(vec![]));
    }

    #[test]
    fn test_get_metric_time_series_with_nonexistent_metric() {
        let mut history: History = History::new();

        let metric_types: Vec<MulticlassMetricType> = vec![MulticlassMetricType::Accuracy];
        let metrics: Metrics = Metrics::multiclass_classification(&metric_types);
        let benchmark: Benchmark = Benchmark::new(metrics);

        history.history.push(benchmark);

        let metric_series: Option<Vec<f64>> =
            history.get_metric_time_series(MulticlassMetricType::MacroPrecision);

        assert_eq!(metric_series, None);
    }
}
