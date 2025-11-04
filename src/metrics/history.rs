use super::{Benchmark, MetricsType};

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

#[cfg(test)]
mod tests {
    use super::*;

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

        let loss_series: Vec<f64> = history.get_loss_time_series();
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

        let metric_series: Option<Vec<f64>> =
            history.get_metric_time_series(MetricsType::MacroRecall);
        assert_eq!(metric_series, None);
    }
}
