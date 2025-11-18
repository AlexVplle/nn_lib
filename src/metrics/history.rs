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
