use super::Metrics;

#[derive(Clone, PartialEq, Debug)]
pub struct Benchmark {
    pub metrics: Metrics,
    pub loss: f64,
}

impl Benchmark {
    pub fn new(metrics: Metrics) -> Self {
        Self { metrics, loss: 0f64 }
    }
}
