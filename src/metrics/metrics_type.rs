#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default, PartialOrd, Ord)]
pub enum MetricsType {
    #[default]
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
