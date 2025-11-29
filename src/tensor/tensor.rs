#[derive(Eq, PartialEq, Debug, Clone, Default, PartialOrd, Ord, Hash)]
pub struct Tensor<T, D> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    device: D,
    gradient: Vec<T>,
}

impl Tensor<T, D> {
    fn new(arg: Type) -> Self {
        Self {}
    }
}
