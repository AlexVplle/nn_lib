use std::sync::Arc;

use crate::tensor::{storage::Storage, Layout};

#[derive(Eq, PartialEq, Debug, Clone, Default, PartialOrd, Ord, Hash)]
pub struct Tensor<T, D> {
    storage: Arc<Storage>,
    layout: Layout,
    device: D,
    gradient: Vec<T>,
}
