use std::{collections::HashMap, sync::RwLock};

use crate::tensor::backend::metal::{
    compute_pipeline::ComputePipeline, constant_values::ConstantValues, library::Library,
    source::Source,
};

type Libraries = HashMap<Source, Library>;
type Pipelines = HashMap<(String, Option<ConstantValues>), ComputePipeline>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let pipelines = RwLock::new(Pipelines::new());
        Self {
            libraries,
            pipelines,
        }
    }
}
