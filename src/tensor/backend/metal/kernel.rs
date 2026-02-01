use std::{collections::HashMap, sync::RwLock};

use crate::tensor::backend::metal::{
    compute_pipeline::ComputePipeline, constant_values::ConstantValues, device::Device,
    library::Library, source::Source,
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

    pub fn get_or_create_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, crate::tensor::backend::metal::error::MetalError> {
        let mut libraries = self.libraries.write().unwrap();
        if let Some(library) = libraries.get(&source) {
            return Ok(library.clone());
        }

        let source_code = source.code();
        let compile_options = objc2_metal::MTLCompileOptions::new();
        #[allow(deprecated)]
        compile_options.setFastMathEnabled(true);

        let library = device.new_library_with_source(source_code, Some(&compile_options))?;
        libraries.insert(source, library.clone());
        Ok(library)
    }

    pub fn get_or_create_pipeline(
        &self,
        device: &Device,
        function_name: &str,
        library: &Library,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipeline, crate::tensor::backend::metal::error::MetalError> {
        let key = (function_name.to_string(), constants.clone());

        let mut pipelines = self.pipelines.write().unwrap();
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(pipeline.clone());
        }

        let function = library.get_function(function_name, constants.as_ref())?;
        let pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}
