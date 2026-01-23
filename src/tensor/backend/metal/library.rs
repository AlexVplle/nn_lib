use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::MTLLibrary;

use crate::tensor::backend::metal::{
    constant_values::ConstantValues, error::MetalError, function::Function,
};

#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct Library {
    raw: Retained<ProtocolObject<dyn MTLLibrary>>,
}

unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLLibrary>>) -> Self {
        Self { raw }
    }

    pub fn get_function(
        &self,
        name: &str,
        constant_values: Option<&ConstantValues>,
    ) -> Result<Function, MetalError> {
        let function = match constant_values {
            Some(constant_values) => self
                .raw
                .newFunctionWithName_constantValues_error(
                    &NSString::from_str(name),
                    &constant_values.function_constant_values().raw,
                )
                .map_err(|error| MetalError::LibraryCompilationError(error.to_string()))?,
            None => self
                .raw
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(MetalError::FunctionNotFound(name.to_string()))?,
        };
        Ok(Function::new(function))
    }
}
