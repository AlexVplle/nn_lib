use crate::tensor::backend::metal::{
    function_constant_values::FunctionConstantValues, value::Value,
};

#[derive(PartialEq, Debug, Clone, Hash)]
pub struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    pub fn function_constant_values(&self) -> FunctionConstantValues {
        let function_constant_values = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::USize(v) => {
                    function_constant_values.set_constant_value_at_index(v, ty, *index)
                }
                Value::F32(v) => {
                    function_constant_values.set_constant_value_at_index(v, ty, *index)
                }
                Value::U16(v) => {
                    function_constant_values.set_constant_value_at_index(v, ty, *index)
                }
                Value::Bool(v) => {
                    function_constant_values.set_constant_value_at_index(v, ty, *index)
                }
            }
        }
        function_constant_values
    }
}
