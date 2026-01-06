use std::{ffi::c_void, ptr};

use objc2::rc::Retained;
use objc2_metal::{MTLDataType, MTLFunctionConstantValues};

#[derive(Eq, PartialEq, Debug, Clone, Default, Hash)]
pub struct FunctionConstantValues {
    pub raw: Retained<MTLFunctionConstantValues>,
}

impl FunctionConstantValues {
    pub fn new() -> Self {
        Self {
            raw: MTLFunctionConstantValues::new(),
        }
    }

    pub fn set_constant_value_at_index<T>(&self, value: &T, dtype: MTLDataType, index: usize) {
        let value = ptr::NonNull::new(value as *const T as *mut c_void).unwrap();
        unsafe {
            self.raw.setConstantValue_type_atIndex(value, dtype, index);
        }
    }
}
