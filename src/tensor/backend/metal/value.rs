use std::hash::Hash;

use objc2_metal::MTLDataType;

#[derive(Debug, Clone, Copy, PartialOrd)]
pub enum Value {
    USize(usize),
    Bool(bool),
    F32(f32),
    U16(u16),
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::USize(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
            Value::F32(v) => v.to_bits().hash(state),
            Value::U16(v) => v.hash(state),
        }
    }
}

impl Eq for Value {}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::USize(a), Value::USize(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::F32(a), Value::F32(b)) => a.to_bits() == b.to_bits(),
            (Value::U16(a), Value::U16(b)) => a == b,
            _ => false,
        }
    }
}

impl Value {
    pub fn data_type(&self) -> MTLDataType {
        match self {
            Value::USize(_) => MTLDataType::ULong,
            Value::Bool(_) => MTLDataType::Bool,
            Value::F32(_) => MTLDataType::Float,
            Value::U16(_) => MTLDataType::UShort,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_hash_f32_zero() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let v1 = Value::F32(0.0);
        let v2 = Value::F32(-0.0);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        v1.hash(&mut hasher1);
        v2.hash(&mut hasher2);

        assert_ne!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_value_data_type() {
        assert_eq!(Value::USize(0).data_type(), MTLDataType::ULong);
        assert_eq!(Value::Bool(true).data_type(), MTLDataType::Bool);
        assert_eq!(Value::F32(0.0).data_type(), MTLDataType::Float);
        assert_eq!(Value::U16(0).data_type(), MTLDataType::UShort);
    }
}
