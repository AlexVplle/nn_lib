use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLFunction;

pub struct Function {
    raw: Retained<ProtocolObject<dyn MTLFunction>>,
}

impl AsRef<ProtocolObject<dyn MTLFunction>> for Function {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLFunction> {
        &self.raw
    }
}

impl Function {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLFunction>>) -> Self {
        Self { raw }
    }
}
