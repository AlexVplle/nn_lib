use std::error::Error;

#[derive(PartialEq, Debug, Clone, Default, PartialOrd)]
pub struct CpuStorage {
    data: Box<[f32]>,
}

impl CpuStorage {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size].into_boxed_slice(),
        }
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        Self {
            data: data.into_boxed_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn try_clone(&self) -> Result<Self, Error> {
        Ok(CpuStorage {
            data: self.data.to_vec().into_boxed_slice(),
        })
    }
}
