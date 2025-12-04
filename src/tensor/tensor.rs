use std::{
    ops::Deref,
    sync::{Arc, RwLock},
};

use crate::{
    error::NeuralNetworkError,
    tensor::{
        storage::{cpu_storage::CpuStorage, storage::StorageBackend},
        Device, Layout,
    },
};

struct Tensor_ {
    storage: Arc<RwLock<Box<dyn StorageBackend>>>,
    layout: Layout,
    gradient: Option<Tensor>,
    require_gradient: bool,
}

#[derive(Clone)]
pub struct Tensor(Arc<Tensor_>);

impl Deref for Tensor {
    type Target = Tensor_;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    pub fn new(
        data: Vec<f32>,
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, NeuralNetworkError> {
        let size: usize = shape.iter().product();
        if data.len() != size {
            return Err(NeuralNetworkError::IncompatibleShape {
                shape_given: shape,
                tensor_shape: vec![data.len()],
            });
        }

        let storage: Box<dyn StorageBackend> = match device {
            Device::CPU => Box::new(CpuStorage::from_vec(data)),
            Device::CUDA(_id) => todo!(),
            Device::Metal(_id) => todo!(),
        };

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::new(shape),
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn zeros(shape: Vec<usize>, device: Device) -> Result<Self, NeuralNetworkError> {
        let size: usize = shape.iter().product();
        let storage: Box<dyn StorageBackend> = match device {
            Device::CPU => Box::new(CpuStorage::new(size)),
            Device::CUDA(_id) => todo!(),
            Device::Metal(_id) => todo!(),
        };

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::new(shape),
            gradient: None,
            require_gradient: false,
        })))
    }

    pub fn ones(shape: Vec<usize>, device: Device) -> Result<Self, NeuralNetworkError> {
        let size: usize = shape.iter().product();
        let storage: Box<dyn StorageBackend> = match device {
            Device::CPU => Box::new(CpuStorage::filled(size, 1.0)),
            Device::CUDA(_id) => todo!(),
            Device::Metal(_id) => todo!(),
        };

        Ok(Tensor(Arc::new(Tensor_ {
            storage: Arc::new(RwLock::new(storage)),
            layout: Layout::new(shape),
            gradient: None,
            require_gradient: false,
        })))
    }
}
