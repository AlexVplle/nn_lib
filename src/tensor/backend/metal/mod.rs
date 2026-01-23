use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use objc2_metal::MTLResourceOptions;

use crate::tensor::{
    backend::{
        backend_device::BackendDevice,
        cpu::CpuStorage,
        metal::{
            buffer::{Buffer, BufferMap},
            commands::Commands,
            device::Device,
            encoder::{BlitCommandEncoder, ComputeCommandEncoder},
            error::MetalError,
            kernel::Kernels,
            storage::MetalStorage,
        },
    },
    TensorError,
};

pub mod buffer;
pub mod command_buffer;
pub mod command_semaphore;
pub mod commands;
pub mod compute_pipeline;
pub mod constant_values;
pub mod device;
pub mod encoder;
pub mod error;
pub mod function;
pub mod function_constant_values;
pub mod kernel;
pub mod library;
pub mod source;
pub mod storage;
pub mod value;

#[derive(Debug, Clone, PartialEq)]
struct DeviceId(usize);

impl DeviceId {
    pub fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone)]
pub struct MetalDevice {
    id: DeviceId,
    device: Device,
    commands: Arc<RwLock<Commands>>,
    buffers: Arc<RwLock<BufferMap>>,
    kernels: Arc<Kernels>,
}

impl MetalDevice {
    pub fn recommended_max_working_set_size(&self) -> usize {
        self.device.recommended_max_working_set_size()
    }

    pub fn current_allocated_size(&self) -> usize {
        self.device.current_allocated_size()
    }

    pub fn allocate_buffer(&self, size: usize) -> Result<Arc<Buffer>, MetalError> {
        let mut buffers = self
            .buffers
            .write()
            .map_err(|_| MetalError::BufferLockFailed("buffer map".to_string()))?;
        if let Some(buffer) = find_available_buffer(size, &buffers) {
            return Ok(buffer.clone());
        }
        let size = buffer_size(size);
        let subbuffers = buffers.entry(size).or_insert(vec![]);
        let new_buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::StorageModeShared)?;
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }
    pub fn command_encoder(&self) -> Result<ComputeCommandEncoder, MetalError> {
        let commands = self
            .commands
            .write()
            .map_err(|_| MetalError::CommandLockFailed("commands".to_string()))?;
        let (flush, command_encoder) = commands.command_encoder()?;
        if flush {
            self.drop_unused_buffers()?
        }
        Ok(command_encoder)
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandEncoder, MetalError> {
        let commands = self
            .commands
            .write()
            .map_err(|_| MetalError::CommandLockFailed("commands".to_string()))?;
        let (flush, command_encoder) = commands.blit_command_encoder()?;
        if flush {
            self.drop_unused_buffers()?
        }
        Ok(command_encoder)
    }

    fn drop_unused_buffers(&self) -> Result<(), MetalError> {
        let mut buffers = self
            .buffers
            .write()
            .map_err(|_| MetalError::BufferLockFailed("buffer map".to_string()))?;
        for subbuffers in buffers.values_mut() {
            let newbuffers = subbuffers
                .iter()
                .filter(|s| Arc::strong_count(*s) > 1)
                .map(Arc::clone)
                .collect();
            *subbuffers = newbuffers;
        }
        Ok(())
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalError> {
        let commands = self
            .commands
            .write()
            .map_err(|_| MetalError::CommandLockFailed("commands".to_string()))?;
        commands.wait_until_completed()?;
        Ok(())
    }

    pub fn new_buffer_with_data(&self, data: &[f32]) -> Result<Arc<Buffer>, MetalError> {
        let size = data.len() * std::mem::size_of::<f32>();
        let new_buffer = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            size,
            MTLResourceOptions::StorageModeShared,
        )?;
        let mut buffers = self
            .buffers
            .write()
            .map_err(|_| MetalError::BufferLockFailed("buffer map".to_string()))?;
        let subbuffers = buffers.entry(size).or_insert(vec![]);
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }
}

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(ordinal: usize) -> Result<Self, TensorError> {
        let device = Device::all().swap_remove(ordinal);
        let command_queue = device.new_command_queue()?;
        let commands = Commands::new(command_queue)?;
        let kernels = Arc::new(Kernels::new());
        Ok(Self {
            id: DeviceId::new(),
            device,
            commands: Arc::new(RwLock::new(commands)),
            buffers: Arc::new(RwLock::new(HashMap::new())),
            kernels,
        })
    }

    fn storage_from_cpu_storage_owned(
        &self,
        cpu_storage: CpuStorage,
    ) -> Result<Self::Storage, TensorError> {
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn storage_from_cpu_storage(
        &self,
        cpu_storage: &CpuStorage,
    ) -> Result<Self::Storage, TensorError> {
        let count = cpu_storage.0.len();
        let buffer = self.new_buffer_with_data(&cpu_storage.0)?;
        Ok(Self::Storage::new(buffer, self.clone(), count))
    }

    fn storage_from_vec(&self, data: Vec<f32>) -> Result<Self::Storage, TensorError> {
        let count = data.len();
        let buffer = self.new_buffer_with_data(&data)?;
        Ok(Self::Storage::new(buffer, self.clone(), count))
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }
}

fn find_available_buffer(size: usize, buffers: &BufferMap) -> Option<Arc<Buffer>> {
    let mut best_buffer: Option<&Arc<Buffer>> = None;
    let mut best_buffer_size = usize::MAX;
    for (buffer_size, subbuffers) in buffers.iter() {
        if buffer_size >= &size && buffer_size < &best_buffer_size {
            for sub in subbuffers {
                if Arc::strong_count(sub) == 1 {
                    best_buffer = Some(sub);
                    best_buffer_size = *buffer_size
                }
            }
        }
    }
    best_buffer.cloned()
}

fn buffer_size(size: usize) -> usize {
    size.saturating_sub(1).next_power_of_two()
}
