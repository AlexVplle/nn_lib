use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};

use crate::tensor::backend::metal::{
    command_buffer::CommandBuffer,
    command_semaphore::{CommandSemaphore, CommandStatus},
    device::CommandQueue,
    encoder::{BlitCommandEncoder, ComputeCommandEncoder},
    error::MetalError,
};

pub fn create_command_buffer(
    command_queue: &CommandQueue,
    semaphore: Arc<CommandSemaphore>,
) -> Result<CommandBuffer, MetalError> {
    command_queue
        .commandBuffer()
        .map(|raw| CommandBuffer::new(raw, semaphore))
        .ok_or(MetalError::CommandBufferCreationFailed)
}

#[derive(Debug)]
pub struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
}

#[derive(Debug)]
pub struct CommandBufferEntry {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    semaphore: Arc<CommandSemaphore>,
}

#[derive(Debug)]
pub struct Commands {
    pool: Vec<Arc<CommandBufferEntry>>,
    command_queue: CommandQueue,
    compute_per_buffer: usize,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalError> {
        let compute_per_buffer = 50;
        let pool_size = 5;
        let pool: Vec<Arc<CommandBufferEntry>> = (0..pool_size)
            .map(|_| Self::create_pool_entry(&command_queue))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            pool,
            command_queue,
            compute_per_buffer,
        })
    }

    fn create_pool_entry(
        command_queue: &CommandQueue,
    ) -> Result<Arc<CommandBufferEntry>, MetalError> {
        let semaphore = Arc::new(CommandSemaphore::new());
        let command_buffer = create_command_buffer(command_queue, Arc::clone(&semaphore))?;
        Ok(Arc::new(CommandBufferEntry {
            state: Mutex::new(EntryState {
                current: command_buffer,
                in_flight: Vec::new(),
            }),
            compute_count: AtomicUsize::new(0),
            semaphore,
        }))
    }

    pub fn command_encoder(&self) -> Result<(bool, ComputeCommandEncoder), MetalError> {
        let entry = self.select_entry()?;
        self.finalize_entry(entry, |command_buffer| {
            command_buffer.compute_command_encoder()
        })
    }
    pub fn blit_command_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalError> {
        let entry = self.select_entry()?;
        self.finalize_entry(entry, |command_buffer| {
            command_buffer.blit_command_encoder()
        })
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalError> {
        self.flush_and_wait()
    }

    pub fn flush_and_wait(&self) -> Result<(), MetalError> {
        for entry in &self.pool {
            let to_wait = {
                let _guard = entry
                    .semaphore
                    .wait_until(|s| matches!(s, CommandStatus::Available));
                let mut state = entry.state.lock()?;
                if entry.compute_count.load(Ordering::Acquire) > 0 {
                    self.commit_swap_locked(&entry, &mut state, 0)?;
                }
                std::mem::take(&mut state.in_flight)
            };
            for cb in to_wait {
                Self::ensure_completed(&cb)?;
            }
        }
        Ok(())
    }

    fn ensure_completed(command_buffer: &CommandBuffer) -> Result<(), MetalError> {
        match command_buffer.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                command_buffer.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => {
                let message = command_buffer
                    .error()
                    .map(|error| error.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalError::CommandBufferExecutionError(message));
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn select_entry(&self) -> Result<Arc<CommandBufferEntry>, MetalError> {
        for entry in &self.pool {
            if let Ok(mut status) = entry.semaphore.status.try_lock() {
                if matches!(*status, CommandStatus::Available) {
                    *status = CommandStatus::Encoding;
                    return Ok(Arc::clone(entry));
                }
            }
        }
        let entry = self
            .pool
            .iter()
            .max_by_key(|e| e.compute_count.load(Ordering::Acquire))
            .ok_or(MetalError::CommandBufferPoolEmpty)?;
        let entry = Arc::clone(entry);
        {
            let mut guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));
            *guard = CommandStatus::Encoding;
        }
        Ok(entry)
    }

    fn finalize_entry<F, E>(
        &self,
        entry: Arc<CommandBufferEntry>,
        create_encoder: F,
    ) -> Result<(bool, E), MetalError>
    where
        F: FnOnce(&mut CommandBuffer) -> E,
    {
        let mut state = entry.state.lock()?;
        let count = entry.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;
        if flush {
            self.commit_swap_locked(&entry, &mut state, 1)?;
        }
        let encoder = create_encoder(&mut state.current);
        Ok((flush, encoder))
    }

    fn commit_swap_locked(
        &self,
        entry: &CommandBufferEntry,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalError> {
        state.current.commit();
        let old_buffer = std::mem::replace(
            &mut state.current,
            create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?,
        );
        state.in_flight.push(old_buffer);
        entry.compute_count.store(reset_to, Ordering::Release);
        Ok(())
    }
}
