use crate::tensor::{
    backend::{
        backend_storage::BackendStorage,
        cpu::CpuStorage,
        metal::{buffer::Buffer, error::MetalError, MetalDevice},
    },
    TensorError,
};

use objc2_metal::MTLSize;
use std::{slice, sync::Arc};

#[derive(Debug, Clone)]
pub struct MetalStorage {
    buffer: Arc<Buffer>,
    device: MetalDevice,
    count: usize,
}

impl MetalStorage {
    pub fn new(buffer: Arc<Buffer>, device: MetalDevice, count: usize) -> Self {
        Self {
            buffer,
            device,
            count,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn to_cpu(&self) -> Result<Vec<f32>, MetalError> {
        let size = self.count * std::mem::size_of::<f32>();
        let buffer = self.device.allocate_buffer(size)?;
        {
            let blit = self.device.blit_command_encoder()?;
            blit.set_label("blit to cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, size);
            blit.end_encoding();
        }
        self.device.wait_until_completed()?;
        Ok(read_to_vec(&buffer, self.count))
    }
}

impl BackendStorage for MetalStorage {
    type Device = MetalDevice;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage, TensorError> {
        Ok(CpuStorage(self.to_cpu()?))
    }

    fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.count != rhs.count {
            return Err(TensorError::DimensionMismatch);
        }

        let output_buffer = self
            .device
            .allocate_buffer(self.count * std::mem::size_of::<f32>())?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::Add, "add")?;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&rhs.buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);

            let grid_size = MTLSize {
                width: self.count,
                height: 1,
                depth: 1,
            };
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(self.count),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            self.count,
        ))
    }

    fn matmul(
        &self,
        rhs: &Self,
        m: usize,
        k: usize,
        n: usize,
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<Self, TensorError> {
        let output_count = m * n;
        let output_buffer = self
            .device
            .allocate_buffer(output_count * std::mem::size_of::<f32>())?;

        let lhs_contiguous = lhs_strides.len() == 2 && lhs_strides[0] == k && lhs_strides[1] == 1;
        let rhs_contiguous = rhs_strides.len() == 2 && rhs_strides[0] == n && rhs_strides[1] == 1;

        if lhs_contiguous && rhs_contiguous {
            let pipeline = self
                .device
                .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::Matmul, "matmul")?;

            let m_val = m as u32;
            let k_val = k as u32;
            let n_val = n as u32;

            {
                let encoder = self.device.command_encoder()?;
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&self.buffer), 0);
                encoder.set_buffer(1, Some(&rhs.buffer), 0);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                encoder.set_bytes(3, &m_val);
                encoder.set_bytes(4, &k_val);
                encoder.set_bytes(5, &n_val);

                const TILE_SIZE: usize = 32;
                let threadgroup_size = MTLSize {
                    width: TILE_SIZE,
                    height: TILE_SIZE,
                    depth: 1,
                };

                let grid_size = MTLSize {
                    width: (n + TILE_SIZE - 1) / TILE_SIZE,
                    height: (m + TILE_SIZE - 1) / TILE_SIZE,
                    depth: 1,
                };

                encoder.dispatch_thread_groups(grid_size, threadgroup_size);
            }
        } else {
            let pipeline = self
                .device
                .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::MatmulStrided, "matmul_strided")?;

            let m_val = m as u32;
            let k_val = k as u32;
            let n_val = n as u32;
            let lhs_row_stride = lhs_strides[0] as u32;
            let lhs_col_stride = lhs_strides[1] as u32;
            let rhs_row_stride = rhs_strides[0] as u32;
            let rhs_col_stride = rhs_strides[1] as u32;

            {
                let encoder = self.device.command_encoder()?;
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&self.buffer), 0);
                encoder.set_buffer(1, Some(&rhs.buffer), 0);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                encoder.set_bytes(3, &m_val);
                encoder.set_bytes(4, &k_val);
                encoder.set_bytes(5, &n_val);
                encoder.set_bytes(6, &lhs_row_stride);
                encoder.set_bytes(7, &lhs_col_stride);
                encoder.set_bytes(8, &rhs_row_stride);
                encoder.set_bytes(9, &rhs_col_stride);

                let grid_size = MTLSize {
                    width: n,
                    height: m,
                    depth: 1,
                };

                let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
                let threadgroup_width = (max_threads as f32).sqrt() as usize;
                let threadgroup_size = MTLSize {
                    width: threadgroup_width.min(n),
                    height: threadgroup_width.min(m),
                    depth: 1,
                };

                encoder.dispatch_threads(grid_size, threadgroup_size);
            }
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            output_count,
        ))
    }

    fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.count != rhs.count {
            return Err(TensorError::DimensionMismatch);
        }

        let output_buffer = self
            .device
            .allocate_buffer(self.count * std::mem::size_of::<f32>())?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::Mul, "mul")?;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&rhs.buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);

            let grid_size = MTLSize {
                width: self.count,
                height: 1,
                depth: 1,
            };
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(self.count),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            self.count,
        ))
    }

    fn relu(&self) -> Result<Self, TensorError> {
        let output_buffer = self
            .device
            .allocate_buffer(self.count * std::mem::size_of::<f32>())?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::ReLU, "relu")?;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);

            let grid_size = MTLSize {
                width: self.count,
                height: 1,
                depth: 1,
            };
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(self.count),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            self.count,
        ))
    }

    fn tanh(&self) -> Result<Self, TensorError> {
        let output_buffer = self
            .device
            .allocate_buffer(self.count * std::mem::size_of::<f32>())?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::Tanh, "tanh_kernel")?;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);

            let grid_size = MTLSize {
                width: self.count,
                height: 1,
                depth: 1,
            };
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(self.count),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            self.count,
        ))
    }

    fn sigmoid(&self) -> Result<Self, TensorError> {
        let output_buffer = self
            .device
            .allocate_buffer(self.count * std::mem::size_of::<f32>())?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::Sigmoid, "sigmoid")?;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);

            let grid_size = MTLSize {
                width: self.count,
                height: 1,
                depth: 1,
            };
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(self.count),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            self.count,
        ))
    }

    fn softmax(&self, batch_size: usize, vector_size: usize) -> Result<Self, TensorError> {
        if self.count != batch_size * vector_size {
            return Err(TensorError::DimensionMismatch);
        }

        let output_buffer = self
            .device
            .allocate_buffer(self.count * std::mem::size_of::<f32>())?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::Softmax, "softmax")?;

        let vector_size_u32 = vector_size as u32;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_bytes(2, &vector_size_u32);

            let grid_size = MTLSize {
                width: batch_size,
                height: 1,
                depth: 1,
            };
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(batch_size),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            self.count,
        ))
    }

    fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.count != rhs.count {
            return Err(TensorError::DimensionMismatch);
        }

        let output_buffer = self
            .device
            .allocate_buffer(self.count * std::mem::size_of::<f32>())?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::Sub, "sub")?;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&rhs.buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);

            let grid_size = MTLSize {
                width: self.count,
                height: 1,
                depth: 1,
            };
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(self.count),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            self.count,
        ))
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        let output_buffer = self
            .device
            .allocate_buffer(self.count * std::mem::size_of::<f32>())?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::MulScalar, "mul_scalar")?;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_bytes(2, &scalar);

            let grid_size = MTLSize {
                width: self.count,
                height: 1,
                depth: 1,
            };
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(self.count),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            self.count,
        ))
    }

    fn sum_axis(
        &self,
        axis: usize,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<Self, TensorError> {
        if axis >= input_shape.len() {
            return Err(TensorError::InvalidDimension {
                got: axis,
                max_dimension: input_shape.len(),
            });
        }

        let mut input_strides: Vec<u32> = vec![1; input_shape.len()];
        for i in (0..input_shape.len() - 1).rev() {
            input_strides[i] = (input_strides[i + 1] * input_shape[i + 1] as u32) as u32;
        }

        let mut output_strides: Vec<u32> = vec![1; output_shape.len()];
        if !output_shape.is_empty() {
            for i in (0..output_shape.len() - 1).rev() {
                output_strides[i] = (output_strides[i + 1] * output_shape[i + 1] as u32) as u32;
            }
        }

        let output_size: usize = if output_shape.is_empty() {
            1
        } else {
            output_shape.iter().product()
        };
        let axis_size: u32 = input_shape[axis] as u32;

        let output_buffer: Arc<Buffer> = self
            .device
            .allocate_buffer(output_size * std::mem::size_of::<f32>())?;

        let input_shape_u32: Vec<u32> = input_shape.iter().map(|&x| x as u32).collect();
        let output_shape_u32: Vec<u32> = output_shape.iter().map(|&x| x as u32).collect();

        let input_shape_buffer: Arc<Buffer> = self.device.storage_from_slice(&input_shape_u32)?;
        let output_shape_buffer: Arc<Buffer> = self.device.storage_from_slice(&output_shape_u32)?;
        let input_strides_buffer: Arc<Buffer> = self.device.storage_from_slice(&input_strides)?;
        let output_strides_buffer: Arc<Buffer> = self.device.storage_from_slice(&output_strides)?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::SumAxis, "sum_axis")?;

        let axis_u32: u32 = axis as u32;
        let input_ndim: u32 = input_shape.len() as u32;
        let output_ndim: u32 = output_shape.len() as u32;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_buffer(2, Some(&input_shape_buffer), 0);
            encoder.set_buffer(3, Some(&output_shape_buffer), 0);
            encoder.set_buffer(4, Some(&input_strides_buffer), 0);
            encoder.set_buffer(5, Some(&output_strides_buffer), 0);
            encoder.set_bytes(6, &axis_u32);
            encoder.set_bytes(7, &input_ndim);
            encoder.set_bytes(8, &output_ndim);
            encoder.set_bytes(9, &axis_size);

            let grid_size = MTLSize {
                width: output_size,
                height: 1,
                depth: 1,
            };
            let max_threads: usize = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(output_size),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            output_size,
        ))
    }

    fn copy_strided(
        &self,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, TensorError> {
        let total_size: usize = shape.iter().product();
        let ndim: u32 = shape.len() as u32;

        let output_buffer = self
            .device
            .allocate_buffer(total_size * std::mem::size_of::<f32>())?;

        let pipeline = self
            .device
            .get_or_create_pipeline(crate::tensor::backend::metal::source::Source::CopyStrided, "copy_strided")?;

        let shape_buffer = self.device.storage_from_slice(
            &shape.iter().map(|&x| x as u32).collect::<Vec<u32>>()
        )?;

        let strides_buffer = self.device.storage_from_slice(
            &strides.iter().map(|&x| x as u32).collect::<Vec<u32>>()
        )?;

        {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_buffer(2, Some(&shape_buffer), 0);
            encoder.set_buffer(3, Some(&strides_buffer), 0);
            encoder.set_bytes(4, &ndim);
            encoder.set_bytes(5, &(total_size as u32));

            let grid_size = MTLSize {
                width: total_size,
                height: 1,
                depth: 1,
            };
            let max_threads: usize = pipeline.max_total_threads_per_threadgroup() as usize;
            let threadgroup_size = MTLSize {
                width: max_threads.min(total_size),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        Ok(MetalStorage::new(
            output_buffer,
            self.device.clone(),
            total_size,
        ))
    }
}

fn read_to_vec(buffer: &Buffer, n: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const f32;
    assert!(!ptr.is_null());
    let slice = unsafe { slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::backend::backend_device::BackendDevice;

    fn create_test_device() -> MetalDevice {
        MetalDevice::new(0).expect("Failed to create Metal device")
    }

    fn create_storage(data: Vec<f32>, device: &MetalDevice) -> MetalStorage {
        let count = data.len();
        let size = count * std::mem::size_of::<f32>();
        let buffer = device.allocate_buffer(size).unwrap();

        let ptr = buffer.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, count);
        }

        MetalStorage::new(buffer, device.clone(), count)
    }

    #[test]
    fn test_relu() {
        let device = create_test_device();
        let input = create_storage(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &device);
        let output = input.relu().unwrap();
        let result = output.to_cpu().unwrap();

        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tanh() {
        let device = create_test_device();
        let input = create_storage(vec![-1.0, 0.0, 1.0], &device);
        let output = input.tanh().unwrap();
        let result = output.to_cpu().unwrap();

        assert!((result[0] - (-0.7616)).abs() < 0.001);
        assert!((result[1] - 0.0).abs() < 0.001);
        assert!((result[2] - 0.7616).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid() {
        let device = create_test_device();
        let input = create_storage(vec![-2.0, 0.0, 2.0], &device);
        let output = input.sigmoid().unwrap();
        let result = output.to_cpu().unwrap();

        assert!((result[0] - 0.1192).abs() < 0.001);
        assert!((result[1] - 0.5).abs() < 0.001);
        assert!((result[2] - 0.8808).abs() < 0.001);
    }

    #[test]
    fn test_softmax() {
        let device = create_test_device();
        let input = create_storage(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &device);
        let output = input.softmax(2, 3).unwrap();
        let result = output.to_cpu().unwrap();

        let sum1: f32 = result[0..3].iter().sum();
        assert!((sum1 - 1.0).abs() < 0.001);

        let sum2: f32 = result[3..6].iter().sum();
        assert!((sum2 - 1.0).abs() < 0.001);

        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_add() {
        let device = create_test_device();
        let a = create_storage(vec![1.0, 2.0, 3.0], &device);
        let b = create_storage(vec![4.0, 5.0, 6.0], &device);
        let result = a.add(&b).unwrap();
        let output = result.to_cpu().unwrap();

        assert_eq!(output, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul() {
        let device = create_test_device();
        let a = create_storage(vec![1.0, 2.0, 3.0], &device);
        let b = create_storage(vec![2.0, 3.0, 4.0], &device);
        let result = a.mul(&b).unwrap();
        let output = result.to_cpu().unwrap();

        assert_eq!(output, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_matmul() {
        let device = create_test_device();
        let a = create_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device);
        let b = create_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device);
        let result = a.matmul(&b, 2, 3, 2, &[3, 1], &[2, 1]).unwrap();
        let output = result.to_cpu().unwrap();

        assert_eq!(output, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_sub() {
        let device = create_test_device();
        let a = create_storage(vec![5.0, 7.0, 9.0], &device);
        let b = create_storage(vec![1.0, 2.0, 3.0], &device);
        let result = a.sub(&b).unwrap();
        let output = result.to_cpu().unwrap();

        assert_eq!(output, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let device = create_test_device();
        let a = create_storage(vec![1.0, 2.0, 3.0], &device);
        let result = a.mul_scalar(2.5).unwrap();
        let output = result.to_cpu().unwrap();

        assert_eq!(output, vec![2.5, 5.0, 7.5]);
    }

    #[test]
    fn test_sum_axis() {
        let device = create_test_device();
        let a = create_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device);
        let input_shape: &[usize] = &[2, 3];
        let output_shape: &[usize] = &[3];
        let result = a.sum_axis(0, input_shape, output_shape).unwrap();
        let output = result.to_cpu().unwrap();

        assert_eq!(output, vec![5.0, 7.0, 9.0]);
    }
}
