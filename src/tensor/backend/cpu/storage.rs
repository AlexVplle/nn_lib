use crate::tensor::{
    backend::{backend_storage::BackendStorage, cpu::CpuDevice},
    TensorError,
};

#[derive(PartialEq, Debug, Clone, Default, PartialOrd)]
pub struct CpuStorage(pub Vec<f32>);

impl BackendStorage for CpuStorage {
    type Device = CpuDevice;

    fn device(&self) -> &Self::Device {
        &CpuDevice
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage, TensorError> {
        Ok(self.clone())
    }

    fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        let result: Vec<f32> = self
            .0
            .iter()
            .zip(rhs.0.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(CpuStorage(result))
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
        let mut result: Vec<f32> = vec![0.0; m * n];

        let lhs_row_stride: usize = lhs_strides[0];
        let lhs_col_stride: usize = lhs_strides[1];
        let rhs_row_stride: usize = rhs_strides[0];
        let rhs_col_stride: usize = rhs_strides[1];

        for i in 0..m {
            for j in 0..n {
                let mut sum: f32 = 0.0;
                for p in 0..k {
                    let lhs_idx: usize = i * lhs_row_stride + p * lhs_col_stride;
                    let rhs_idx: usize = p * rhs_row_stride + j * rhs_col_stride;
                    sum += self.0[lhs_idx] * rhs.0[rhs_idx];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(CpuStorage(result))
    }

    fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.0.len() != rhs.0.len() {
            return Err(TensorError::DimensionMismatch);
        }

        let result: Vec<f32> = self
            .0
            .iter()
            .zip(rhs.0.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(CpuStorage(result))
    }

    fn relu(&self) -> Result<Self, TensorError> {
        let result: Vec<f32> = self.0.iter().map(|&x| x.max(0.0)).collect();
        Ok(CpuStorage(result))
    }

    fn tanh(&self) -> Result<Self, TensorError> {
        let result: Vec<f32> = self.0.iter().map(|&x| x.tanh()).collect();
        Ok(CpuStorage(result))
    }

    fn sigmoid(&self) -> Result<Self, TensorError> {
        let result: Vec<f32> = self.0.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Ok(CpuStorage(result))
    }

    fn softmax(&self, batch_size: usize, vector_size: usize) -> Result<Self, TensorError> {
        if self.0.len() != batch_size * vector_size {
            return Err(TensorError::DimensionMismatch);
        }

        let mut result = vec![0.0; batch_size * vector_size];

        for b in 0..batch_size {
            let start = b * vector_size;
            let end = start + vector_size;
            let row = &self.0[start..end];

            let max_val = row.iter().fold(f32::NEG_INFINITY, |max, &val| max.max(val));
            let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exps: f32 = exps.iter().sum::<f32>() + 1e-10;

            for (i, &exp_val) in exps.iter().enumerate() {
                result[start + i] = exp_val / sum_exps;
            }
        }

        Ok(CpuStorage(result))
    }

    fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.0.len() != rhs.0.len() {
            return Err(TensorError::DimensionMismatch);
        }

        let result: Vec<f32> = self
            .0
            .iter()
            .zip(rhs.0.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(CpuStorage(result))
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        let result: Vec<f32> = self.0.iter().map(|&x| x * scalar).collect();
        Ok(CpuStorage(result))
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

        let mut input_strides: Vec<usize> = vec![1; input_shape.len()];
        for i in (0..input_shape.len() - 1).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        let mut output_strides: Vec<usize> = vec![1; output_shape.len()];
        if !output_shape.is_empty() {
            for i in (0..output_shape.len() - 1).rev() {
                output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
            }
        }

        let output_size: usize = if output_shape.is_empty() {
            1
        } else {
            output_shape.iter().product()
        };
        let axis_size: usize = input_shape[axis];

        let mut result: Vec<f32> = vec![0.0; output_size];

        for out_idx in 0..output_size {
            let mut out_indices: Vec<usize> = vec![0; output_shape.len()];
            let mut remaining: usize = out_idx;
            for (i, &stride) in output_strides.iter().enumerate() {
                out_indices[i] = remaining / stride;
                remaining %= stride;
            }

            let mut input_indices: Vec<usize> = Vec::with_capacity(input_shape.len());
            let mut out_pos: usize = 0;
            for i in 0..input_shape.len() {
                if i == axis {
                    input_indices.push(0);
                } else {
                    input_indices.push(out_indices[out_pos]);
                    out_pos += 1;
                }
            }

            for axis_idx in 0..axis_size {
                input_indices[axis] = axis_idx;

                let mut in_idx: usize = 0;
                for (i, &idx) in input_indices.iter().enumerate() {
                    in_idx += idx * input_strides[i];
                }

                result[out_idx] += self.0[in_idx];
            }
        }

        Ok(CpuStorage(result))
    }

    fn copy_strided(
        &self,
        shape: &[usize],
        strides: &[usize],
        base_offset: usize,
    ) -> Result<Self, TensorError> {
        let total_size: usize = shape.iter().product();
        let mut result: Vec<f32> = Vec::with_capacity(total_size);

        fn copy_recursive(
            data: &[f32],
            shape: &[usize],
            strides: &[usize],
            base_offset: usize,
            indices: &mut Vec<usize>,
            dim: usize,
            out: &mut Vec<f32>,
        ) {
            if dim == shape.len() {
                let mut offset: usize = base_offset;
                for (i, &stride) in strides.iter().enumerate() {
                    offset += indices[i] * stride;
                }
                out.push(data[offset]);
                return;
            }

            for i in 0..shape[dim] {
                indices[dim] = i;
                copy_recursive(data, shape, strides, base_offset, indices, dim + 1, out);
            }
        }

        let mut indices: Vec<usize> = vec![0; shape.len()];
        copy_recursive(&self.0, shape, strides, base_offset, &mut indices, 0, &mut result);

        Ok(CpuStorage(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let input = CpuStorage(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let output = input.relu().unwrap();
        assert_eq!(output.0, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tanh() {
        let input = CpuStorage(vec![-1.0, 0.0, 1.0]);
        let output = input.tanh().unwrap();

        assert!((output.0[0] - (-0.7616)).abs() < 0.001);
        assert!((output.0[1] - 0.0).abs() < 0.001);
        assert!((output.0[2] - 0.7616).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid() {
        let input = CpuStorage(vec![-2.0, 0.0, 2.0]);
        let output = input.sigmoid().unwrap();

        assert!((output.0[0] - 0.1192).abs() < 0.001);
        assert!((output.0[1] - 0.5).abs() < 0.001);
        assert!((output.0[2] - 0.8808).abs() < 0.001);
    }

    #[test]
    fn test_softmax() {
        let input = CpuStorage(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let output = input.softmax(2, 3).unwrap();

        let sum1: f32 = output.0[0..3].iter().sum();
        assert!((sum1 - 1.0).abs() < 0.001);

        let sum2: f32 = output.0[3..6].iter().sum();
        assert!((sum2 - 1.0).abs() < 0.001);

        assert!(output.0[2] > output.0[1]);
        assert!(output.0[1] > output.0[0]);
    }

    #[test]
    fn test_add() {
        let a = CpuStorage(vec![1.0, 2.0, 3.0]);
        let b = CpuStorage(vec![4.0, 5.0, 6.0]);
        let result = a.add(&b).unwrap();
        assert_eq!(result.0, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul() {
        let a = CpuStorage(vec![1.0, 2.0, 3.0]);
        let b = CpuStorage(vec![2.0, 3.0, 4.0]);
        let result = a.mul(&b).unwrap();
        assert_eq!(result.0, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_matmul() {
        let a = CpuStorage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = CpuStorage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = a.matmul(&b, 2, 3, 2, &[3, 1], &[2, 1]).unwrap();
        assert_eq!(result.0, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_sub() {
        let a = CpuStorage(vec![5.0, 7.0, 9.0]);
        let b = CpuStorage(vec![1.0, 2.0, 3.0]);
        let result = a.sub(&b).unwrap();
        assert_eq!(result.0, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let a = CpuStorage(vec![1.0, 2.0, 3.0]);
        let result = a.mul_scalar(2.5).unwrap();
        assert_eq!(result.0, vec![2.5, 5.0, 7.5]);
    }

    #[test]
    fn test_sum_axis() {
        let a: CpuStorage = CpuStorage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let input_shape: &[usize] = &[2, 3];
        let output_shape: &[usize] = &[3];
        let result: CpuStorage = a.sum_axis(0, input_shape, output_shape).unwrap();
        assert_eq!(result.0, vec![5.0, 7.0, 9.0]);
    }
}
