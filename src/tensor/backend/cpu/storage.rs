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

    fn matmul(&self, rhs: &Self, m: usize, k: usize, n: usize) -> Result<Self, TensorError> {
        if self.0.len() != m * k {
            return Err(TensorError::DimensionMismatch);
        }
        if rhs.0.len() != k * n {
            return Err(TensorError::DimensionMismatch);
        }

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.0[i * k + p] * rhs.0[p * n + j];
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
        let result = a.matmul(&b, 2, 3, 2).unwrap();
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
}
