use crate::tensor::{
    backend::{backend_device::BackendDevice, cpu::CpuStorage},
    TensorError,
};

pub trait BackendStorage: Sized {
    type Device: BackendDevice;

    fn device(&self) -> &Self::Device;
    fn to_cpu_storage(&self) -> Result<CpuStorage, TensorError>;
    fn add(&self, rhs: &Self) -> Result<Self, TensorError>;
    fn matmul(
        &self,
        rhs: &Self,
        m: usize,
        k: usize,
        n: usize,
        lhs_strides: &[usize],
        rhs_strides: &[usize],
    ) -> Result<Self, TensorError>;
    fn mul(&self, rhs: &Self) -> Result<Self, TensorError>;
    fn relu(&self) -> Result<Self, TensorError>;
    fn tanh(&self) -> Result<Self, TensorError>;
    fn sigmoid(&self) -> Result<Self, TensorError>;
    fn softmax(&self, batch_size: usize, vector_size: usize) -> Result<Self, TensorError>;
    fn sub(&self, rhs: &Self) -> Result<Self, TensorError>;
    fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorError>;
    fn sum_axis(
        &self,
        axis: usize,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<Self, TensorError>;

    fn copy_strided(
        &self,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, TensorError>;
}
