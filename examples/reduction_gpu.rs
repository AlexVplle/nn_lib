use std::marker::PhantomData;

use cubecl::{
    client::ComputeClient,
    cube,
    frontend::CompilationArg,
    prelude::{Float, FloatExpand, Tensor, TensorArg},
    server::Handle,
    std::tensor::compact_strides,
    CubeCount, CubeDim, CubeElement, Runtime,
};

#[derive(Debug)]
pub struct GpuTensor<R: Runtime, F: Float + CubeElement> {
    data: Handle,
    shape: Vec<usize>,
    strides: Vec<usize>,
    _r: PhantomData<R>,
    _f: PhantomData<F>,
}

impl<R: Runtime, F: Float + CubeElement> Clone for GpuTensor<R, F> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _r: PhantomData,
            _f: PhantomData,
        }
    }
}

impl<R: Runtime, F: Float + CubeElement> GpuTensor<R, F> {
    pub fn arange(shape: Vec<usize>, client: &ComputeClient<R::Server>) -> Self {
        let size: usize = shape.iter().product();
        let data: Vec<F> = (0..size).map(|i: usize| F::from_int(i as i64)).collect();
        let data: Handle = client.create(F::as_bytes(&data));
        let strides: Vec<usize> = compact_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    pub fn empty(shape: Vec<usize>, client: &ComputeClient<R::Server>) -> Self {
        let size: usize = shape.iter().product::<usize>() * core::mem::size_of::<F>();
        let data: Handle = client.empty(size);
        let strides: Vec<usize> = compact_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    pub fn into_tensor_arg(&self, line_size: u8) -> TensorArg<'_, R> {
        unsafe { TensorArg::from_raw_parts::<F>(&self.data, &self.strides, &self.shape, line_size) }
    }

    pub fn read(self, client: &ComputeClient<R::Server>) -> Vec<F> {
        let bytes = client.read_one(self.data);
        F::from_bytes(&bytes).to_vec()
    }
}

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    for i in 0..input.shape(0) {
        let mut acc: F = F::new(0.0f32);
        for j in 0..input.shape(1) {
            acc += input[i * input.stride(0) + j];
        }
        output[i] = acc;
    }
}

pub fn launch<R: Runtime, F: Float + CubeElement>(device: &R::Device) {
    let client = R::client(device);
    let input: GpuTensor<R, F> = GpuTensor::arange(vec![3, 3], &client);
    let output: GpuTensor<R, F> = GpuTensor::empty(vec![3], &client);
    unsafe {
        reduce_matrix::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            input.into_tensor_arg(1),
            output.into_tensor_arg(1),
        );
    }

    println!(
        "Executed reduction with runtime {:?} => {:?}",
        R::name(&client),
        output.read(&client)
    );
}

fn main() {
    launch::<cubecl_wgpu::WgpuRuntime, f32>(&Default::default());
}
