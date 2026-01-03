use cubecl::{
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future::block_on,
    prelude::{CubeElement, TensorHandleRef},
    server::Handle,
    std::tensor::compact_strides,
    Runtime,
};
use cubecl_reduce::{instructions::Sum, reduce, ReducePrecision};

pub struct ReductionBench<R: Runtime> {
    input_shape: Vec<usize>,
    input_stride: Vec<usize>,
    output_shape: Vec<usize>,
    output_stride: Vec<usize>,
    axis: usize,
    client: ComputeClient<R::Server>,
}

impl<R: Runtime> Benchmark for ReductionBench<R> {
    type Input = Handle;
    type Output = Handle;

    fn prepare(&self) -> Self::Input {
        let size: usize = self.input_shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        self.client.create(f32::as_bytes(&data))
    }

    fn name(&self) -> String {
        format!("{}-reduction-{:?}", R::name(&self.client), self.input_shape).to_lowercase()
    }

    fn sync(&self) {
        block_on(self.client.sync())
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let output_values: Vec<f32> = vec![0.0f32; self.output_shape.iter().product()];
        let output: Handle = self.client.create(f32::as_bytes(&output_values));

        let input_ref = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input,
                &self.input_stride,
                &self.input_shape,
                core::mem::size_of::<f32>(),
            )
        };

        let output_ref = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &output,
                &self.output_stride,
                &self.output_shape,
                core::mem::size_of::<f32>(),
            )
        };

        reduce::<R, DefaultPrecision, f32, Sum>(
            &self.client,
            input_ref,
            output_ref,
            self.axis,
            None,
            (),
        )
        .map_err(|e| format!("{:?}", e))?;

        Ok(output)
    }
}

struct DefaultPrecision;

impl ReducePrecision for DefaultPrecision {
    type EI = f32;
    type EA = f32;
}

pub fn verify_correctness(device: &<cubecl_wgpu::WgpuRuntime as Runtime>::Device) {
    let client = cubecl_wgpu::WgpuRuntime::client(device);

    let input_shape = vec![4, 16];
    let axis = 1;
    let input_stride = compact_strides(&input_shape);

    let mut output_shape = input_shape.clone();
    output_shape[axis] = 1;
    let output_stride = compact_strides(&output_shape);

    let bench = ReductionBench::<cubecl_wgpu::WgpuRuntime> {
        input_shape: input_shape.clone(),
        input_stride,
        output_shape: output_shape.clone(),
        output_stride,
        axis,
        client: client.clone(),
    };

    println!("\n=== Vérification de la justesse ===");
    println!("Input shape: {:?}", bench.input_shape);

    let input_handle = bench.prepare();
    let input_bytes = client.read_one(input_handle.clone());
    let input_data = f32::from_bytes(&input_bytes);

    println!("\nInput data:");
    for row in 0..bench.input_shape[0] {
        print!("Row {}: [", row);
        for col in 0..bench.input_shape[1] {
            print!("{:4.0} ", input_data[row * bench.input_shape[1] + col]);
        }
        println!("]");
    }

    let mut expected = vec![0.0f32; bench.input_shape[0]];
    for row in 0..bench.input_shape[0] {
        let mut sum = 0.0f32;
        for col in 0..bench.input_shape[1] {
            sum += input_data[row * bench.input_shape[1] + col];
        }
        expected[row] = sum;
    }

    println!("\nExpected output (CPU):");
    for (i, val) in expected.iter().enumerate() {
        println!("Row {}: {}", i, val);
    }

    let output_handle = bench.execute(input_handle).unwrap();
    let output_bytes = client.read_one(output_handle);
    let result = f32::from_bytes(&output_bytes);

    println!("\nActual output (GPU):");
    for (i, val) in result.iter().enumerate() {
        println!("Row {}: {}", i, val);
    }

    println!("\n=== Comparaison ===");
    let mut all_match = true;
    for i in 0..bench.input_shape[0] {
        let diff = (result[i] - expected[i]).abs();
        let matches = diff < 0.001;
        println!(
            "Row {}: Expected={:.2}, Actual={:.2}, Diff={:.6}, Match={}",
            i, expected[i], result[i], diff, matches
        );
        if !matches {
            all_match = false;
        }
    }

    if all_match {
        println!("\n✓ Tous les résultats sont corrects!");
    } else {
        println!("\n✗ ERREUR: Les résultats ne correspondent pas!");
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    println!("\n=== Benchmarks ===");

    for (rows, cols) in [(512, 8 * 1024), (128, 32 * 1024)] {
        let input_shape = vec![rows, cols];
        let axis = 1;
        let input_stride = compact_strides(&input_shape);

        let mut output_shape = input_shape.clone();
        output_shape[axis] = 1;
        let output_stride = compact_strides(&output_shape);

        let bench = ReductionBench::<R> {
            input_shape,
            input_stride,
            output_shape,
            output_stride,
            axis,
            client: client.clone(),
        };

        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::System).unwrap());
    }
}

fn main() {
    let device = Default::default();
    verify_correctness(&device);
    launch::<cubecl_wgpu::WgpuRuntime>(&device);
}
