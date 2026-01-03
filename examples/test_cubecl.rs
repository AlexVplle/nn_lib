use cubecl::{prelude::*, server::Handle};
use cubecl_wgpu::{WgpuDevice, WgpuServer};

#[cube(launch_unchecked)]
fn add_arrays<F: Float>(a: &Array<Line<F>>, b: &Array<Line<F>>, c: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() {
        c[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

fn main() {
    type Runtime = cubecl_wgpu::WgpuRuntime;
    let device: WgpuDevice = Default::default();
    let client: ComputeClient<WgpuServer> = Runtime::client(&device);

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let vectorization: u8 = 4;

    let a_handle: Handle = client.create(f32::as_bytes(&a));
    let b_handle: Handle = client.create(f32::as_bytes(&b));
    let c_handle: Handle = client.empty(a.len() * core::mem::size_of::<f32>());

    unsafe {
        add_arrays::launch_unchecked::<f32, Runtime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<Line<f32>>(&a_handle, a.len(), vectorization),
            ArrayArg::from_raw_parts::<Line<f32>>(&b_handle, b.len(), vectorization),
            ArrayArg::from_raw_parts::<Line<f32>>(&c_handle, a.len(), vectorization),
        );
    }

    let bytes = client.read_one(c_handle);
    let result: &[f32] = f32::from_bytes(&bytes);
    println!("Result: {:?}", result);
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    println!("CubeCL is working!");
}
