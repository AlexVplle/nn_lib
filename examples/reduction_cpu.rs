/// Example of a naive multidimensional tensor in pure Rust
#[derive(Debug, Clone)]
pub struct CpuTensor {
    /// Raw contiguous value buffer
    pub data: Vec<f32>,
    /// How many element are between each dimensions
    pub strides: Vec<usize>,
    /// Dimension of the tensor
    pub shape: Vec<usize>,
}

/// Function to compute strides in a compact layout
fn compact_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl CpuTensor {
    /// Create a CpuTensor with a shape filled by number in order
    pub fn arange(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = (0..size).map(|i| i as f32).collect();
        let strides = compact_strides(&shape);
        Self {
            data,
            strides,
            shape,
        }
    }

    /// Create an empty CpuTensor with a shape
    pub fn empty(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = vec![0.0; size];
        let strides = compact_strides(&shape);
        Self {
            data,
            strides,
            shape,
        }
    }

    /// Read the inner data
    pub fn read(self) -> Vec<f32> {
        self.data
    }
}

fn reduce_matrix(input: &CpuTensor, output: &mut CpuTensor) {
    for i in 0..input.shape[0] {
        let mut acc = 0.0f32;
        for j in 0..input.shape[1] {
            acc += input.data[i * input.strides[0] + j];
        }
        output.data[i] = acc;
    }
}

fn launch() {
    let input_shape = vec![3, 3];
    let output_shape = vec![3];
    let input = CpuTensor::arange(input_shape);
    let mut output = CpuTensor::empty(output_shape);

    reduce_matrix(&input, &mut output);

    println!("Executed reduction => {:?}", output.read());
}

fn main() {
    launch();
}
