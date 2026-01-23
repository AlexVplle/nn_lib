pub const ADD: &str = include_str!("kernels/add.metal");
pub const MATMUL: &str = include_str!("kernels/matmul.metal");
pub const MUL: &str = include_str!("kernels/mul.metal");
pub const RELU: &str = include_str!("kernels/relu.metal");
pub const TANH: &str = include_str!("kernels/tanh.metal");
pub const SIGMOID: &str = include_str!("kernels/sigmoid.metal");
pub const SOFTMAX: &str = include_str!("kernels/softmax.metal");
pub const SUB: &str = include_str!("kernels/sub.metal");
pub const MUL_SCALAR: &str = include_str!("kernels/mul_scalar.metal");

#[derive(Debug)]
pub enum Source {
    Add,
    Matmul,
    Mul,
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
    Sub,
    MulScalar,
}
