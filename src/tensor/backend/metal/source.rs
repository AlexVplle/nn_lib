pub const ADD: &str = include_str!("kernels/add.metal");
pub const MATMUL: &str = include_str!("kernels/matmul.metal");
pub const MATMUL_STRIDED: &str = include_str!("kernels/matmul_strided.metal");
pub const MUL: &str = include_str!("kernels/mul.metal");
pub const RELU: &str = include_str!("kernels/relu.metal");
pub const TANH: &str = include_str!("kernels/tanh.metal");
pub const SIGMOID: &str = include_str!("kernels/sigmoid.metal");
pub const SOFTMAX: &str = include_str!("kernels/softmax.metal");
pub const SUB: &str = include_str!("kernels/sub.metal");
pub const MUL_SCALAR: &str = include_str!("kernels/mul_scalar.metal");
pub const SUM_AXIS: &str = include_str!("kernels/sum_axis.metal");
pub const COPY_STRIDED: &str = include_str!("kernels/copy_strided.metal");

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub enum Source {
    Add,
    Matmul,
    MatmulStrided,
    Mul,
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
    Sub,
    MulScalar,
    SumAxis,
    CopyStrided,
}

impl Source {
    pub fn code(&self) -> &'static str {
        match self {
            Source::Add => ADD,
            Source::Matmul => MATMUL,
            Source::MatmulStrided => MATMUL_STRIDED,
            Source::Mul => MUL,
            Source::ReLU => RELU,
            Source::Tanh => TANH,
            Source::Sigmoid => SIGMOID,
            Source::Softmax => SOFTMAX,
            Source::Sub => SUB,
            Source::MulScalar => MUL_SCALAR,
            Source::SumAxis => SUM_AXIS,
            Source::CopyStrided => COPY_STRIDED,
        }
    }
}
