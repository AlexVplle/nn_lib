#[derive(Debug, PartialEq)]
pub enum Device {
    CPU,
    CUDA(usize),
    Metal(usize),
}
