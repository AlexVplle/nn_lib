#[derive(Debug)]
pub enum Device {
    CPU,
    CUDA(usize),
    Metal(usize),
}
