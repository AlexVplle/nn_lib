use crate::tensor::storage::cpu_storage::CpuStorage;

#[derive(PartialEq, Debug, Clone, PartialOrd)]
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
}
