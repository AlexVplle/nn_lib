pub struct CudaBackend;

impl Backend for CudaBackend {
    fn add(&self, lhs: &Arc<RwLock<Box<dyn StorageBackend>>>, rhs: &Arc<RwLock<Box<dyn StorageBackend>>>) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
        
    }

    fn relu(&self, storage: &Arc<RwLock<Box<dyn StorageBackend>>>) -> Result<Box<dyn StorageBackend>, NeuralNetworkError> {
    }

    fn device(&self) -> Device {
        Device::Cuda
    }
}