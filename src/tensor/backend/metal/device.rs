use std::{ffi::c_void, ptr};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};

use crate::{
    error::NeuralNetworkError,
    tensor::backend::metal::{
        buffer::Buffer, compute_pipeline::ComputePipeline, function::Function, library::Library,
    },
};

pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl AsRef<ProtocolObject<dyn MTLDevice>> for Device {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }
}

impl Device {
    pub fn registry_id(&self) -> u64 {
        self.as_ref().registryID()
    }

    pub fn all() -> Vec<Self> {
        MTLCreateSystemDefaultDevice()
            .into_iter()
            .map(|raw| Device { raw })
            .collect()
    }

    pub fn system_default() -> Option<Self> {
        MTLCreateSystemDefaultDevice().map(|raw| Device { raw })
    }

    pub fn new_buffer(
        &self,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, crate::error::NeuralNetworkError> {
        self.as_ref()
            .newBufferWithLength_options(length, options)
            .map(Buffer::new)
            .ok_or(NeuralNetworkError::AllocationFailed(
                "Metal buffer creation failed".to_string(),
            ))
    }

    pub fn new_buffer_with_data(
        &self,
        pointer: *const c_void,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, NeuralNetworkError> {
        let pointer = ptr::NonNull::new(pointer as *mut c_void).unwrap();
        unsafe {
            self.as_ref()
                .newBufferWithBytes_length_options(pointer, length, options)
                .map(Buffer::new)
                .ok_or(NeuralNetworkError::AllocationFailed(
                    "Metal buffer creation failed".to_string(),
                ))
        }
    }

    pub fn new_library_with_source(
        &self,
        source: &str,
        options: Option<&MTLCompileOptions>,
    ) -> Result<Library, NeuralNetworkError> {
        let raw = self
            .as_ref()
            .newLibraryWithSource_options_error(&NSString::from_str(source), options)
            .unwrap();
        Ok(Library::new(raw))
    }

    pub fn new_compute_pipeline_state_with_function(
        &self,
        function: &Function,
    ) -> Result<ComputePipeline, NeuralNetworkError> {
        let raw = self
            .as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .unwrap();
        Ok(ComputePipeline::new(raw))
    }

    pub fn new_command_queue(&self) -> Result<CommandQueue, NeuralNetworkError> {
        let raw = self.as_ref().newCommandQueue().unwrap();
        Ok(raw)
    }

    pub fn recommended_max_working_set_size(&self) -> usize {
        self.as_ref().recommendedMaxWorkingSetSize() as usize
    }

    pub fn current_allocated_size(&self) -> usize {
        self.as_ref().currentAllocatedSize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SHADER: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void simple_kernel(device float* output [[buffer(0)]]) {
            output[0] = 1.0;
        }
        
        constant float MULTIPLIER [[function_constant(0)]];
        
        kernel void kernel_with_constants(device float* output [[buffer(0)]]) {
            output[0] = MULTIPLIER;
        }
    "#;

    #[test]
    fn test_device_system_default() {
        let device = Device::system_default();
        assert!(device.is_some());
    }

    #[test]
    fn test_device_all() {
        let devices = Device::all();
        assert!(!devices.is_empty());
    }

    #[test]
    fn test_device_registry_id() {
        let device = Device::system_default().unwrap();
        let id = device.registry_id();
        assert!(id > 0);
    }

    #[test]
    fn test_device_registry_id_consistent() {
        let device = Device::system_default().unwrap();
        let id1 = device.registry_id();
        let id2 = device.registry_id();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_device_new_buffer() {
        let device = Device::system_default().unwrap();
        let buffer = device.new_buffer(1024, MTLResourceOptions::StorageModeShared);

        assert!(buffer.is_ok());
        assert_eq!(buffer.unwrap().length(), 1024);
    }

    #[test]
    fn test_device_new_buffer_different_sizes() {
        let device = Device::system_default().unwrap();
        let sizes = vec![16, 256, 1024, 4096, 1_000_000];

        for size in sizes {
            let buffer = device.new_buffer(size, MTLResourceOptions::StorageModeShared);
            assert!(buffer.is_ok());
            assert_eq!(buffer.unwrap().length(), size);
        }
    }

    #[test]
    fn test_device_new_buffer_with_data() {
        let device = Device::system_default().unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let byte_size = data.len() * std::mem::size_of::<f32>();

        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );

        assert!(buffer.is_ok());
        let buffer = buffer.unwrap();
        assert_eq!(buffer.length(), byte_size);

        unsafe {
            let ptr = buffer.contents() as *const f32;
            for (i, &expected) in data.iter().enumerate() {
                assert_eq!(*ptr.add(i), expected);
            }
        }
    }

    #[test]
    fn test_device_new_library_with_source() {
        let device = Device::system_default().unwrap();
        let library = device.new_library_with_source(TEST_SHADER, None);

        assert!(library.is_ok());
    }

    #[test]
    fn test_device_new_library_with_options() {
        let device = Device::system_default().unwrap();
        let options = MTLCompileOptions::new();
        let library = device.new_library_with_source(TEST_SHADER, Some(&options));

        assert!(library.is_ok());
    }

    #[test]
    fn test_device_get_function_from_library() {
        let device = Device::system_default().unwrap();
        let library = device.new_library_with_source(TEST_SHADER, None).unwrap();

        let function = library.get_function("simple_kernel", None);
        assert!(function.is_ok());
    }

    #[test]
    fn test_device_get_function_not_found() {
        let device = Device::system_default().unwrap();
        let library = device.new_library_with_source(TEST_SHADER, None).unwrap();

        let function = library.get_function("nonexistent_kernel", None);
        assert!(function.is_err());
    }

    #[test]
    fn test_device_new_compute_pipeline_state() {
        let device = Device::system_default().unwrap();
        let library = device.new_library_with_source(TEST_SHADER, None).unwrap();
        let function = library.get_function("simple_kernel", None).unwrap();

        let pipeline = device.new_compute_pipeline_state_with_function(&function);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_device_compute_pipeline_max_threads() {
        let device = Device::system_default().unwrap();
        let library = device.new_library_with_source(TEST_SHADER, None).unwrap();
        let function = library.get_function("simple_kernel", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();

        let max_threads = pipeline.max_total_threads_per_threadgroup();
        assert!(max_threads > 0);
        assert!(max_threads >= 256);
    }

    #[test]
    fn test_device_new_command_queue() {
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();

        assert!(queue.is_ok());
    }

    #[test]
    fn test_device_multiple_command_queues() {
        let device = Device::system_default().unwrap();

        let queue1 = device.new_command_queue();
        let queue2 = device.new_command_queue();

        assert!(queue1.is_ok());
        assert!(queue2.is_ok());
    }

    #[test]
    fn test_device_recommended_max_working_set_size() {
        let device = Device::system_default().unwrap();
        let size = device.recommended_max_working_set_size();

        assert!(size > 0);
    }

    #[test]
    fn test_device_current_allocated_size() {
        let device = Device::system_default().unwrap();
        let size_before = device.current_allocated_size();

        let _buffer = device
            .new_buffer(1024 * 1024, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let size_after = device.current_allocated_size();
        assert!(size_after >= size_before);
    }

    #[test]
    fn test_device_thread_safe() {
        use std::sync::Arc;
        use std::thread;

        let device = Arc::new(Device::system_default().unwrap());
        let device_clone = device.clone();

        let handle = thread::spawn(move || {
            let buffer = device_clone.new_buffer(1024, MTLResourceOptions::StorageModeShared);
            assert!(buffer.is_ok());
        });

        handle.join().unwrap();
    }

    #[test]
    fn test_device_create_multiple_buffers() {
        let device = Device::system_default().unwrap();

        let buffers: Vec<_> = (0..10)
            .map(|i| device.new_buffer(1024 * (i + 1), MTLResourceOptions::StorageModeShared))
            .collect();

        for (i, buffer) in buffers.iter().enumerate() {
            assert!(buffer.is_ok());
            assert_eq!(buffer.as_ref().unwrap().length(), 1024 * (i + 1));
        }
    }

    #[test]
    fn test_device_compile_multiple_libraries() {
        let device = Device::system_default().unwrap();

        let lib1 = device.new_library_with_source(TEST_SHADER, None);
        let lib2 = device.new_library_with_source(TEST_SHADER, None);

        assert!(lib1.is_ok());
        assert!(lib2.is_ok());
    }

    #[test]
    fn test_device_full_pipeline_workflow() {
        let device = Device::system_default().unwrap();

        let library = device.new_library_with_source(TEST_SHADER, None).unwrap();
        let function = library.get_function("simple_kernel", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        let buffer = device
            .new_buffer(16, MTLResourceOptions::StorageModeShared)
            .unwrap();

        assert!(pipeline.max_total_threads_per_threadgroup() > 0);
        assert_eq!(buffer.length(), 16);
    }

    #[test]
    fn test_device_buffer_with_data_preserves_content() {
        let device = Device::system_default().unwrap();

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let byte_size = data.len() * std::mem::size_of::<f32>();

        let buffer = device
            .new_buffer_with_data(
                data.as_ptr() as *const c_void,
                byte_size,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();

        unsafe {
            let ptr = buffer.contents() as *const f32;
            for i in 0..100 {
                assert_eq!(*ptr.add(i), i as f32);
            }
        }
    }

    #[test]
    fn test_device_as_ref() {
        let device = Device::system_default().unwrap();
        let raw: &ProtocolObject<dyn MTLDevice> = device.as_ref();

        assert_eq!(raw.registryID(), device.registry_id());
    }

    #[test]
    fn test_device_allocated_size_increases_with_buffers() {
        let device = Device::system_default().unwrap();
        let initial_size = device.current_allocated_size();

        let _buffers: Vec<_> = (0..10)
            .map(|_| {
                device
                    .new_buffer(1024 * 1024, MTLResourceOptions::StorageModeShared)
                    .unwrap()
            })
            .collect();

        let final_size = device.current_allocated_size();
        assert!(final_size > initial_size);
    }
}
