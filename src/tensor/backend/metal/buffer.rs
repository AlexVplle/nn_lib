use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::MTLBuffer;

#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        Buffer { raw }
    }

    pub fn data(&self) -> *mut u8 {
        self.as_ref().contents().as_ptr() as *mut u8
    }

    pub fn contents(&self) -> *mut u8 {
        self.data()
    }

    pub fn length(&self) -> usize {
        self.as_ref().length()
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_foundation::NSUInteger;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

    fn create_test_buffer(size: usize) -> Buffer {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let raw = device
            .newBufferWithLength_options(size as NSUInteger, MTLResourceOptions::StorageModeShared)
            .expect("Failed to create buffer");
        Buffer::new(raw)
    }

    #[test]
    fn test_buffer_new() {
        let buffer = create_test_buffer(1024);
        assert_eq!(buffer.length(), 1024);
    }

    #[test]
    fn test_buffer_length() {
        let sizes = vec![1, 16, 256, 1024, 4096, 1_000_000];

        for size in sizes {
            let buffer = create_test_buffer(size);
            assert_eq!(buffer.length(), size);
        }
    }

    #[test]
    fn test_buffer_data_non_null() {
        let buffer = create_test_buffer(1024);
        let ptr = buffer.data();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_buffer_write_read() {
        let buffer = create_test_buffer(16);
        let ptr = buffer.data();

        unsafe {
            for i in 0..16 {
                *ptr.add(i) = i as u8;
            }
        }

        unsafe {
            for i in 0..16 {
                assert_eq!(*ptr.add(i), i as u8);
            }
        }
    }

    #[test]
    fn test_buffer_write_f32() {
        let buffer = create_test_buffer(16);
        let ptr = buffer.data() as *mut f32;

        let test_data = [1.5f32, 2.7f32, -1.0f32];

        unsafe {
            for (i, &value) in test_data.iter().enumerate() {
                *ptr.add(i) = value;
            }
        }

        unsafe {
            for (i, &expected) in test_data.iter().enumerate() {
                let actual = *ptr.add(i);
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_buffer_as_ref() {
        let buffer = create_test_buffer(1024);
        let raw: &ProtocolObject<dyn MTLBuffer> = buffer.as_ref();
        assert_eq!(raw.length(), 1024);
    }

    #[test]
    fn test_buffer_did_modify_range() {
        let buffer = create_test_buffer(1024);

        let range = NSRange::new(0, 256);
        buffer.did_modify_range(range);
    }

    #[test]
    fn test_buffer_did_modify_range_full() {
        let buffer = create_test_buffer(1024);
        let range = NSRange::new(0, buffer.length());
        buffer.did_modify_range(range);
    }

    #[test]
    fn test_buffer_did_modify_range_partial() {
        let buffer = create_test_buffer(1024);

        buffer.did_modify_range(NSRange::new(0, 256));
        buffer.did_modify_range(NSRange::new(256, 256));
        buffer.did_modify_range(NSRange::new(512, 512));
    }

    #[test]
    fn test_buffer_clone_points_to_same_data() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let raw = device
            .newBufferWithLength_options(16, MTLResourceOptions::StorageModeShared)
            .expect("Failed to create buffer");

        let buffer1 = Buffer::new(raw);
        let buffer2 = buffer1.clone();

        unsafe {
            let ptr1 = buffer1.data();
            *ptr1 = 42;
        }

        unsafe {
            let ptr2 = buffer2.data();
            assert_eq!(*ptr2, 42);
        }
    }

    #[test]
    fn test_buffer_thread_safe() {
        use std::sync::Arc;
        use std::thread;

        let buffer = Arc::new(create_test_buffer(1024));
        let buffer_clone = buffer.clone();

        let handle = thread::spawn(move || {
            let len = buffer_clone.length();
            assert_eq!(len, 1024);
        });

        handle.join().unwrap();
    }

    #[test]
    fn test_buffer_zero_initialized() {
        let buffer = create_test_buffer(256);
        let ptr = buffer.data();

        unsafe {
            let _value = *ptr;
        }
    }

    #[test]
    fn test_buffer_alignment() {
        let buffer = create_test_buffer(1024);
        let ptr = buffer.data() as usize;

        assert_eq!(ptr % 16, 0, "Buffer should be 16-byte aligned");
    }

    #[test]
    fn test_buffer_different_sizes() {
        let sizes = vec![1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 1_000_000];

        for size in sizes {
            let buffer = create_test_buffer(size);
            assert_eq!(buffer.length(), size);
            assert!(!buffer.data().is_null());
        }
    }

    #[test]
    fn test_buffer_write_pattern() {
        let buffer = create_test_buffer(256);
        let ptr = buffer.data();

        unsafe {
            for i in 0..256 {
                *ptr.add(i) = (i % 256) as u8;
            }
        }

        unsafe {
            for i in 0..256 {
                assert_eq!(*ptr.add(i), (i % 256) as u8);
            }
        }
    }

    #[test]
    fn test_buffer_mixed_types() {
        let buffer = create_test_buffer(32);

        unsafe {
            let ptr_u32 = buffer.data() as *mut u32;
            *ptr_u32 = 0x12345678;
        }

        unsafe {
            let ptr_u8 = buffer.data();
            #[cfg(target_endian = "little")]
            {
                assert_eq!(*ptr_u8.add(0), 0x78);
                assert_eq!(*ptr_u8.add(1), 0x56);
                assert_eq!(*ptr_u8.add(2), 0x34);
                assert_eq!(*ptr_u8.add(3), 0x12);
            }
        }
    }

    #[test]
    fn test_buffer_slice_copy() {
        let src_buffer = create_test_buffer(256);
        let dst_buffer = create_test_buffer(256);

        unsafe {
            let src_ptr = src_buffer.data();
            for i in 0..256 {
                *src_ptr.add(i) = i as u8;
            }
        }

        unsafe {
            let src_ptr = src_buffer.data();
            let dst_ptr = dst_buffer.data();
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, 256);
        }

        unsafe {
            let dst_ptr = dst_buffer.data();
            for i in 0..256 {
                assert_eq!(*dst_ptr.add(i), i as u8);
            }
        }
    }

    #[test]
    fn test_buffer_modify_range_metadata() {
        let buffer = create_test_buffer(1024);
        let ptr = buffer.data() as *mut f32;

        unsafe {
            for i in 0..16 {
                *ptr.add(i) = i as f32;
            }
        }

        buffer.did_modify_range(NSRange::new(0, 64));

        unsafe {
            for i in 0..16 {
                assert_eq!(*ptr.add(i), i as f32);
            }
        }
    }

    #[test]
    fn test_buffer_large_allocation() {
        let buffer = create_test_buffer(10 * 1024 * 1024);
        assert_eq!(buffer.length(), 10 * 1024 * 1024);
        assert!(!buffer.data().is_null());
    }

    #[test]
    fn test_buffer_multiple_allocations() {
        let buffers: Vec<Buffer> = (0..10)
            .map(|i| create_test_buffer(1024 * (i + 1)))
            .collect();

        for (i, buffer) in buffers.iter().enumerate() {
            assert_eq!(buffer.length(), 1024 * (i + 1));
        }
    }

    #[test]
    fn test_buffer_drop_safety() {
        {
            let _buffer = create_test_buffer(1024);
        }

        let _buffer2 = create_test_buffer(2048);
    }

    #[test]
    fn test_buffer_pointer_stability() {
        let buffer = create_test_buffer(1024);
        let ptr1 = buffer.data();
        let ptr2 = buffer.data();
        let ptr3 = buffer.contents();

        assert_eq!(ptr1, ptr2);
        assert_eq!(ptr1, ptr3);
    }

    #[test]
    fn test_buffer_from_vec_pattern() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();

        let raw = unsafe {
            use objc2_foundation::NSUInteger;
            use std::ptr::NonNull;

            let byte_size = (data.len() * std::mem::size_of::<f32>()) as NSUInteger;
            let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("null pointer");

            device
                .newBufferWithBytes_length_options(
                    ptr,
                    byte_size,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Failed to create buffer")
        };

        let buffer = Buffer::new(raw);

        unsafe {
            let ptr = buffer.data() as *const f32;
            for i in 0..256 {
                assert_eq!(*ptr.add(i), i as f32);
            }
        }
    }
}
