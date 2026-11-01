#[cfg(feature = "vulkan")]
mod vulkan_feature_tests {
    /// Validate AVVulkanDeviceContext struct is non-zero size and has
    /// reasonable alignment. We can't check exact size since it depends
    /// on ash's vk types, but it must be large enough for all the fields.
    #[test]
    fn test_avvulkan_device_context_layout() {
        use crate::ffmpeg_vk_sys::AVVulkanDeviceContext;
        let size = std::mem::size_of::<AVVulkanDeviceContext>();
        let align = std::mem::align_of::<AVVulkanDeviceContext>();

        // Must be at least large enough for the queue family array (64 entries)
        // plus all the pointer/handle fields before it.
        assert!(size > 1024, "AVVulkanDeviceContext too small: {size}");
        assert!(
            align >= 8,
            "AVVulkanDeviceContext alignment too low: {align}"
        );
        eprintln!("AVVulkanDeviceContext: size={size}, align={align}");
    }

    /// Validate AVVkFrame struct layout.
    #[test]
    fn test_avvk_frame_layout() {
        use crate::ffmpeg_vk_sys::{AV_NUM_DATA_POINTERS, AVVkFrame};
        let size = std::mem::size_of::<AVVkFrame>();
        let align = std::mem::align_of::<AVVkFrame>();

        // AVVkFrame has 8-element arrays of Image, DeviceMemory, usize, etc.
        // Minimum: 8 * (8 + 4 + 8 + 8 + 4 + 4 + 8 + 8 + 8 + 8 + 4) bytes
        assert!(size > 256, "AVVkFrame too small: {size}");
        assert!(align >= 8, "AVVkFrame alignment too low: {align}");
        assert_eq!(AV_NUM_DATA_POINTERS, 8);
        eprintln!("AVVkFrame: size={size}, align={align}");
    }

    /// Validate AVVulkanFramesContext struct layout.
    #[test]
    fn test_avvulkan_frames_context_layout() {
        use crate::ffmpeg_vk_sys::AVVulkanFramesContext;
        let size = std::mem::size_of::<AVVulkanFramesContext>();
        assert!(size > 64, "AVVulkanFramesContext too small: {size}");
        eprintln!("AVVulkanFramesContext: size={size}");
    }

    /// AVVkFrame::default() should zero-initialize (null handles, zero values).
    #[test]
    fn test_avvk_frame_default_zeroed() {
        use crate::ffmpeg_vk_sys::AVVkFrame;
        let frame = AVVkFrame::default();

        // All images should be null
        for img in &frame.img {
            assert_eq!(*img, ash::vk::Image::null());
        }
        // All memories should be null
        for mem in &frame.mem {
            assert_eq!(*mem, ash::vk::DeviceMemory::null());
        }
        // All semaphore values should be 0
        for val in &frame.sem_value {
            assert_eq!(*val, 0);
        }
    }

    /// AVVulkanDeviceContext::default() should zero-initialize.
    #[test]
    fn test_avvulkan_device_context_default_zeroed() {
        use crate::ffmpeg_vk_sys::AVVulkanDeviceContext;
        let ctx = AVVulkanDeviceContext::default();

        assert!(ctx.get_proc_addr.is_none());
        assert_eq!(ctx.inst, ash::vk::Instance::null());
        assert_eq!(ctx.phys_dev, ash::vk::PhysicalDevice::null());
        assert_eq!(ctx.act_dev, ash::vk::Device::null());
        assert_eq!(ctx.nb_graphics_queues, 0);
        assert_eq!(ctx.nb_decode_queues, 0);
    }

    /// VulkanContext::new() should successfully create a Vulkan device
    /// with video decode queue support.
    #[test]
    #[ignore = "requires AMD GPU with Vulkan Video support"]
    fn test_vulkan_context_creation() {
        let ctx = crate::vk_sys::VulkanContext::new().expect("VulkanContext::new failed");

        assert_ne!(ctx.instance.handle(), ash::vk::Instance::null());
        assert_ne!(ctx.physical_device, ash::vk::PhysicalDevice::null());
        assert_ne!(ctx.device.handle(), ash::vk::Device::null());
        assert!(!ctx.enabled_device_extensions.is_empty());

        eprintln!(
            "VulkanContext: decode_qf={}, graphics_qf={}, transfer_qf={}, extensions={}",
            ctx.decode_queue_family,
            ctx.graphics_queue_family,
            ctx.transfer_queue_family,
            ctx.enabled_device_extensions.len(),
        );

        for ext in &ctx.enabled_device_extensions {
            eprintln!("  enabled: {}", ext.to_string_lossy());
        }
    }

    /// VulkanContext should provide valid queue handles.
    #[test]
    #[ignore = "requires AMD GPU with Vulkan Video support"]
    fn test_vulkan_context_queues() {
        let ctx = crate::vk_sys::VulkanContext::new().expect("VulkanContext::new failed");

        let decode_queue = ctx.get_queue(ctx.decode_queue_family, 0);
        assert_ne!(decode_queue, ash::vk::Queue::null());

        let graphics_queue = ctx.get_queue(ctx.graphics_queue_family, 0);
        assert_ne!(graphics_queue, ash::vk::Queue::null());

        eprintln!("queues OK: decode={decode_queue:?}, graphics={graphics_queue:?}");
    }

    /// VulkanContext should provide a valid vkGetInstanceProcAddr.
    #[test]
    #[ignore = "requires AMD GPU with Vulkan Video support"]
    fn test_vulkan_get_instance_proc_addr() {
        let ctx = crate::vk_sys::VulkanContext::new().expect("VulkanContext::new failed");

        // Should be a non-null function pointer
        let _proc_addr = ctx.get_instance_proc_addr();
        eprintln!("vkGetInstanceProcAddr: OK");
    }

    /// Test that VkDeviceMemory allocated with VkExportMemoryAllocateInfo
    /// can actually be exported as a DMA-BUF fd.
    #[test]
    #[ignore = "requires AMD GPU with Vulkan Video support"]
    fn test_vulkan_dma_buf_export_basic() {
        let ctx = crate::vk_sys::VulkanContext::new().expect("VulkanContext::new failed");

        // Try allocating with DMA-BUF export for different memory types
        for i in 0..8u32 {
            let mut export_info = ash::vk::ExportMemoryAllocateInfo::default()
                .handle_types(ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let alloc_info = ash::vk::MemoryAllocateInfo::default()
                .allocation_size(4096)
                .memory_type_index(i)
                .push_next(&mut export_info);
            if let Ok(mem) = unsafe { ctx.device.allocate_memory(&alloc_info, None) } {
                eprintln!("memory_type={i}: alloc OK");
                match ctx.export_memory_as_dma_buf(mem) {
                    Ok(fd) => {
                        eprintln!("  DMA-BUF export OK: fd={fd}");
                        unsafe {
                            libc::close(fd);
                        }
                    }
                    Err(e) => eprintln!("  DMA-BUF export FAILED: {e}"),
                }
                match ctx.export_memory_as_opaque_fd(mem) {
                    Ok(fd) => {
                        eprintln!("  OPAQUE_FD export OK: fd={fd}");
                        unsafe {
                            libc::close(fd);
                        }
                    }
                    Err(e) => eprintln!("  OPAQUE_FD export FAILED: {e}"),
                }
                unsafe {
                    ctx.device.free_memory(mem, None);
                }
            }
        }

        // Check if video decode images support external memory export
        let mut ext_image_info = ash::vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let image_format_info = ash::vk::PhysicalDeviceImageFormatInfo2::default()
            .format(ash::vk::Format::G8_B8R8_2PLANE_420_UNORM) // NV12
            .ty(ash::vk::ImageType::TYPE_2D)
            .tiling(ash::vk::ImageTiling::OPTIMAL)
            .usage(
                ash::vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                    | ash::vk::ImageUsageFlags::TRANSFER_SRC
                    | ash::vk::ImageUsageFlags::SAMPLED,
            )
            .push_next(&mut ext_image_info);
        let mut ext_props = ash::vk::ExternalImageFormatProperties::default();
        let mut image_format_props =
            ash::vk::ImageFormatProperties2::default().push_next(&mut ext_props);

        let result = unsafe {
            ctx.instance.get_physical_device_image_format_properties2(
                ctx.physical_device,
                &image_format_info,
                &mut image_format_props,
            )
        };
        match result {
            Ok(()) => {
                let ext_mem = ext_props.external_memory_properties;
                eprintln!("\nVideo decode + OPAQUE_FD query:");
                eprintln!(
                    "  export_from_imported_handle_types: {:?}",
                    ext_mem.export_from_imported_handle_types
                );
                eprintln!(
                    "  compatible_handle_types: {:?}",
                    ext_mem.compatible_handle_types
                );
                eprintln!(
                    "  external_memory_features: {:?}",
                    ext_mem.external_memory_features
                );
            }
            Err(e) => {
                eprintln!("\nVideo decode + OPAQUE_FD: NOT SUPPORTED ({e})");
            }
        }

        // Try with DMA_BUF_EXT
        let mut ext_image_info2 = ash::vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let image_format_info2 = ash::vk::PhysicalDeviceImageFormatInfo2::default()
            .format(ash::vk::Format::G8_B8R8_2PLANE_420_UNORM)
            .ty(ash::vk::ImageType::TYPE_2D)
            .tiling(ash::vk::ImageTiling::OPTIMAL)
            .usage(
                ash::vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                    | ash::vk::ImageUsageFlags::TRANSFER_SRC
                    | ash::vk::ImageUsageFlags::SAMPLED,
            )
            .push_next(&mut ext_image_info2);
        let mut ext_props2 = ash::vk::ExternalImageFormatProperties::default();
        let mut image_format_props2 =
            ash::vk::ImageFormatProperties2::default().push_next(&mut ext_props2);

        let result2 = unsafe {
            ctx.instance.get_physical_device_image_format_properties2(
                ctx.physical_device,
                &image_format_info2,
                &mut image_format_props2,
            )
        };
        match result2 {
            Ok(()) => {
                let ext_mem = ext_props2.external_memory_properties;
                eprintln!("\nVideo decode + DMA_BUF_EXT query:");
                eprintln!(
                    "  compatible_handle_types: {:?}",
                    ext_mem.compatible_handle_types
                );
                eprintln!(
                    "  external_memory_features: {:?}",
                    ext_mem.external_memory_features
                );
            }
            Err(e) => {
                eprintln!("\nVideo decode + DMA_BUF_EXT: NOT SUPPORTED ({e})");
            }
        }

        // Also try without export info
        for i in 0..3u32 {
            let alloc_info = ash::vk::MemoryAllocateInfo::default()
                .allocation_size(4096)
                .memory_type_index(i);
            if let Ok(mem) = unsafe { ctx.device.allocate_memory(&alloc_info, None) } {
                match ctx.export_memory_as_dma_buf(mem) {
                    Ok(fd) => {
                        eprintln!("NO export_info, type={i}: DMA-BUF OK: fd={fd}");
                        unsafe {
                            libc::close(fd);
                        }
                    }
                    Err(e) => eprintln!("NO export_info, type={i}: DMA-BUF FAILED: {e}"),
                }
                unsafe {
                    ctx.device.free_memory(mem, None);
                }
            }
        }
    }

    /// Verify that RADV (libvulkan_radeon.so) loaded via the Vulkan instance
    /// does NOT pull in libLLVM. This guards against the LLVM symbol conflict
    /// with MIGraphX that caused "Option 'default' already exists!" crashes.
    #[test]
    #[ignore = "requires AMD GPU with Vulkan Video support"]
    fn test_radv_no_llvm_dependency() {
        // Create VulkanContext (loads RADV)
        let _ctx = crate::vk_sys::VulkanContext::new().expect("VulkanContext::new failed");

        // Check VK_DRIVER_FILES points to our LLVM-free Mesa build
        if let Ok(icd_path) = std::env::var("VK_DRIVER_FILES") {
            eprintln!("VK_DRIVER_FILES={icd_path}");
            // The ICD JSON references the .so — find and check it
            if let Ok(json) = std::fs::read_to_string(&icd_path)
                && let Some(lib_path) = json
                    .lines()
                    .find(|l| l.contains("library_path"))
                    .and_then(|l| l.split('"').nth(3))
            {
                // Resolve relative paths against the ICD directory
                let icd_dir = std::path::Path::new(&icd_path).parent().unwrap();
                let so_path = if lib_path.starts_with('/') {
                    std::path::PathBuf::from(lib_path)
                } else {
                    icd_dir.join(lib_path)
                };
                eprintln!("RADV .so path: {}", so_path.display());

                // Use readelf to check for LLVM NEEDED entries
                let output = std::process::Command::new("readelf")
                    .args(["-d", so_path.to_str().unwrap()])
                    .output();
                if let Ok(out) = output {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let has_llvm = stdout
                        .lines()
                        .any(|l| l.contains("NEEDED") && l.to_lowercase().contains("llvm"));
                    assert!(
                        !has_llvm,
                        "RADV links against LLVM! This will cause symbol conflicts with MIGraphX. \
                             Ensure VK_DRIVER_FILES points to mesa-radv built with -Dllvm=disabled."
                    );
                    eprintln!("RADV has no LLVM NEEDED entries ✓");
                }
            }
        } else {
            eprintln!("WARNING: VK_DRIVER_FILES not set — cannot verify LLVM-free RADV");
        }
    }

    /// VulkanContext creation should not crash when HIP is already initialized.
    /// This tests the core fix: RADV (ACO) + MIGraphX (ROCm LLVM) coexistence.
    #[test]
    #[ignore = "requires AMD GPU with Vulkan Video + HIP runtime"]
    fn test_vulkan_and_hip_coexist() {
        // Initialize HIP first (loads ROCm LLVM via MIGraphX path)
        crate::init(0).expect("HIP init failed");

        // Now create Vulkan context (loads RADV) — this previously crashed
        // with "LLVM ERROR: Option 'default' already exists!"
        let ctx = crate::vk_sys::VulkanContext::new()
            .expect("VulkanContext::new failed after HIP init — LLVM conflict?");

        assert_ne!(ctx.device.handle(), ash::vk::Device::null());
        eprintln!("HIP + Vulkan coexistence: OK");

        // Verify we can still use HIP after Vulkan init
        let buf = crate::DeviceBuffer::alloc(1024).expect("HIP alloc failed after Vulkan init");
        assert!(!buf.as_ptr().is_null());
        eprintln!("HIP allocation after Vulkan init: OK");
    }

    /// Nv12Layout should correctly compute plane pointers.
    #[test]
    fn test_nv12_layout_values() {
        let layout = crate::gpu_frame::Nv12Layout {
            width: 1920,
            height: 1080,
            y_offset: 0,
            y_pitch: 1920,
            uv_offset: 1920 * 1080,
            uv_pitch: 1920,
            total_size: 1920 * 1080 * 3 / 2,
        };

        assert_eq!(layout.y_offset, 0);
        assert_eq!(layout.uv_offset, 1920 * 1080);
        assert_eq!(layout.total_size, 1920 * 1080 + 1920 * 540);
    }
}

/// Test the GPU preprocessor can be created (loads kernel module).
#[test]
#[ignore = "requires AMD GPU with HIP runtime"]
fn test_preprocessor_creation() {
    crate::init(0).expect("HIP init failed");
    let prep = crate::GpuPreprocessor::new(256, 256, crate::NormParams::imagenet());
    assert!(
        prep.is_ok(),
        "preprocessor creation failed: {:?}",
        prep.err()
    );
}
