//! Vulkan context for hardware video decoding with DMA-BUF export.
//!
//! Creates a Vulkan instance and device with the extensions needed for
//! FFmpeg's Vulkan hwaccel and DMA-BUF export to HIP. Uses RADV (Mesa's
//! Vulkan driver for AMD), which uses the ACO shader compiler instead of
//! LLVM, avoiding the fatal LLVM symbol conflict with MIGraphX.

#![allow(non_snake_case)]

use std::ffi::{CStr, CString};
use std::os::raw::c_int;

use ash::vk;

use crate::error::{GpuError, Result};

/// Required device extensions for video decode + DMA-BUF export.
const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    // Video decode
    ash::khr::video_queue::NAME,
    ash::khr::video_decode_queue::NAME,
    // External memory for DMA-BUF export
    ash::khr::external_memory::NAME,
    ash::khr::external_memory_fd::NAME,
    ash::ext::external_memory_dma_buf::NAME,
    // Required by FFmpeg's Vulkan hwaccel
    ash::khr::synchronization2::NAME,
    // Timeline semaphores (core in Vulkan 1.2, but FFmpeg may need the extension)
    ash::khr::timeline_semaphore::NAME,
];

/// Optional device extensions for specific codec support.
const OPTIONAL_DEVICE_EXTENSIONS: &[&CStr] = &[
    ash::khr::video_decode_h264::NAME,
    ash::khr::video_decode_h265::NAME,
    ash::khr::video_decode_av1::NAME,
    ash::khr::video_maintenance1::NAME,
];

/// Vulkan context owning instance, device, and queues for video decode.
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    /// Queue family index for video decode operations.
    pub decode_queue_family: u32,
    /// Queue family index for graphics/compute (needed by FFmpeg).
    pub graphics_queue_family: u32,
    /// Queue family index for transfer operations.
    pub transfer_queue_family: u32,
    /// Extensions that were actually enabled on the device.
    pub enabled_device_extensions: Vec<CString>,
    /// Extensions that were actually enabled on the instance.
    pub enabled_instance_extensions: Vec<CString>,
    /// The external_memory_fd function loader.
    pub external_memory_fd: ash::khr::external_memory_fd::Device,
}

impl VulkanContext {
    /// Create a Vulkan context suitable for video decode and DMA-BUF export.
    ///
    /// Selects a discrete AMD GPU with video decode queue support.
    pub fn new() -> Result<Self> {
        // SAFETY: Ash loads Vulkan entry points from the process loader and
        // returns an owning handle with no borrowed inputs.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| GpuError::Vulkan(format!("failed to load libvulkan.so: {e}"), 0))?;

        // Create instance with Vulkan 1.3 (required by FFmpeg's Vulkan hwaccel)
        let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));

        let instance_extensions: Vec<*const i8> = vec![];
        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        // SAFETY: `instance_ci` references local data that lives through the
        // call and we pass no custom allocator.
        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateInstance: {e}"), e.as_raw()))?;

        // Find a suitable physical device
        // SAFETY: `instance` is a valid live Vulkan instance.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|e| {
            GpuError::Vulkan(format!("enumerate_physical_devices: {e}"), e.as_raw())
        })?;

        let (physical_device, decode_qf, graphics_qf, transfer_qf) =
            Self::select_physical_device(&instance, &physical_devices)?;

        // Log selected device
        // SAFETY: `physical_device` was returned by Vulkan for this instance.
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        // SAFETY: Vulkan guarantees `device_name` is a null-terminated string
        // within `VkPhysicalDeviceProperties`.
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        tracing::info!(
            device = ?device_name,
            decode_queue_family = decode_qf,
            graphics_queue_family = graphics_qf,
            "selected Vulkan physical device"
        );

        // Determine which extensions to enable
        // SAFETY: `physical_device` belongs to `instance` and the output is
        // owned Rust data copied by ash.
        let available_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }.map_err(
                |e| {
                    GpuError::Vulkan(
                        format!("enumerate_device_extension_properties: {e}"),
                        e.as_raw(),
                    )
                },
            )?;

        let available_names: Vec<&CStr> = available_extensions
            .iter()
            .map(|e| {
                // SAFETY: Vulkan extension properties expose fixed-size
                // null-terminated C strings in `extension_name`.
                unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }
            })
            .collect();

        let mut enabled_extensions: Vec<CString> = Vec::new();

        // Add required extensions (fail if missing)
        for &ext in REQUIRED_DEVICE_EXTENSIONS {
            if available_names.contains(&ext) {
                enabled_extensions.push(ext.to_owned());
            } else {
                return Err(GpuError::Vulkan(
                    format!(
                        "required Vulkan extension not available: {}",
                        ext.to_str().unwrap_or("?")
                    ),
                    0,
                ));
            }
        }

        // Add optional extensions if available
        for &ext in OPTIONAL_DEVICE_EXTENSIONS {
            if available_names.contains(&ext) {
                enabled_extensions.push(ext.to_owned());
                tracing::debug!(extension = ?ext, "enabling optional Vulkan extension");
            }
        }

        let ext_ptrs: Vec<*const i8> = enabled_extensions.iter().map(|e| e.as_ptr()).collect();

        // Create queues: one for each unique queue family
        let mut unique_families = vec![decode_qf, graphics_qf, transfer_qf];
        unique_families.sort();
        unique_families.dedup();

        let queue_priorities = [1.0f32];
        let queue_cis: Vec<vk::DeviceQueueCreateInfo> = unique_families
            .iter()
            .map(|&qf| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(qf)
                    .queue_priorities(&queue_priorities)
            })
            .collect();

        // Enable Vulkan 1.3 features (includes synchronization2, timeline semaphores)
        let mut vk13_features =
            vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let mut vk12_features =
            vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut vk13_features)
            .push_next(&mut vk12_features);

        // SAFETY: Queue infos and feature structs outlive the call, and we use
        // no custom allocator.
        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateDevice: {e}"), e.as_raw()))?;

        let external_memory_fd = ash::khr::external_memory_fd::Device::new(&instance, &device);

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            decode_queue_family: decode_qf,
            graphics_queue_family: graphics_qf,
            transfer_queue_family: transfer_qf,
            enabled_device_extensions: enabled_extensions,
            enabled_instance_extensions: Vec::new(),
            external_memory_fd,
        })
    }

    /// Select a physical device with video decode support.
    /// Returns (physical_device, decode_qf, graphics_qf, transfer_qf).
    fn select_physical_device(
        instance: &ash::Instance,
        devices: &[vk::PhysicalDevice],
    ) -> Result<(vk::PhysicalDevice, u32, u32, u32)> {
        for &phys_dev in devices {
            // SAFETY: `phys_dev` originates from `enumerate_physical_devices`
            // on this instance.
            let props = unsafe { instance.get_physical_device_properties(phys_dev) };

            // Prefer discrete GPUs
            if props.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
                continue;
            }

            // SAFETY: `phys_dev` belongs to `instance`; ash copies the queue
            // family properties into owned Rust values.
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };

            let mut decode_qf = None;
            let mut graphics_qf = None;
            let mut transfer_qf = None;

            for (i, qf) in queue_families.iter().enumerate() {
                let idx = i as u32;
                if qf.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR) && decode_qf.is_none()
                {
                    decode_qf = Some(idx);
                }
                if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics_qf.is_none() {
                    graphics_qf = Some(idx);
                }
                if qf.queue_flags.contains(vk::QueueFlags::TRANSFER) && transfer_qf.is_none() {
                    transfer_qf = Some(idx);
                }
            }

            if let (Some(d), Some(g)) = (decode_qf, graphics_qf) {
                let t = transfer_qf.unwrap_or(g);
                return Ok((phys_dev, d, g, t));
            }
        }

        Err(GpuError::Vulkan(
            "no discrete GPU with video decode queue found".into(),
            0,
        ))
    }

    /// Export a VkDeviceMemory as a DMA-BUF file descriptor.
    pub fn export_memory_as_dma_buf(&self, memory: vk::DeviceMemory) -> Result<c_int> {
        let get_fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        // SAFETY: `memory` is owned by this device and `get_fd_info` names a
        // handle type supported by the created device.
        let fd = unsafe { self.external_memory_fd.get_memory_fd(&get_fd_info) }.map_err(|e| {
            GpuError::Vulkan(format!("vkGetMemoryFdKHR (DMA_BUF): {e}"), e.as_raw())
        })?;

        Ok(fd)
    }

    /// Export a VkDeviceMemory as an opaque fd.
    ///
    /// On AMD/HIP, the opaque fd path goes through the same kernel DMA-BUF
    /// import mechanism. This is an alternative when DMA-BUF export is not
    /// supported for a given memory type (e.g., video decode images).
    pub fn export_memory_as_opaque_fd(&self, memory: vk::DeviceMemory) -> Result<c_int> {
        let get_fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        // SAFETY: Same reasoning as the DMA-BUF export path above, but for the
        // opaque-fd handle flavor.
        let fd = unsafe { self.external_memory_fd.get_memory_fd(&get_fd_info) }.map_err(|e| {
            GpuError::Vulkan(format!("vkGetMemoryFdKHR (OPAQUE_FD): {e}"), e.as_raw())
        })?;

        Ok(fd)
    }

    /// Export a VkDeviceMemory as an fd, trying DMA-BUF first then opaque fd.
    pub fn export_memory_as_fd(&self, memory: vk::DeviceMemory) -> Result<c_int> {
        self.export_memory_as_dma_buf(memory)
            .or_else(|_| self.export_memory_as_opaque_fd(memory))
    }

    /// Get a queue from a queue family.
    pub fn get_queue(&self, queue_family: u32, queue_index: u32) -> vk::Queue {
        // SAFETY: Callers are expected to pass a family/index pair exposed by
        // this logical device; Vulkan returns a borrowed queue handle.
        unsafe { self.device.get_device_queue(queue_family, queue_index) }
    }

    /// Get the raw VkInstance handle (for FFmpeg).
    pub fn raw_instance(&self) -> vk::Instance {
        self.instance.handle()
    }

    /// Get the raw VkDevice handle (for FFmpeg).
    pub fn raw_device(&self) -> vk::Device {
        self.device.handle()
    }

    /// Get the vkGetInstanceProcAddr function pointer (for FFmpeg).
    pub fn get_instance_proc_addr(&self) -> vk::PFN_vkGetInstanceProcAddr {
        self.entry.static_fn().get_instance_proc_addr
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: We own both handles. Waiting for idle before destruction
        // avoids tearing down the device while work is still in flight.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// SAFETY: Ash's entry/instance/device handles are thin owners around Vulkan
// objects, which are externally synchronized by the caller when required.
unsafe impl Send for VulkanContext {}
// SAFETY: Shared references do not provide interior mutable Rust aliases; the
// synchronization contract remains the one defined by Vulkan.
unsafe impl Sync for VulkanContext {}

/// Wait on a timeline semaphore to the given value.
pub fn wait_timeline_semaphore(
    device: &ash::Device,
    semaphore: vk::Semaphore,
    value: u64,
    timeout_ns: u64,
) -> Result<()> {
    let wait_info = vk::SemaphoreWaitInfo::default()
        .semaphores(std::slice::from_ref(&semaphore))
        .values(std::slice::from_ref(&value));

    // SAFETY: `semaphore` belongs to `device`, the wait info borrows stable
    // local slices, and Vulkan performs the actual synchronization.
    unsafe { device.wait_semaphores(&wait_info, timeout_ns) }
        .map_err(|e| GpuError::Vulkan(format!("vkWaitSemaphores: {e}"), e.as_raw()))
}

/// Query the subresource layout of a VkImage (for NV12 plane offsets/pitches).
pub fn get_image_subresource_layout(
    device: &ash::Device,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
) -> vk::SubresourceLayout {
    let subresource = vk::ImageSubresource {
        aspect_mask: aspect,
        mip_level: 0,
        array_layer: 0,
    };
    // SAFETY: The caller provides an image created from `device`, and the
    // subresource descriptor is fully initialized by value.
    unsafe { device.get_image_subresource_layout(image, subresource) }
}

/// Check if a VkImage uses linear tiling (required for subresource layout query).
pub fn is_linear_tiling(tiling: vk::ImageTiling) -> bool {
    tiling == vk::ImageTiling::LINEAR
}
