//! Manual FFI bindings for FFmpeg's Vulkan hwcontext types.
//!
//! `ffmpeg-sys-next` does not generate bindings from `hwcontext_vulkan.h`,
//! so we define the structs manually to match FFmpeg 8.0.1 (libavutil 60).
//!
//! These structs are used to pass our Vulkan device to FFmpeg for hardware
//! video decoding, and to extract decoded `AVVkFrame` data for DMA-BUF export.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;
use std::os::raw::c_int;

use ash::vk;

/// Number of data pointers in AVFrame (matches `AV_NUM_DATA_POINTERS = 8`).
pub const AV_NUM_DATA_POINTERS: usize = 8;

/// Queue family descriptor used in AVVulkanDeviceContext.
/// Matches `AVVulkanDeviceQueueFamily` from hwcontext_vulkan.h.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AVVulkanDeviceQueueFamily {
    pub idx: c_int,
    pub num: c_int,
    pub flags: vk::QueueFlags,
    pub video_caps: u32, // VkVideoCodecOperationFlagBitsKHR
}

impl Default for AVVulkanDeviceQueueFamily {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Main Vulkan device context, allocated as `AVHWDeviceContext.hwctx`.
/// Matches `AVVulkanDeviceContext` from FFmpeg 8.0.1 hwcontext_vulkan.h.
///
/// Layout includes deprecated queue family fields (FF_API_VULKAN_FIXED_QUEUES
/// is true for libavutil < 61, and FFmpeg 8.0.1 has libavutil 60).
#[repr(C)]
pub struct AVVulkanDeviceContext {
    /// Custom memory allocator (VkAllocationCallbacks*), or null.
    pub alloc: *const c_void,

    /// vkGetInstanceProcAddr function pointer. If null, FFmpeg loads libvulkan.
    /// Using Option<> because PFN_vkGetInstanceProcAddr is non-null.
    pub get_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,

    /// Vulkan instance (must be >= 1.3).
    pub inst: vk::Instance,

    /// Physical device.
    pub phys_dev: vk::PhysicalDevice,

    /// Active (logical) device.
    pub act_dev: vk::Device,

    /// Device features (VkPhysicalDeviceFeatures2).
    /// This is a large struct (vk::PhysicalDeviceFeatures2 = 8 + 4 + 55*4 + padding).
    /// We store it as raw bytes since we don't need to inspect it.
    pub device_features: vk::PhysicalDeviceFeatures2<'static>,

    /// Enabled instance extensions.
    pub enabled_inst_extensions: *const *const i8,
    pub nb_enabled_inst_extensions: c_int,

    /// Enabled device extensions.
    pub enabled_dev_extensions: *const *const i8,
    pub nb_enabled_dev_extensions: c_int,

    // --- Deprecated queue family fields (FF_API_VULKAN_FIXED_QUEUES) ---
    // Present in libavutil 60 (FFmpeg 8.0.1).
    pub queue_family_index: c_int,
    pub nb_graphics_queues: c_int,
    pub queue_family_tx_index: c_int,
    pub nb_tx_queues: c_int,
    pub queue_family_comp_index: c_int,
    pub nb_comp_queues: c_int,
    pub queue_family_encode_index: c_int,
    pub nb_encode_queues: c_int,
    pub queue_family_decode_index: c_int,
    pub nb_decode_queues: c_int,

    /// Queue lock callback.
    pub lock_queue: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void, // AVHWDeviceContext*
            queue_family: u32,
            index: u32,
        ),
    >,

    /// Queue unlock callback.
    pub unlock_queue: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void, // AVHWDeviceContext*
            queue_family: u32,
            index: u32,
        ),
    >,

    /// Queue families used. Array of 64 entries.
    pub qf: [AVVulkanDeviceQueueFamily; 64],
    pub nb_qf: c_int,
}

impl Default for AVVulkanDeviceContext {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Frame flags for AVVulkanFramesContext.
pub type AVVkFrameFlags = u64;
pub const AV_VK_FRAME_FLAG_NONE: AVVkFrameFlags = 1 << 0;
pub const AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE: AVVkFrameFlags = 1 << 2;

/// Frames context, allocated as `AVHWFramesContext.hwctx`.
/// Matches `AVVulkanFramesContext` from hwcontext_vulkan.h.
#[repr(C)]
pub struct AVVulkanFramesContext {
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub create_pnext: *mut c_void,
    pub alloc_pnext: [*mut c_void; AV_NUM_DATA_POINTERS],
    pub flags: AVVkFrameFlags,
    pub img_flags: vk::ImageCreateFlags,
    pub format: [vk::Format; AV_NUM_DATA_POINTERS],
    pub nb_layers: c_int,
    pub lock_frame: Option<
        unsafe extern "C" fn(
            fc: *mut c_void, // AVHWFramesContext*
            vkf: *mut AVVkFrame,
        ),
    >,
    pub unlock_frame: Option<
        unsafe extern "C" fn(
            fc: *mut c_void, // AVHWFramesContext*
            vkf: *mut AVVkFrame,
        ),
    >,
}

impl Default for AVVulkanFramesContext {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Decoded Vulkan frame. Stored at `AVFrame.data[0]`.
/// Matches `struct AVVkFrame` from hwcontext_vulkan.h.
///
/// Note: the size of this struct is NOT part of FFmpeg's ABI.
/// We only access it read-only for DMA-BUF export.
#[repr(C)]
pub struct AVVkFrame {
    /// Vulkan images (one per plane, or one for multiplane).
    pub img: [vk::Image; AV_NUM_DATA_POINTERS],
    /// Image tiling.
    pub tiling: vk::ImageTiling,
    /// Memory backing the images.
    pub mem: [vk::DeviceMemory; AV_NUM_DATA_POINTERS],
    /// Size of each memory allocation.
    pub size: [usize; AV_NUM_DATA_POINTERS],
    /// OR'd memory property flags.
    pub flags: vk::MemoryPropertyFlags,
    /// Access flags per image (updated after barriers).
    pub access: [vk::AccessFlags; AV_NUM_DATA_POINTERS],
    /// Image layout per image.
    pub layout: [vk::ImageLayout; AV_NUM_DATA_POINTERS],
    /// Timeline semaphores (one per image).
    pub sem: [vk::Semaphore; AV_NUM_DATA_POINTERS],
    /// Semaphore values.
    pub sem_value: [u64; AV_NUM_DATA_POINTERS],
    /// Internal data (opaque).
    pub internal: *mut c_void,
    /// Binding offset per image.
    pub offset: [isize; AV_NUM_DATA_POINTERS],
    /// Queue family per image.
    pub queue_family: [u32; AV_NUM_DATA_POINTERS],
}

impl Default for AVVkFrame {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Allocate a `VkExportMemoryAllocateInfo` configured for external memory export.
///
/// Enables both DMA-BUF and opaque fd handle types so that the exported memory
/// can be imported by HIP regardless of which type the driver supports.
///
/// Returns a raw pointer suitable for use as `AVVulkanFramesContext.alloc_pnext`.
/// The caller is responsible for freeing via `drop_export_memory_alloc_info`.
pub fn alloc_export_memory_info_dma_buf() -> *mut c_void {
    let info = Box::new(
        vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
    );
    Box::into_raw(info) as *mut c_void
}

/// Free a `VkExportMemoryAllocateInfo` previously created by
/// `alloc_export_memory_info_dma_buf`.
///
/// # Safety
/// `ptr` must be a pointer returned by `alloc_export_memory_info_dma_buf`
/// that has not yet been freed.
pub unsafe fn drop_export_memory_alloc_info(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr as *mut vk::ExportMemoryAllocateInfo<'_>);
        }
    }
}
