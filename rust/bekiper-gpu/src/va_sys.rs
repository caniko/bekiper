//! Minimal FFI bindings to libva for VA-API surface operations.
//!
//! Covers DMA-BUF export (`vaExportSurfaceHandle`) and surface readback
//! (`vaDeriveImage` + `vaMapBuffer`) for the GPU preprocessing pipeline.

#![allow(non_camel_case_types, non_upper_case_globals, dead_code)]

use std::ffi::c_void;
use std::os::raw::c_int;

// ── Types ───────────────────────────────────────────────────────────

pub type VADisplay = *mut c_void;
pub type VASurfaceID = u32;
pub type VAStatus = c_int;

pub const VA_STATUS_SUCCESS: VAStatus = 0;

// Surface export flags
pub const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2: u32 = 0x40000000;
pub const VA_EXPORT_SURFACE_READ_ONLY: u32 = 0x0001;
pub const VA_EXPORT_SURFACE_SEPARATE_LAYERS: u32 = 0x0004;

// ── DRM PRIME 2 surface descriptor ──────────────────────────────────

pub const VA_DRM_PRIME_MAX_OBJECTS: usize = 4;
pub const VA_DRM_PRIME_MAX_LAYERS: usize = 4;
pub const VA_DRM_PRIME_MAX_PLANES: usize = 4;

/// Pixel format (FourCC) constants.
pub const VA_FOURCC_NV12: u32 = u32::from_le_bytes(*b"NV12");
pub const VA_FOURCC_P010: u32 = u32::from_le_bytes(*b"P010");

#[repr(C)]
#[derive(Debug, Clone)]
pub struct VADRMPRIMESurfaceDescriptor {
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub num_objects: u32,
    pub objects: [VADRMPRIMEObject; VA_DRM_PRIME_MAX_OBJECTS],
    pub num_layers: u32,
    pub layers: [VADRMPRIMELayer; VA_DRM_PRIME_MAX_LAYERS],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VADRMPRIMEObject {
    pub fd: c_int,
    pub size: u32,
    pub drm_format_modifier: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VADRMPRIMELayer {
    pub drm_format: u32,
    pub num_planes: u32,
    pub object_index: [u32; VA_DRM_PRIME_MAX_PLANES],
    pub offset: [u32; VA_DRM_PRIME_MAX_PLANES],
    pub pitch: [u32; VA_DRM_PRIME_MAX_PLANES],
}

impl Default for VADRMPRIMESurfaceDescriptor {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ── VAImage for surface readback ─────────────────────────────────────

pub type VAImageID = u32;
pub type VABufferID = u32;

/// VA image format descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VAImageFormat {
    pub fourcc: u32,
    pub byte_order: u32,
    pub bits_per_pixel: u32,
    pub depth: u32,
    pub red_mask: u32,
    pub green_mask: u32,
    pub blue_mask: u32,
    pub alpha_mask: u32,
    pub va_reserved: [u32; 4],
}

/// VA image structure returned by `vaDeriveImage`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VAImage {
    pub image_id: VAImageID,
    pub format: VAImageFormat,
    pub buf: VABufferID,
    pub width: u16,
    pub height: u16,
    pub data_size: u32,
    pub num_planes: u32,
    pub pitches: [u32; 3],
    pub offsets: [u32; 3],
    pub num_palette_entries: i32,
    pub entry_bytes: i32,
    pub component_order: [i8; 4],
    pub va_reserved: [u32; 4],
}

impl Default for VAImage {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ── FFI declarations ────────────────────────────────────────────────

#[link(name = "va")]
unsafe extern "C" {
    pub fn vaSyncSurface(dpy: VADisplay, surface: VASurfaceID) -> VAStatus;

    pub fn vaExportSurfaceHandle(
        dpy: VADisplay,
        surface: VASurfaceID,
        mem_type: u32,
        flags: u32,
        descriptor: *mut c_void,
    ) -> VAStatus;

    pub fn vaDeriveImage(dpy: VADisplay, surface: VASurfaceID, image: *mut VAImage) -> VAStatus;

    pub fn vaMapBuffer(dpy: VADisplay, buf_id: VABufferID, pbuf: *mut *mut c_void) -> VAStatus;

    pub fn vaUnmapBuffer(dpy: VADisplay, buf_id: VABufferID) -> VAStatus;

    pub fn vaDestroyImage(dpy: VADisplay, image_id: VAImageID) -> VAStatus;
}
