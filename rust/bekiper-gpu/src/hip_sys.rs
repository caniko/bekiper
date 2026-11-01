//! Minimal FFI bindings to the HIP runtime API.
//!
//! Uses runtime loading (`dlopen`) via `libloading` instead of link-time
//! binding to avoid conflict with ORT's ROCm EP, which also initializes
//! HIP internally. Link-time loading causes a deadlock on HIP's global
//! init mutex.

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    dead_code,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc
)]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::OnceLock;

// ── Error type ──────────────────────────────────────────────────────

pub type hipError_t = c_int;
pub const hipSuccess: hipError_t = 0;

// ── Opaque handles ──────────────────────────────────────────────────

pub type hipStream_t = *mut c_void;
pub type hipExternalMemory_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipDeviceptr_t = *mut c_void;

// ── Memory copy kinds ───────────────────────────────────────────────

pub type hipMemcpyKind = c_int;
pub const hipMemcpyHostToDevice: hipMemcpyKind = 1;
pub const hipMemcpyDeviceToHost: hipMemcpyKind = 2;
pub const hipMemcpyDeviceToDevice: hipMemcpyKind = 3;

// ── External memory handle types ────────────────────────────────────

pub type hipExternalMemoryHandleType = c_int;
pub const hipExternalMemoryHandleTypeOpaqueFd: hipExternalMemoryHandleType = 1;
pub const hipExternalMemoryHandleTypeDmaBufFd: hipExternalMemoryHandleType = 16;

// ── External memory descriptors ─────────────────────────────────────

#[repr(C)]
pub struct hipExternalMemoryHandleDesc {
    pub type_: hipExternalMemoryHandleType,
    pub handle: hipExternalMemoryHandle,
    pub size: u64,
    pub flags: u32,
    _reserved: [u32; 16],
}

#[repr(C)]
pub union hipExternalMemoryHandle {
    pub fd: c_int,
    /// win32: { void *handle; const void *name; } — 16 bytes, 8-byte aligned
    _win32: [*mut c_void; 2],
}

#[repr(C)]
pub struct hipExternalMemoryBufferDesc {
    pub offset: u64,
    pub size: u64,
    pub flags: u32,
    _reserved: [u32; 16],
}

impl Default for hipExternalMemoryHandleDesc {
    fn default() -> Self {
        // SAFETY: The HIP C API treats a zeroed descriptor as the empty/default
        // state before the caller fills in the active handle variant and size.
        unsafe { std::mem::zeroed() }
    }
}

impl Default for hipExternalMemoryBufferDesc {
    fn default() -> Self {
        // SAFETY: The HIP C API treats a zeroed descriptor as the empty/default
        // state before the caller fills in the mapped range.
        unsafe { std::mem::zeroed() }
    }
}

// ── dim3 for kernel launch ──────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

// ── Runtime-loaded HIP function table ───────────────────────────────

struct HipLib {
    _lib: libloading::Library,

    hip_init: unsafe extern "C" fn(u32) -> hipError_t,
    hip_set_device: unsafe extern "C" fn(c_int) -> hipError_t,
    hip_device_synchronize: unsafe extern "C" fn() -> hipError_t,

    hip_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> hipError_t,
    hip_free: unsafe extern "C" fn(*mut c_void) -> hipError_t,
    hip_memcpy:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, hipMemcpyKind) -> hipError_t,
    hip_memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> hipError_t,

    hip_stream_create: unsafe extern "C" fn(*mut hipStream_t) -> hipError_t,
    hip_stream_destroy: unsafe extern "C" fn(hipStream_t) -> hipError_t,
    hip_stream_synchronize: unsafe extern "C" fn(hipStream_t) -> hipError_t,

    hip_import_external_memory: unsafe extern "C" fn(
        *mut hipExternalMemory_t,
        *const hipExternalMemoryHandleDesc,
    ) -> hipError_t,
    hip_external_memory_get_mapped_buffer: unsafe extern "C" fn(
        *mut *mut c_void,
        hipExternalMemory_t,
        *const hipExternalMemoryBufferDesc,
    ) -> hipError_t,
    hip_destroy_external_memory: unsafe extern "C" fn(hipExternalMemory_t) -> hipError_t,

    hip_module_load: unsafe extern "C" fn(*mut hipModule_t, *const u8) -> hipError_t,
    hip_module_load_data: unsafe extern "C" fn(*mut hipModule_t, *const c_void) -> hipError_t,
    hip_module_unload: unsafe extern "C" fn(hipModule_t) -> hipError_t,
    hip_module_get_function:
        unsafe extern "C" fn(*mut hipFunction_t, hipModule_t, *const u8) -> hipError_t,
    hip_module_launch_kernel: unsafe extern "C" fn(
        hipFunction_t,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        hipStream_t,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> hipError_t,
}

// SAFETY: `HipLib` only stores an owned `Library` handle plus immutable function
// pointers resolved from it. The underlying HIP runtime is documented as
// process-global and thread-safe once initialized.
unsafe impl Send for HipLib {}
// SAFETY: See the `Send` justification above; shared access never mutates the
// stored symbols after initialization.
unsafe impl Sync for HipLib {}

static HIP: OnceLock<HipLib> = OnceLock::new();

unsafe fn load_symbol<T: Copy>(lib: &libloading::Library, name: &[u8]) -> T {
    // SAFETY: The caller supplies the exact function-pointer type for `name`,
    // and we copy the pointer value out while the library remains owned by
    // `HipLib` for the rest of the process.
    let sym: libloading::Symbol<'_, T> = unsafe { lib.get(name) }
        .unwrap_or_else(|e| panic!("symbol {} not found: {e}", String::from_utf8_lossy(name)));
    *sym
}

fn hip() -> &'static HipLib {
    HIP.get_or_init(|| unsafe {
        // SAFETY: `libamdhip64.so` is the process-global HIP runtime we target.
        // `HipLib` retains the library handle for the lifetime of every copied
        // function pointer below.
        let lib = libloading::Library::new("libamdhip64.so")
            .expect("failed to load libamdhip64.so — is ROCm installed?");

        HipLib {
            hip_init: load_symbol(&lib, b"hipInit"),
            hip_set_device: load_symbol(&lib, b"hipSetDevice"),
            hip_device_synchronize: load_symbol(&lib, b"hipDeviceSynchronize"),
            hip_malloc: load_symbol(&lib, b"hipMalloc"),
            hip_free: load_symbol(&lib, b"hipFree"),
            hip_memcpy: load_symbol(&lib, b"hipMemcpy"),
            hip_memset: load_symbol(&lib, b"hipMemset"),
            hip_stream_create: load_symbol(&lib, b"hipStreamCreate"),
            hip_stream_destroy: load_symbol(&lib, b"hipStreamDestroy"),
            hip_stream_synchronize: load_symbol(&lib, b"hipStreamSynchronize"),
            hip_import_external_memory: load_symbol(&lib, b"hipImportExternalMemory"),
            hip_external_memory_get_mapped_buffer: load_symbol(
                &lib,
                b"hipExternalMemoryGetMappedBuffer",
            ),
            hip_destroy_external_memory: load_symbol(&lib, b"hipDestroyExternalMemory"),
            hip_module_load: load_symbol(&lib, b"hipModuleLoad"),
            hip_module_load_data: load_symbol(&lib, b"hipModuleLoadData"),
            hip_module_unload: load_symbol(&lib, b"hipModuleUnload"),
            hip_module_get_function: load_symbol(&lib, b"hipModuleGetFunction"),
            hip_module_launch_kernel: load_symbol(&lib, b"hipModuleLaunchKernel"),
            _lib: lib,
        }
    })
}

// ── Public API (same signatures as the original extern block) ───────

pub unsafe fn hipInit(flags: u32) -> hipError_t {
    (hip().hip_init)(flags)
}

pub unsafe fn hipSetDevice(device_id: c_int) -> hipError_t {
    (hip().hip_set_device)(device_id)
}

pub unsafe fn hipDeviceSynchronize() -> hipError_t {
    (hip().hip_device_synchronize)()
}

pub unsafe fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t {
    (hip().hip_malloc)(ptr, size)
}

pub unsafe fn hipFree(ptr: *mut c_void) -> hipError_t {
    (hip().hip_free)(ptr)
}

pub unsafe fn hipMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    kind: hipMemcpyKind,
) -> hipError_t {
    (hip().hip_memcpy)(dst, src, size, kind)
}

pub unsafe fn hipMemset(ptr: *mut c_void, value: c_int, size: usize) -> hipError_t {
    (hip().hip_memset)(ptr, value, size)
}

pub unsafe fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t {
    (hip().hip_stream_create)(stream)
}

pub unsafe fn hipStreamDestroy(stream: hipStream_t) -> hipError_t {
    (hip().hip_stream_destroy)(stream)
}

pub unsafe fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t {
    (hip().hip_stream_synchronize)(stream)
}

pub unsafe fn hipImportExternalMemory(
    ext_mem: *mut hipExternalMemory_t,
    desc: *const hipExternalMemoryHandleDesc,
) -> hipError_t {
    (hip().hip_import_external_memory)(ext_mem, desc)
}

pub unsafe fn hipExternalMemoryGetMappedBuffer(
    dev_ptr: *mut *mut c_void,
    ext_mem: hipExternalMemory_t,
    desc: *const hipExternalMemoryBufferDesc,
) -> hipError_t {
    (hip().hip_external_memory_get_mapped_buffer)(dev_ptr, ext_mem, desc)
}

pub unsafe fn hipDestroyExternalMemory(ext_mem: hipExternalMemory_t) -> hipError_t {
    (hip().hip_destroy_external_memory)(ext_mem)
}

pub unsafe fn hipModuleLoad(module: *mut hipModule_t, file_name: *const u8) -> hipError_t {
    (hip().hip_module_load)(module, file_name)
}

pub unsafe fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t {
    (hip().hip_module_load_data)(module, image)
}

pub unsafe fn hipModuleUnload(module: hipModule_t) -> hipError_t {
    (hip().hip_module_unload)(module)
}

pub unsafe fn hipModuleGetFunction(
    func: *mut hipFunction_t,
    module: hipModule_t,
    name: *const u8,
) -> hipError_t {
    (hip().hip_module_get_function)(func, module, name)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn hipModuleLaunchKernel(
    func: hipFunction_t,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: hipStream_t,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> hipError_t {
    (hip().hip_module_launch_kernel)(
        func,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        stream,
        kernel_params,
        extra,
    )
}
