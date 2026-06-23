use std::env;
use std::ffi::{CStr, CString};
use std::mem::{size_of, transmute_copy};
use std::path::{Path, PathBuf};
use std::ptr;
use std::rc::Rc;

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

use std::os::raw::{c_char, c_int, c_uint, c_void};

use snafu::Snafu;

pub const VX_MEM_READ: u32 = 0x1;
pub const VX_MEM_WRITE: u32 = 0x2;
pub const VX_MEM_READ_WRITE: u32 = 0x3;
pub const VX_QUEUE_PRIORITY_NORMAL: u32 = 1;
pub const VX_TIMEOUT_INFINITE: u64 = u64::MAX;

const VX_SUCCESS: VxResult = 0;
const VX_CAPS_NUM_THREADS: u32 = 0x1;
const VX_CAPS_NUM_WARPS: u32 = 0x2;
const VX_CAPS_LOCAL_MEM_SIZE: u32 = 0x6;

type VxResult = c_int;
type VxDeviceH = *mut c_void;
type VxBufferH = *mut c_void;
type VxQueueH = *mut c_void;
type VxEventH = *mut c_void;
type VxModuleH = *mut c_void;
type VxKernelH = *mut c_void;

#[derive(Debug, Snafu)]
pub enum VortexError {
    #[snafu(display(
        "failed to load Vortex runtime library: {message}\ntried candidates:\n{}",
        format_runtime_library_candidates(candidates)
    ))]
    Load {
        candidates: Vec<PathBuf>,
        message: String,
    },
    #[snafu(display("failed to load Vortex symbol {name}: {message}"))]
    Symbol { name: &'static str, message: String },
    #[snafu(display("path contains an interior NUL byte: {}", path.display()))]
    PathContainsNul { path: PathBuf },
    #[snafu(display("Vortex API {operation} failed with code {code}: {message}"))]
    Api {
        operation: &'static str,
        code: VxResult,
        message: String,
    },
    #[snafu(display("invalid Vortex value: {message}"))]
    InvalidValue { message: &'static str },
}

pub type Result<T> = std::result::Result<T, VortexError>;

fn format_runtime_library_candidates(candidates: &[PathBuf]) -> String {
    candidates
        .iter()
        .map(|candidate| format!("  {}", candidate.display()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Clone)]
pub struct Runtime {
    api: Rc<VortexApi>,
}

impl Runtime {
    pub fn load() -> Result<Self> {
        Self::load_from_candidates(default_runtime_library_candidates())
    }

    pub fn load_from_path(path: impl Into<PathBuf>) -> Result<Self> {
        Self::load_from_candidates([path.into()])
    }

    pub fn load_from_candidates<I, P>(candidates: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        let candidates = candidates.into_iter().map(Into::into).collect();
        VortexApi::load(candidates).map(|api| Self { api: Rc::new(api) })
    }

    pub fn open_device(&self, index: u32) -> Result<Device> {
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a valid out pointer and the function pointer was resolved
        // from a loaded Vortex runtime library with the expected C ABI.
        let result = unsafe { (self.api.vx_device_open)(index, &mut handle) };
        self.api.check(result, "vx_device_open")?;
        if handle.is_null() {
            return Err(VortexError::InvalidValue {
                message: "vx_device_open returned a null device",
            });
        }
        Ok(Device {
            api: self.api.clone(),
            handle,
        })
    }
}

pub struct Device {
    api: Rc<VortexApi>,
    handle: VxDeviceH,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VortexDeviceCaps {
    pub(crate) threads_per_warp: u64,
    pub(crate) warps_per_core: u64,
    pub(crate) local_memory_bytes: u64,
}

impl VortexDeviceCaps {
    pub(crate) const fn max_workgroup_threads(self) -> Option<u64> {
        self.threads_per_warp.checked_mul(self.warps_per_core)
    }
}

impl Device {
    pub fn create_queue(&self) -> Result<Queue> {
        let info = VxQueueInfo {
            struct_size: size_of::<VxQueueInfo>(),
            next: ptr::null(),
            priority: VX_QUEUE_PRIORITY_NORMAL,
            flags: 0,
        };
        let mut handle = ptr::null_mut();
        // SAFETY: `self.handle` is owned by `Device`, `info` lives for the call,
        // and `handle` is a valid out pointer for the C API.
        let result = unsafe { (self.api.vx_queue_create)(self.handle, &info, &mut handle) };
        self.api.check(result, "vx_queue_create")?;
        if handle.is_null() {
            return Err(VortexError::InvalidValue {
                message: "vx_queue_create returned a null queue",
            });
        }
        Ok(Queue {
            api: self.api.clone(),
            handle,
        })
    }

    pub fn create_buffer(&self, size: u64, access: u32) -> Result<Buffer> {
        let mut handle = ptr::null_mut();
        // SAFETY: `self.handle` is a valid Vortex device and `handle` is a valid
        // out pointer. The runtime owns the returned buffer handle.
        let result = unsafe { (self.api.vx_buffer_create)(self.handle, size, access, &mut handle) };
        self.api.check(result, "vx_buffer_create")?;
        if handle.is_null() {
            return Err(VortexError::InvalidValue {
                message: "vx_buffer_create returned a null buffer",
            });
        }
        Ok(Buffer {
            api: self.api.clone(),
            handle,
            size,
        })
    }

    pub(crate) fn query_cap(&self, caps_id: u32) -> Result<u64> {
        let mut value = 0;
        // SAFETY: `self.handle` is a valid Vortex device and `value` is a valid out pointer.
        let result = unsafe { (self.api.vx_device_query)(self.handle, caps_id, &mut value) };
        self.api.check(result, "vx_device_query")?;
        Ok(value)
    }

    pub(crate) fn capabilities(&self) -> Result<VortexDeviceCaps> {
        Ok(VortexDeviceCaps {
            threads_per_warp: self.query_cap(VX_CAPS_NUM_THREADS)?,
            warps_per_core: self.query_cap(VX_CAPS_NUM_WARPS)?,
            local_memory_bytes: self.query_cap(VX_CAPS_LOCAL_MEM_SIZE)?,
        })
    }

    pub fn load_module_file(&self, path: &Path) -> Result<Module> {
        let c_path = path_to_cstring(path)?;
        let mut handle = ptr::null_mut();
        // SAFETY: `c_path` is a NUL-terminated path that lives for the call and
        // `handle` is a valid out pointer for the loaded module handle.
        let result =
            unsafe { (self.api.vx_module_load_file)(self.handle, c_path.as_ptr(), &mut handle) };
        self.api.check(result, "vx_module_load_file")?;
        if handle.is_null() {
            return Err(VortexError::InvalidValue {
                message: "vx_module_load_file returned a null module",
            });
        }
        Ok(Module {
            api: self.api.clone(),
            handle,
        })
    }

    pub fn dump_perf(&self) -> Result<()> {
        // SAFETY: `self.handle` is a valid device. Passing NULL asks Vortex to use stdout.
        let result = unsafe { (self.api.vx_device_dump_perf)(self.handle, ptr::null_mut()) };
        self.api.check(result, "vx_device_dump_perf")
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: `Device` owns this handle and calls the matching release once in Drop.
        unsafe { (self.api.vx_device_release)(self.handle) };
    }
}

pub struct Queue {
    api: Rc<VortexApi>,
    handle: VxQueueH,
}

impl Queue {
    pub fn enqueue_write<T>(&self, dst: &Buffer, offset: u64, src: &[T]) -> Result<()> {
        let bytes = byte_len(src)?;
        // SAFETY: `src.as_ptr()` is valid for `bytes` bytes for the duration of the call;
        // Vortex copies/stages the data before returning from enqueue submission.
        let result = unsafe {
            (self.api.vx_enqueue_write)(
                self.handle,
                dst.handle,
                offset,
                src.as_ptr().cast::<c_void>(),
                bytes,
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        };
        self.api.check(result, "vx_enqueue_write")
    }

    pub fn enqueue_read<T>(&self, dst: &mut [T], src: &Buffer, offset: u64) -> Result<Event> {
        let bytes = byte_len(dst)?;
        let mut event = ptr::null_mut();
        // SAFETY: `dst.as_mut_ptr()` is valid for `bytes` bytes for the duration of the call,
        // and `event` is a valid out pointer. The caller controls queue ordering/waits.
        let result = unsafe {
            (self.api.vx_enqueue_read)(
                self.handle,
                dst.as_mut_ptr().cast::<c_void>(),
                src.handle,
                offset,
                bytes,
                0,
                ptr::null(),
                &mut event,
            )
        };
        self.api.check(result, "vx_enqueue_read")?;
        Event::new(self.api.clone(), event, "vx_enqueue_read")
    }

    pub fn enqueue_read_after<T>(
        &self,
        dst: &mut [T],
        src: &Buffer,
        offset: u64,
        wait: &Event,
    ) -> Result<Event> {
        let bytes = byte_len(dst)?;
        let wait_events = [wait.handle];
        let mut event = ptr::null_mut();
        // SAFETY: `dst.as_mut_ptr()` is valid for `bytes` bytes, `wait_events` contains
        // a live event handle, and `event` is a valid out pointer.
        let result = unsafe {
            (self.api.vx_enqueue_read)(
                self.handle,
                dst.as_mut_ptr().cast::<c_void>(),
                src.handle,
                offset,
                bytes,
                1,
                wait_events.as_ptr(),
                &mut event,
            )
        };
        self.api.check(result, "vx_enqueue_read")?;
        Event::new(self.api.clone(), event, "vx_enqueue_read")
    }

    pub fn enqueue_launch<A>(
        &self,
        kernel: &Kernel,
        args: &A,
        ndim: u32,
        grid_dim: [u32; 3],
        block_dim: [u32; 3],
        lmem_size: u32,
    ) -> Result<Event> {
        if !(1..=3).contains(&ndim) {
            return Err(VortexError::InvalidValue {
                message: "launch ndim must be 1, 2, or 3",
            });
        }
        let info = VxLaunchInfo {
            struct_size: size_of::<VxLaunchInfo>(),
            next: ptr::null(),
            kernel: kernel.handle,
            args_host: ptr::from_ref(args).cast::<c_void>(),
            args_size: size_of::<A>(),
            ndim,
            grid_dim,
            block_dim,
            lmem_size,
            cluster_dim: [1, 1, 1],
        };
        let mut event = ptr::null_mut();
        // SAFETY: `info` and `args` live for the call. The Vortex runtime copies the
        // args blob during enqueue, and `event` is a valid out pointer.
        let result =
            unsafe { (self.api.vx_enqueue_launch)(self.handle, &info, 0, ptr::null(), &mut event) };
        self.api.check(result, "vx_enqueue_launch")?;
        Event::new(self.api.clone(), event, "vx_enqueue_launch")
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // SAFETY: `Queue` owns this handle and calls the matching release once in Drop.
        unsafe { (self.api.vx_queue_release)(self.handle) };
    }
}

pub struct Buffer {
    api: Rc<VortexApi>,
    handle: VxBufferH,
    size: u64,
}

impl Buffer {
    pub const fn size(&self) -> u64 {
        self.size
    }

    pub fn address(&self) -> Result<u64> {
        let mut address = 0;
        // SAFETY: `self.handle` is a live buffer and `address` is a valid out pointer.
        let result = unsafe { (self.api.vx_buffer_address)(self.handle, &mut address) };
        self.api.check(result, "vx_buffer_address")?;
        Ok(address)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: `Buffer` owns this handle and calls the matching release once in Drop.
        unsafe { (self.api.vx_buffer_release)(self.handle) };
    }
}

pub struct Module {
    api: Rc<VortexApi>,
    handle: VxModuleH,
}

impl Module {
    pub fn get_kernel(&self, name: &str) -> Result<Kernel> {
        let c_name = CString::new(name).map_err(|_| VortexError::InvalidValue {
            message: "kernel name contains an interior NUL byte",
        })?;
        let mut handle = ptr::null_mut();
        // SAFETY: `c_name` is NUL-terminated and lives for the call, while `handle`
        // is a valid out pointer for the kernel handle.
        let result =
            unsafe { (self.api.vx_module_get_kernel)(self.handle, c_name.as_ptr(), &mut handle) };
        self.api.check(result, "vx_module_get_kernel")?;
        if handle.is_null() {
            return Err(VortexError::InvalidValue {
                message: "vx_module_get_kernel returned a null kernel",
            });
        }
        Ok(Kernel {
            api: self.api.clone(),
            handle,
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: `Module` owns this handle and calls the matching release once in Drop.
        unsafe { (self.api.vx_module_release)(self.handle) };
    }
}

pub struct Kernel {
    api: Rc<VortexApi>,
    handle: VxKernelH,
}

impl Kernel {
    pub fn max_block_size(&self) -> Result<[u32; 3]> {
        let mut x = 0;
        let mut y = 0;
        let mut z = 0;
        // SAFETY: `self.handle` is a live kernel and the output pointers are valid.
        let result =
            unsafe { (self.api.vx_kernel_get_max_block_size)(self.handle, &mut x, &mut y, &mut z) };
        self.api.check(result, "vx_kernel_get_max_block_size")?;
        Ok([x, y, z])
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: `Kernel` owns this handle and calls the matching release once in Drop.
        unsafe { (self.api.vx_kernel_release)(self.handle) };
    }
}

pub struct Event {
    api: Rc<VortexApi>,
    handle: VxEventH,
}

impl Event {
    fn new(api: Rc<VortexApi>, handle: VxEventH, operation: &'static str) -> Result<Self> {
        if handle.is_null() {
            return Err(VortexError::InvalidValue {
                message: match operation {
                    "vx_enqueue_launch" => "vx_enqueue_launch returned a null event",
                    "vx_enqueue_read" => "vx_enqueue_read returned a null event",
                    _ => "Vortex enqueue returned a null event",
                },
            });
        }
        Ok(Self { api, handle })
    }

    pub fn wait(&self) -> Result<()> {
        // SAFETY: `self.handle` is a live event. Waiting for value 1 is the runtime's
        // completion convention used by Vortex regression tests.
        let result = unsafe { (self.api.vx_event_wait_value)(self.handle, 1, VX_TIMEOUT_INFINITE) };
        self.api.check(result, "vx_event_wait_value")
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: `Event` owns this handle and calls the matching release once in Drop.
        unsafe { (self.api.vx_event_release)(self.handle) };
    }
}

#[repr(C)]
struct VxQueueInfo {
    struct_size: usize,
    next: *const c_void,
    priority: c_uint,
    flags: c_uint,
}

#[repr(C)]
struct VxLaunchInfo {
    struct_size: usize,
    next: *const c_void,
    kernel: VxKernelH,
    args_host: *const c_void,
    args_size: usize,
    ndim: c_uint,
    grid_dim: [c_uint; 3],
    block_dim: [c_uint; 3],
    lmem_size: c_uint,
    cluster_dim: [c_uint; 3],
}

type VxResultString = unsafe extern "C" fn(VxResult) -> *const c_char;
type VxDeviceOpen = unsafe extern "C" fn(c_uint, *mut VxDeviceH) -> VxResult;
type VxDeviceRelease = unsafe extern "C" fn(VxDeviceH) -> VxResult;
type VxDeviceDumpPerf = unsafe extern "C" fn(VxDeviceH, *mut c_void) -> VxResult;
type VxDeviceQuery = unsafe extern "C" fn(VxDeviceH, c_uint, *mut u64) -> VxResult;
type VxQueueCreate = unsafe extern "C" fn(VxDeviceH, *const VxQueueInfo, *mut VxQueueH) -> VxResult;
type VxQueueRelease = unsafe extern "C" fn(VxQueueH) -> VxResult;
type VxBufferCreate = unsafe extern "C" fn(VxDeviceH, u64, c_uint, *mut VxBufferH) -> VxResult;
type VxBufferRelease = unsafe extern "C" fn(VxBufferH) -> VxResult;
type VxBufferAddress = unsafe extern "C" fn(VxBufferH, *mut u64) -> VxResult;
type VxModuleLoadFile = unsafe extern "C" fn(VxDeviceH, *const c_char, *mut VxModuleH) -> VxResult;
type VxModuleRelease = unsafe extern "C" fn(VxModuleH) -> VxResult;
type VxModuleGetKernel = unsafe extern "C" fn(VxModuleH, *const c_char, *mut VxKernelH) -> VxResult;
type VxKernelRelease = unsafe extern "C" fn(VxKernelH) -> VxResult;
type VxKernelGetMaxBlockSize =
    unsafe extern "C" fn(VxKernelH, *mut c_uint, *mut c_uint, *mut c_uint) -> VxResult;
type VxEnqueueWrite = unsafe extern "C" fn(
    VxQueueH,
    VxBufferH,
    u64,
    *const c_void,
    u64,
    c_uint,
    *const VxEventH,
    *mut VxEventH,
) -> VxResult;
type VxEnqueueRead = unsafe extern "C" fn(
    VxQueueH,
    *mut c_void,
    VxBufferH,
    u64,
    u64,
    c_uint,
    *const VxEventH,
    *mut VxEventH,
) -> VxResult;
type VxEnqueueLaunch = unsafe extern "C" fn(
    VxQueueH,
    *const VxLaunchInfo,
    c_uint,
    *const VxEventH,
    *mut VxEventH,
) -> VxResult;
type VxEventWaitValue = unsafe extern "C" fn(VxEventH, u64, u64) -> VxResult;
type VxEventRelease = unsafe extern "C" fn(VxEventH) -> VxResult;

struct VortexApi {
    _library: DynamicLibrary,
    vx_result_string: VxResultString,
    vx_device_open: VxDeviceOpen,
    vx_device_release: VxDeviceRelease,
    vx_device_dump_perf: VxDeviceDumpPerf,
    vx_device_query: VxDeviceQuery,
    vx_queue_create: VxQueueCreate,
    vx_queue_release: VxQueueRelease,
    vx_buffer_create: VxBufferCreate,
    vx_buffer_release: VxBufferRelease,
    vx_buffer_address: VxBufferAddress,
    vx_module_load_file: VxModuleLoadFile,
    vx_module_release: VxModuleRelease,
    vx_module_get_kernel: VxModuleGetKernel,
    vx_kernel_release: VxKernelRelease,
    vx_kernel_get_max_block_size: VxKernelGetMaxBlockSize,
    vx_enqueue_write: VxEnqueueWrite,
    vx_enqueue_read: VxEnqueueRead,
    vx_enqueue_launch: VxEnqueueLaunch,
    vx_event_wait_value: VxEventWaitValue,
    vx_event_release: VxEventRelease,
}

impl VortexApi {
    fn load(candidates: Vec<PathBuf>) -> Result<Self> {
        let mut last_error = "no runtime library candidates were available".to_owned();
        for candidate in &candidates {
            match DynamicLibrary::open(candidate) {
                Ok(library) => return Self::from_library(library),
                Err(error) => last_error = error.to_string(),
            }
        }
        Err(VortexError::Load {
            candidates,
            message: last_error,
        })
    }

    fn from_library(library: DynamicLibrary) -> Result<Self> {
        Ok(Self {
            vx_result_string: library.symbol(c"vx_result_string")?,
            vx_device_open: library.symbol(c"vx_device_open")?,
            vx_device_release: library.symbol(c"vx_device_release")?,
            vx_device_dump_perf: library.symbol(c"vx_device_dump_perf")?,
            vx_device_query: library.symbol(c"vx_device_query")?,
            vx_queue_create: library.symbol(c"vx_queue_create")?,
            vx_queue_release: library.symbol(c"vx_queue_release")?,
            vx_buffer_create: library.symbol(c"vx_buffer_create")?,
            vx_buffer_release: library.symbol(c"vx_buffer_release")?,
            vx_buffer_address: library.symbol(c"vx_buffer_address")?,
            vx_module_load_file: library.symbol(c"vx_module_load_file")?,
            vx_module_release: library.symbol(c"vx_module_release")?,
            vx_module_get_kernel: library.symbol(c"vx_module_get_kernel")?,
            vx_kernel_release: library.symbol(c"vx_kernel_release")?,
            vx_kernel_get_max_block_size: library.symbol(c"vx_kernel_get_max_block_size")?,
            vx_enqueue_write: library.symbol(c"vx_enqueue_write")?,
            vx_enqueue_read: library.symbol(c"vx_enqueue_read")?,
            vx_enqueue_launch: library.symbol(c"vx_enqueue_launch")?,
            vx_event_wait_value: library.symbol(c"vx_event_wait_value")?,
            vx_event_release: library.symbol(c"vx_event_release")?,
            _library: library,
        })
    }

    fn check(&self, result: VxResult, operation: &'static str) -> Result<()> {
        if result == VX_SUCCESS {
            return Ok(());
        }
        let message = self.result_message(result);
        Err(VortexError::Api {
            operation,
            code: result,
            message,
        })
    }

    fn result_message(&self, result: VxResult) -> String {
        // SAFETY: `vx_result_string` is a pure Vortex C API returning a static string
        // for any result code. NULL is handled defensively below.
        let raw = unsafe { (self.vx_result_string)(result) };
        if raw.is_null() {
            return "<null vx_result_string>".to_owned();
        }
        // SAFETY: Vortex returns a NUL-terminated static C string for result names.
        unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned()
    }
}

pub fn default_runtime_library_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("MANDREL_VORTEX_RUNTIME_LIB").map(PathBuf::from) {
        candidates.push(path);
    }
    if let Some(path) = env::var_os("VORTEX_PATH").map(PathBuf::from) {
        candidates.push(path.join("runtime/lib").join(runtime_library_name()));
    }
    if let Some(path) = env::var_os("VORTEX_BUILD_DIR").map(PathBuf::from) {
        candidates.push(path.join("sw/runtime").join(runtime_library_name()));
    }
    candidates.push(PathBuf::from(runtime_library_name()));
    candidates
}

fn runtime_library_name() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libvortex.dylib"
    }
    #[cfg(not(target_os = "macos"))]
    {
        "libvortex.so"
    }
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    {
        CString::new(path.as_os_str().as_bytes()).map_err(|_| VortexError::PathContainsNul {
            path: path.to_path_buf(),
        })
    }
    #[cfg(not(unix))]
    {
        CString::new(path.to_string_lossy().as_bytes()).map_err(|_| VortexError::PathContainsNul {
            path: path.to_path_buf(),
        })
    }
}

fn byte_len<T>(slice: &[T]) -> Result<u64> {
    let bytes = slice
        .len()
        .checked_mul(size_of::<T>())
        .ok_or(VortexError::InvalidValue {
            message: "slice byte length overflow",
        })?;
    u64::try_from(bytes).map_err(|_| VortexError::InvalidValue {
        message: "slice byte length does not fit u64",
    })
}

struct DynamicLibrary {
    handle: *mut c_void,
}

impl DynamicLibrary {
    fn open(path: &Path) -> Result<Self> {
        let c_path = path_to_cstring(path)?;
        // SAFETY: `c_path` is a valid NUL-terminated path. `dlopen` returns either
        // a library handle or NULL with an error retrievable through `dlerror`.
        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(VortexError::Load {
                candidates: vec![path.to_path_buf()],
                message: dl_error_message(),
            });
        }
        Ok(Self { handle })
    }

    fn symbol<T: Copy>(&self, name: &'static CStr) -> Result<T> {
        // SAFETY: `name` is a static NUL-terminated symbol name and `self.handle`
        // is a live dynamic-library handle.
        let symbol = unsafe { dlsym(self.handle, name.as_ptr()) };
        if symbol.is_null() {
            return Err(VortexError::Symbol {
                name: name.to_str().unwrap_or("<non-utf8>"),
                message: dl_error_message(),
            });
        }
        debug_assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
        // SAFETY: every requested `T` is an `unsafe extern "C" fn` pointer type,
        // which has the same representation as the raw symbol pointer on this target.
        Ok(unsafe { transmute_copy::<*mut c_void, T>(&symbol) })
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        // SAFETY: `DynamicLibrary` owns this `dlopen` handle and closes it once.
        unsafe { dlclose(self.handle) };
    }
}

const RTLD_NOW: c_int = 2;

#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

fn dl_error_message() -> String {
    // SAFETY: `dlerror` returns NULL or a NUL-terminated thread-local error string.
    let raw = unsafe { dlerror() };
    if raw.is_null() {
        "unknown dynamic-loader error".to_owned()
    } else {
        // SAFETY: non-NULL `dlerror` output is a NUL-terminated C string.
        unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{VxLaunchInfo, VxQueueInfo};
    use std::mem::size_of;

    #[test]
    fn c_layouts_match_expected_sizes_on_64_bit_host() {
        if size_of::<usize>() == 8 {
            assert_eq!(size_of::<VxQueueInfo>(), 24);
            assert_eq!(size_of::<VxLaunchInfo>(), 88);
        }
    }
}
