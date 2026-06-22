use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::fmt;
use std::mem::size_of;
use std::os::raw::c_char;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;
use std::slice;

#[cfg(unix)]
use std::ffi::OsStr;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

use crate::backend::{VortexBackend, VortexBackendConfig};
use crate::matmul::MatmulShape;
use crate::vortex2::Runtime;

pub const MANDREL_VORTEX_STATUS_OK: i32 = 0;
pub const MANDREL_VORTEX_STATUS_NULL_POINTER: i32 = 1;
pub const MANDREL_VORTEX_STATUS_INVALID_ARGUMENT: i32 = 2;
pub const MANDREL_VORTEX_STATUS_RUNTIME_ERROR: i32 = 3;
pub const MANDREL_VORTEX_STATUS_PANIC: i32 = 255;

pub const MANDREL_VORTEX_DTYPE_I8: u32 = 1;
pub const MANDREL_VORTEX_DTYPE_I32: u32 = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MandrelVortexBackendConfig {
    pub struct_size: usize,
    pub runtime_library_path: *const c_char,
    pub kernel_vxbin_path: *const c_char,
    pub device_index: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MandrelVortexMulMatDesc {
    pub struct_size: usize,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lhs_stride: u32,
    pub rhs_stride: u32,
    pub out_stride: u32,
    pub lhs_dtype: u32,
    pub rhs_dtype: u32,
    pub out_dtype: u32,
    pub flags: u32,
}

pub struct MandrelVortexBackend {
    backend: VortexBackend,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(c"".to_owned());
}

#[unsafe(no_mangle)]
pub extern "C" fn mandrel_vortex_backend_name() -> *const c_char {
    c"mandrel-vortex".as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn mandrel_vortex_status_message(status: i32) -> *const c_char {
    match status {
        MANDREL_VORTEX_STATUS_OK => c"ok".as_ptr(),
        MANDREL_VORTEX_STATUS_NULL_POINTER => c"null pointer".as_ptr(),
        MANDREL_VORTEX_STATUS_INVALID_ARGUMENT => c"invalid argument".as_ptr(),
        MANDREL_VORTEX_STATUS_RUNTIME_ERROR => c"runtime error".as_ptr(),
        MANDREL_VORTEX_STATUS_PANIC => c"panic".as_ptr(),
        _ => c"unknown status".as_ptr(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mandrel_vortex_last_error_message() -> *const c_char {
    LAST_ERROR.with(|message| message.borrow().as_ptr())
}

/// # Safety
///
/// `config` must point to a valid `MandrelVortexBackendConfig` for the duration of the call.
/// `out_backend` must point to writable storage for one backend handle. C strings in `config`
/// must be NUL-terminated. The returned handle must be released with
/// `mandrel_vortex_backend_destroy` exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mandrel_vortex_backend_create(
    config: *const MandrelVortexBackendConfig,
    out_backend: *mut *mut MandrelVortexBackend,
) -> i32 {
    ffi_status(|| {
        clear_last_error();
        if out_backend.is_null() {
            return fail_status(
                MANDREL_VORTEX_STATUS_NULL_POINTER,
                "out_backend must not be null",
            );
        }
        // SAFETY: The safety contract of this function requires `out_backend` to be writable.
        unsafe { *out_backend = ptr::null_mut() };

        let config = match require_ref(config, "config") {
            Ok(config) => config,
            Err(status) => return status,
        };
        if let Err(status) = validate_struct_size(
            config.struct_size,
            size_of::<MandrelVortexBackendConfig>(),
            "MandrelVortexBackendConfig",
        ) {
            return status;
        }

        let kernel_vxbin = match path_from_c_str(config.kernel_vxbin_path, "kernel_vxbin_path") {
            Ok(path) => path,
            Err(status) => return status,
        };
        let runtime = if config.runtime_library_path.is_null() {
            match Runtime::load() {
                Ok(runtime) => runtime,
                Err(error) => return fail_display(MANDREL_VORTEX_STATUS_RUNTIME_ERROR, error),
            }
        } else {
            let runtime_library =
                match path_from_c_str(config.runtime_library_path, "runtime_library_path") {
                    Ok(path) => path,
                    Err(status) => return status,
                };
            match Runtime::load_from_path(runtime_library) {
                Ok(runtime) => runtime,
                Err(error) => return fail_display(MANDREL_VORTEX_STATUS_RUNTIME_ERROR, error),
            }
        };

        let backend_config =
            VortexBackendConfig::new(kernel_vxbin).with_device_index(config.device_index);
        let backend = match VortexBackend::from_runtime(runtime, backend_config) {
            Ok(backend) => backend,
            Err(error) => return fail_display(MANDREL_VORTEX_STATUS_RUNTIME_ERROR, error),
        };
        let boxed = Box::new(MandrelVortexBackend { backend });
        // SAFETY: `out_backend` was checked non-null and is writable per this function's contract.
        unsafe { *out_backend = Box::into_raw(boxed) };
        MANDREL_VORTEX_STATUS_OK
    })
}

/// # Safety
///
/// `backend` must either be null or a handle returned by `mandrel_vortex_backend_create` that has
/// not already been destroyed. After this call the handle must not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mandrel_vortex_backend_destroy(backend: *mut MandrelVortexBackend) {
    if backend.is_null() {
        return;
    }
    // SAFETY: The safety contract requires `backend` to be a unique handle from `Box::into_raw`.
    unsafe { drop(Box::from_raw(backend)) };
}

/// # Safety
///
/// `desc` must either be null or point to a valid `MandrelVortexMulMatDesc` for the duration
/// of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mandrel_vortex_can_offload_mul_mat(
    desc: *const MandrelVortexMulMatDesc,
) -> bool {
    ffi_bool(|| {
        clear_last_error();
        let desc = match require_ref(desc, "desc") {
            Ok(desc) => desc,
            Err(status) => return Err(status),
        };
        validate_desc_struct_size(desc)?;
        Ok(desc_is_supported(desc))
    })
}

/// # Safety
///
/// `backend` must be a live handle returned by `mandrel_vortex_backend_create`. `desc` must point
/// to a valid descriptor. `lhs`, `rhs`, and `out` must point to buffers valid for the supplied
/// element counts. The current ABI supports contiguous row-major `i8 * i8 -> i32` only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mandrel_vortex_mul_mat_i8_i8_i32(
    backend: *mut MandrelVortexBackend,
    desc: *const MandrelVortexMulMatDesc,
    lhs: *const i8,
    lhs_len: usize,
    rhs: *const i8,
    rhs_len: usize,
    out: *mut i32,
    out_len: usize,
) -> i32 {
    ffi_status(|| {
        clear_last_error();
        let backend = match require_mut(backend, "backend") {
            Ok(backend) => backend,
            Err(status) => return status,
        };
        let desc = match require_ref(desc, "desc") {
            Ok(desc) => desc,
            Err(status) => return status,
        };
        if let Err(status) = validate_desc_struct_size(desc) {
            return status;
        }
        if !desc_is_supported(desc) {
            return fail_status(
                MANDREL_VORTEX_STATUS_INVALID_ARGUMENT,
                "only contiguous row-major i8*i8->i32 matmul is currently supported",
            );
        }

        let shape = MatmulShape::new(desc.m, desc.n, desc.k);
        let lhs_expected = match checked_matrix_len(desc.m, desc.k, "lhs") {
            Ok(len) => len,
            Err(status) => return status,
        };
        let rhs_expected = match checked_matrix_len(desc.k, desc.n, "rhs") {
            Ok(len) => len,
            Err(status) => return status,
        };
        let out_expected = match checked_matrix_len(desc.m, desc.n, "out") {
            Ok(len) => len,
            Err(status) => return status,
        };
        if lhs_len != lhs_expected {
            return fail_status(
                MANDREL_VORTEX_STATUS_INVALID_ARGUMENT,
                format!("lhs length mismatch: expected {lhs_expected}, got {lhs_len}"),
            );
        }
        if rhs_len != rhs_expected {
            return fail_status(
                MANDREL_VORTEX_STATUS_INVALID_ARGUMENT,
                format!("rhs length mismatch: expected {rhs_expected}, got {rhs_len}"),
            );
        }
        if out_len != out_expected {
            return fail_status(
                MANDREL_VORTEX_STATUS_INVALID_ARGUMENT,
                format!("out length mismatch: expected {out_expected}, got {out_len}"),
            );
        }
        if lhs.is_null() {
            return fail_status(MANDREL_VORTEX_STATUS_NULL_POINTER, "lhs must not be null");
        }
        if rhs.is_null() {
            return fail_status(MANDREL_VORTEX_STATUS_NULL_POINTER, "rhs must not be null");
        }
        if out.is_null() {
            return fail_status(MANDREL_VORTEX_STATUS_NULL_POINTER, "out must not be null");
        }

        // SAFETY: The safety contract plus length checks above require `lhs` to be valid.
        let lhs = unsafe { slice::from_raw_parts(lhs, lhs_len) };
        // SAFETY: The safety contract plus length checks above require `rhs` to be valid.
        let rhs = unsafe { slice::from_raw_parts(rhs, rhs_len) };
        let values = match backend.backend.mul_mat_i8_i8_i32(shape, lhs, rhs) {
            Ok(values) => values,
            Err(error) => return fail_display(MANDREL_VORTEX_STATUS_RUNTIME_ERROR, error),
        };
        if values.len() != out_expected {
            return fail_status(
                MANDREL_VORTEX_STATUS_RUNTIME_ERROR,
                "backend returned an unexpected output length",
            );
        }
        // SAFETY: `out` is valid for `out_len` elements per this function's safety contract,
        // and `values.len()` equals `out_len` by the checks above.
        unsafe { ptr::copy_nonoverlapping(values.as_ptr(), out, values.len()) };
        MANDREL_VORTEX_STATUS_OK
    })
}

fn ffi_status(function: impl FnOnce() -> i32) -> i32 {
    match panic::catch_unwind(AssertUnwindSafe(function)) {
        Ok(status) => status,
        Err(_) => fail_status(
            MANDREL_VORTEX_STATUS_PANIC,
            "panic crossed Vortex C ABI boundary",
        ),
    }
}

fn ffi_bool(function: impl FnOnce() -> Result<bool, i32>) -> bool {
    match panic::catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(value)) => value,
        Ok(Err(_status)) => false,
        Err(_) => {
            let _status = fail_status(
                MANDREL_VORTEX_STATUS_PANIC,
                "panic crossed Vortex C ABI boundary",
            );
            false
        }
    }
}

fn clear_last_error() {
    LAST_ERROR.with(|message| {
        *message.borrow_mut() = c"".to_owned();
    });
}

fn fail_status(status: i32, message: impl fmt::Display) -> i32 {
    set_last_error(message);
    status
}

fn fail_display(status: i32, error: impl fmt::Display) -> i32 {
    fail_status(status, error)
}

fn set_last_error(message: impl fmt::Display) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = cstring_without_nul(&message.to_string());
    });
}

fn cstring_without_nul(message: &str) -> CString {
    let bytes = message
        .as_bytes()
        .iter()
        .map(|byte| if *byte == 0 { b' ' } else { *byte })
        .collect::<Vec<_>>();
    match CString::new(bytes) {
        Ok(value) => value,
        Err(_error) => c"failed to store error message".to_owned(),
    }
}

fn validate_struct_size(actual: usize, expected: usize, name: &'static str) -> Result<(), i32> {
    if actual < expected {
        return Err(fail_status(
            MANDREL_VORTEX_STATUS_INVALID_ARGUMENT,
            format!("{name}.struct_size is too small: expected at least {expected}, got {actual}"),
        ));
    }
    Ok(())
}

fn validate_desc_struct_size(desc: &MandrelVortexMulMatDesc) -> Result<(), i32> {
    validate_struct_size(
        desc.struct_size,
        size_of::<MandrelVortexMulMatDesc>(),
        "MandrelVortexMulMatDesc",
    )
}

fn desc_is_supported(desc: &MandrelVortexMulMatDesc) -> bool {
    if desc.m == 0 || desc.n == 0 || desc.k == 0 {
        return false;
    }
    if desc.lhs_dtype != MANDREL_VORTEX_DTYPE_I8
        || desc.rhs_dtype != MANDREL_VORTEX_DTYPE_I8
        || desc.out_dtype != MANDREL_VORTEX_DTYPE_I32
    {
        return false;
    }

    stride_or_default(desc.lhs_stride, desc.k) == desc.k
        && stride_or_default(desc.rhs_stride, desc.n) == desc.n
        && stride_or_default(desc.out_stride, desc.n) == desc.n
}

const fn stride_or_default(stride: u32, default: u32) -> u32 {
    if stride == 0 { default } else { stride }
}

fn checked_matrix_len(rows: u32, cols: u32, name: &'static str) -> Result<usize, i32> {
    let len = match u64::from(rows).checked_mul(u64::from(cols)) {
        Some(len) => len,
        None => {
            return Err(fail_status(
                MANDREL_VORTEX_STATUS_INVALID_ARGUMENT,
                format!("{name} matrix element count overflow"),
            ));
        }
    };
    match usize::try_from(len) {
        Ok(len) => Ok(len),
        Err(_error) => Err(fail_status(
            MANDREL_VORTEX_STATUS_INVALID_ARGUMENT,
            format!("{name} matrix element count does not fit size_t"),
        )),
    }
}

fn require_ref<'a, T>(ptr: *const T, name: &'static str) -> Result<&'a T, i32> {
    if ptr.is_null() {
        return Err(fail_status(
            MANDREL_VORTEX_STATUS_NULL_POINTER,
            format!("{name} must not be null"),
        ));
    }
    // SAFETY: The caller of the enclosing FFI function guarantees that `ptr` is valid.
    Ok(unsafe { &*ptr })
}

fn require_mut<'a, T>(ptr: *mut T, name: &'static str) -> Result<&'a mut T, i32> {
    if ptr.is_null() {
        return Err(fail_status(
            MANDREL_VORTEX_STATUS_NULL_POINTER,
            format!("{name} must not be null"),
        ));
    }
    // SAFETY: The caller of the enclosing FFI function guarantees unique mutable access.
    Ok(unsafe { &mut *ptr })
}

fn path_from_c_str(ptr: *const c_char, name: &'static str) -> Result<PathBuf, i32> {
    if ptr.is_null() {
        return Err(fail_status(
            MANDREL_VORTEX_STATUS_NULL_POINTER,
            format!("{name} must not be null"),
        ));
    }
    // SAFETY: The enclosing FFI function's safety contract requires `ptr` to be NUL-terminated.
    let value = unsafe { CStr::from_ptr(ptr) };
    if value.to_bytes().is_empty() {
        return Err(fail_status(
            MANDREL_VORTEX_STATUS_INVALID_ARGUMENT,
            format!("{name} must not be empty"),
        ));
    }
    Ok(path_from_cstr(value))
}

#[cfg(unix)]
fn path_from_cstr(value: &CStr) -> PathBuf {
    PathBuf::from(OsStr::from_bytes(value.to_bytes()))
}

#[cfg(not(unix))]
fn path_from_cstr(value: &CStr) -> PathBuf {
    PathBuf::from(value.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::{
        MANDREL_VORTEX_DTYPE_I8, MANDREL_VORTEX_DTYPE_I32, MandrelVortexBackendConfig,
        MandrelVortexMulMatDesc, desc_is_supported,
    };
    use std::mem::size_of;
    use std::ptr;

    #[test]
    fn ffi_layouts_have_stable_sizes_on_64_bit_host() {
        if size_of::<usize>() == 8 {
            assert_eq!(size_of::<MandrelVortexBackendConfig>(), 32);
            assert_eq!(size_of::<MandrelVortexMulMatDesc>(), 48);
        }
    }

    #[test]
    fn contiguous_i8_i8_i32_desc_is_supported() {
        let desc = MandrelVortexMulMatDesc {
            struct_size: size_of::<MandrelVortexMulMatDesc>(),
            m: 8,
            n: 16,
            k: 32,
            lhs_stride: 0,
            rhs_stride: 0,
            out_stride: 0,
            lhs_dtype: MANDREL_VORTEX_DTYPE_I8,
            rhs_dtype: MANDREL_VORTEX_DTYPE_I8,
            out_dtype: MANDREL_VORTEX_DTYPE_I32,
            flags: 0,
        };

        assert!(desc_is_supported(&desc));
    }

    #[test]
    fn non_contiguous_desc_is_not_supported_yet() {
        let desc = MandrelVortexMulMatDesc {
            struct_size: size_of::<MandrelVortexMulMatDesc>(),
            m: 8,
            n: 16,
            k: 32,
            lhs_stride: 33,
            rhs_stride: 16,
            out_stride: 16,
            lhs_dtype: MANDREL_VORTEX_DTYPE_I8,
            rhs_dtype: MANDREL_VORTEX_DTYPE_I8,
            out_dtype: MANDREL_VORTEX_DTYPE_I32,
            flags: 0,
        };

        assert!(!desc_is_supported(&desc));
    }

    #[test]
    fn backend_config_null_runtime_path_is_allowed() {
        let config = MandrelVortexBackendConfig {
            struct_size: size_of::<MandrelVortexBackendConfig>(),
            runtime_library_path: ptr::null(),
            kernel_vxbin_path: c"kernel.vxbin".as_ptr(),
            device_index: 0,
            flags: 0,
        };

        assert_eq!(config.device_index, 0);
        assert!(config.runtime_library_path.is_null());
    }
}
