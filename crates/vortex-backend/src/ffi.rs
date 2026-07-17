use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::fmt;
use std::mem::size_of;
use std::os::raw::c_char;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;

#[cfg(unix)]
use std::ffi::OsStr;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

use crate::backend::{VortexBackend, VortexBackendConfig};
use crate::vortex2::Runtime;

pub const MANDREL_VORTEX_STATUS_OK: i32 = 0;
pub const MANDREL_VORTEX_STATUS_NULL_POINTER: i32 = 1;
pub const MANDREL_VORTEX_STATUS_INVALID_ARGUMENT: i32 = 2;
pub const MANDREL_VORTEX_STATUS_RUNTIME_ERROR: i32 = 3;
pub const MANDREL_VORTEX_STATUS_PANIC: i32 = 255;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MandrelVortexBackendConfig {
    pub struct_size: usize,
    pub runtime_library_path: *const c_char,
    pub device_index: u32,
    pub flags: u32,
}

pub struct MandrelVortexBackend {
    _backend: VortexBackend,
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
        if config.flags != 0 {
            return fail_status(
                MANDREL_VORTEX_STATUS_INVALID_ARGUMENT,
                "MandrelVortexBackendConfig.flags is reserved and must be 0",
            );
        }

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

        let backend_config = VortexBackendConfig::new().with_device_index(config.device_index);
        let backend = match VortexBackend::from_runtime(runtime, backend_config) {
            Ok(backend) => backend,
            Err(error) => return fail_display(MANDREL_VORTEX_STATUS_RUNTIME_ERROR, error),
        };
        let boxed = Box::new(MandrelVortexBackend { _backend: backend });
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

fn ffi_status(function: impl FnOnce() -> i32) -> i32 {
    match panic::catch_unwind(AssertUnwindSafe(function)) {
        Ok(status) => status,
        Err(_) => fail_status(
            MANDREL_VORTEX_STATUS_PANIC,
            "panic crossed Vortex C ABI boundary",
        ),
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
        MANDREL_VORTEX_STATUS_INVALID_ARGUMENT, MandrelVortexBackend, MandrelVortexBackendConfig,
        mandrel_vortex_backend_create, mandrel_vortex_last_error_message,
    };
    use std::ffi::CStr;
    use std::mem::size_of;
    use std::ptr;

    #[test]
    fn ffi_backend_config_layout_has_stable_size_on_64_bit_host() {
        if size_of::<usize>() == 8 {
            assert_eq!(size_of::<MandrelVortexBackendConfig>(), 24);
        }
    }

    #[test]
    fn backend_config_null_runtime_path_is_allowed() {
        let config = MandrelVortexBackendConfig {
            struct_size: size_of::<MandrelVortexBackendConfig>(),
            runtime_library_path: ptr::null(),
            device_index: 0,
            flags: 0,
        };

        assert_eq!(config.device_index, 0);
        assert!(config.runtime_library_path.is_null());
    }

    #[test]
    fn backend_create_rejects_nonzero_reserved_flags_before_runtime_loading() {
        let config = MandrelVortexBackendConfig {
            struct_size: size_of::<MandrelVortexBackendConfig>(),
            runtime_library_path: ptr::null(),
            device_index: 0,
            flags: 1,
        };
        let mut backend: *mut MandrelVortexBackend = ptr::null_mut();

        // SAFETY: Both pointers reference valid storage for the duration of the call.
        let status = unsafe { mandrel_vortex_backend_create(&config, &mut backend) };

        assert_eq!(status, MANDREL_VORTEX_STATUS_INVALID_ARGUMENT);
        assert!(backend.is_null());
        // SAFETY: The FFI function returns a valid thread-local NUL-terminated string.
        let message = unsafe { CStr::from_ptr(mandrel_vortex_last_error_message()) };
        assert!(
            message
                .to_bytes()
                .windows(5)
                .any(|window| window == b"flags")
        );
    }
}
