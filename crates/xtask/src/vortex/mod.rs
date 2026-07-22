mod environment;
mod runtime;

pub(crate) use environment::apply_vortex_env;
pub(crate) use runtime::{
    preferred_vortex_runtime_library, require_file, require_vortex_runtime_libraries,
};
