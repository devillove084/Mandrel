use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use mandrel_vortex_backend::{
    DEFAULT_VORTEX_SYSTEM_TOOLDIR, VortexConfig, VortexStatus, VortexToolchainMode,
};
use tracing::{info, warn};

use crate::command::{
    run_checked, run_checked_with_retries, run_output_checked, shell_quote_lossy,
};
use crate::{Result, XtaskError};

mod environment;
mod install;
mod runtime;
mod source_toolchain;
mod status;
mod system_tools;

pub(crate) use environment::{apply_vortex_env, write_and_print_vortex_env};
pub(crate) use install::{fetch_vortex, install_vortex};
pub(crate) use runtime::{
    ensure_vortex_runtime_libraries, preferred_vortex_runtime_library, require_file,
    run_vortex_vecadd,
};
pub(crate) use source_toolchain::install_vortex_source_toolchain;
pub(crate) use status::print_vortex_status;
pub(crate) use system_tools::{
    prepare_and_print_vortex_system_tools, reject_obvious_incompatible_prebuilt_tools,
};

use environment::*;
use install::*;
use system_tools::*;

const GITHUB_REWRITE_PATTERNS: [&str; 4] = [
    "https://github.com/",
    "http://github.com/",
    "git@github.com:",
    "ssh://git@github.com/",
];

const DEFAULT_VORTEX_SOURCE_TOOLDIR: &str = "external/vortex-source-tools";
const DEFAULT_VORTEX_LLVM_DIR: &str = "external/llvm-vortex";
const DEFAULT_VORTEX_LLVM_BUILD_DIR: &str = "external/llvm-vortex-build";
const DEFAULT_VORTEX_COMPILER_RT_BUILD_DIR: &str = "external/llvm-vortex-compiler-rt-build64";
const DEFAULT_VORTEX_LLVM_URL: &str = "https://github.com/devillove084/llvm.git";
const DEFAULT_VORTEX_LLVM_REF: &str = "vortex_3.x";
const DEFAULT_VORTEX_LLVM_PROJECTS: &str = "clang;lld;mlir";
