use std::io;
use std::process::ExitStatus;

use mandrel_compiler::CompileError;
use mandrel_vortex_backend::{VortexBackendError, VortexToolchainError};
use mandrel_vortex_codegen::{AttentionPlanValidationError, VortexCodegenError};
use snafu::Snafu;

pub(crate) type Result<T> = std::result::Result<T, XtaskError>;

#[derive(Debug, Snafu)]
pub(crate) enum XtaskError {
    #[snafu(display("{message}"))]
    Message { message: String },
    #[snafu(display("failed to spawn {phase}: {source}"))]
    CommandSpawn { phase: String, source: io::Error },
    #[snafu(display("{phase} failed with status: {status}; command: {command}"))]
    CommandFailed {
        phase: String,
        status: ExitStatus,
        command: String,
    },
    #[snafu(display("{phase} failed with status: {status}; command: {command}; stderr: {stderr}"))]
    CommandFailedWithStderr {
        phase: String,
        status: ExitStatus,
        command: String,
        stderr: String,
    },
    #[snafu(display("Vortex toolchain error: {source}"))]
    VortexToolchain { source: VortexToolchainError },
    #[snafu(display("Vortex backend error: {source}"))]
    VortexBackend { source: VortexBackendError },
    #[snafu(display("attention plan validation error: {source}"))]
    AttentionPlanValidation {
        source: AttentionPlanValidationError,
    },
    #[snafu(display("Vortex codegen error: {source}"))]
    VortexCodegen { source: VortexCodegenError },
    #[snafu(display("compile error: {source}"))]
    Compile { source: CompileError },
}

impl XtaskError {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
        }
    }
}

impl From<String> for XtaskError {
    fn from(message: String) -> Self {
        Self::message(message)
    }
}

impl From<&str> for XtaskError {
    fn from(message: &str) -> Self {
        Self::message(message)
    }
}

impl From<VortexToolchainError> for XtaskError {
    fn from(source: VortexToolchainError) -> Self {
        Self::VortexToolchain { source }
    }
}

impl From<VortexBackendError> for XtaskError {
    fn from(source: VortexBackendError) -> Self {
        Self::VortexBackend { source }
    }
}

impl From<AttentionPlanValidationError> for XtaskError {
    fn from(source: AttentionPlanValidationError) -> Self {
        Self::AttentionPlanValidation { source }
    }
}

impl From<VortexCodegenError> for XtaskError {
    fn from(source: VortexCodegenError) -> Self {
        Self::VortexCodegen { source }
    }
}

impl From<CompileError> for XtaskError {
    fn from(source: CompileError) -> Self {
        Self::Compile { source }
    }
}
