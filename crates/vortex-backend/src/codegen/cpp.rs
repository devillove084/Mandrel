use genco::lang::c;

use crate::codegen::VortexCodegenError;

pub(crate) type CppTokens = c::Tokens;

pub(crate) fn render_cpp(tokens: CppTokens) -> Result<String, VortexCodegenError> {
    tokens
        .to_file_string()
        .map_err(|_| VortexCodegenError::FormatFailed)
}
