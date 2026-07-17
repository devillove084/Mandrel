#![allow(dead_code)]

#[derive(Debug, Clone, Default)]
pub(crate) struct MlirModule {
    source: String,
}

impl MlirModule {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push_line(&mut self, line: impl AsRef<str>) {
        self.source.push_str(line.as_ref());
        self.source.push('\n');
    }

    pub(crate) fn blank_line(&mut self) {
        self.source.push('\n');
    }

    pub(crate) fn finish(self) -> String {
        self.source
    }
}

pub(crate) fn mlir_string_literal(value: &str) -> String {
    let mut escaped = String::new();
    for byte in value.bytes() {
        match byte {
            b'\\' => escaped.push_str("\\\\"),
            b'\"' => escaped.push_str("\\\""),
            b'\n' => escaped.push_str("\\0A"),
            b'\r' => escaped.push_str("\\0D"),
            b'\t' => escaped.push_str("\\09"),
            0x20..=0x7e => escaped.push(char::from(byte)),
            _ => escaped.push_str(&format!("\\{byte:02X}")),
        }
    }
    escaped
}

pub(crate) fn mlir_string_literal_with_nul(value: &str) -> String {
    let mut escaped = mlir_string_literal(value);
    escaped.push_str("\\00");
    escaped
}

pub(crate) fn mlir_symbol_ref(name: &str) -> String {
    if is_bare_symbol(name) {
        format!("@{name}")
    } else {
        format!("@\"{}\"", mlir_string_literal(name))
    }
}

fn is_bare_symbol(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first == '$' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::{mlir_string_literal, mlir_string_literal_with_nul, mlir_symbol_ref};

    #[test]
    fn escapes_mlir_strings_and_quotes_non_bare_symbols() {
        assert_eq!(mlir_string_literal("vortex.kernel"), "vortex.kernel");
        assert_eq!(mlir_string_literal("a\nb"), "a\\0Ab");
        assert_eq!(mlir_string_literal("quote\""), "quote\\\"");
        assert_eq!(
            mlir_string_literal_with_nul("vortex.kernel"),
            "vortex.kernel\\00"
        );
        assert_eq!(
            mlir_symbol_ref("attention_prefill_i8"),
            "@attention_prefill_i8"
        );
        assert_eq!(
            mlir_symbol_ref(".mandrel.vortex.kernel"),
            "@\".mandrel.vortex.kernel\""
        );
    }
}
