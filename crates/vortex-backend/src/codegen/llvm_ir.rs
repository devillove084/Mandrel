#[derive(Debug, Clone, Default)]
pub(crate) struct LlvmIrModule {
    source: String,
}

impl LlvmIrModule {
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

pub(crate) fn c_string_literal(value: &str) -> (usize, String) {
    let mut escaped = String::new();
    for byte in value.bytes() {
        match byte {
            b'\\' => escaped.push_str("\\5C"),
            b'\n' => escaped.push_str("\\0A"),
            b'\r' => escaped.push_str("\\0D"),
            b'\t' => escaped.push_str("\\09"),
            b'\"' => escaped.push_str("\\22"),
            0x20..=0x7e => escaped.push(char::from(byte)),
            _ => escaped.push_str(&format!("\\{byte:02X}")),
        }
    }
    escaped.push_str("\\00");
    (value.len() + 1, escaped)
}

pub(crate) fn emit_private_metadata_string(module: &mut LlvmIrModule, symbol: &str, value: &str) {
    let (len, literal) = c_string_literal(value);
    module.push_line(format!(
        "@{symbol} = private unnamed_addr constant [{len} x i8] c\"{literal}\", section \"llvm.metadata\""
    ));
}

#[cfg(test)]
mod tests {
    use super::c_string_literal;

    #[test]
    fn llvm_c_string_literal_appends_nul_and_escapes_special_bytes() {
        assert_eq!(
            c_string_literal("vortex.kernel"),
            (14, "vortex.kernel\\00".to_owned())
        );
        assert_eq!(c_string_literal("a\nb"), (4, "a\\0Ab\\00".to_owned()));
        assert_eq!(c_string_literal("quote\""), (7, "quote\\22\\00".to_owned()));
    }
}
