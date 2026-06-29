use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionOutputComparison {
    pub(crate) elements: usize,
    pub(crate) mismatches: usize,
}

pub(crate) fn compare_attention_outputs(
    expected: &[i8],
    actual: &[i8],
) -> Result<AttentionOutputComparison> {
    if expected.len() != actual.len() {
        return Err(format!(
            "attention output length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        )
        .into());
    }

    let mut mismatch_count = 0usize;
    let mut first_mismatches = Vec::new();
    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual).enumerate() {
        if expected_value != actual_value {
            mismatch_count += 1;
            if first_mismatches.len() < 16 {
                first_mismatches.push(format!(
                    "  index {index}: expected {expected_value}, got {actual_value}"
                ));
            }
        }
    }

    let summary = AttentionOutputComparison {
        elements: expected.len(),
        mismatches: mismatch_count,
    };
    if mismatch_count == 0 {
        return Ok(summary);
    }

    Err(format!(
        "attention output mismatch: {mismatch_count}/{} elements differ\n{}",
        expected.len(),
        first_mismatches.join("\n")
    )
    .into())
}
