use std::fs;
use std::io;
use std::path::Path;

use tracing::warn;

use super::*;

pub(crate) fn print_attention_trace_history(path: &Path, limit: usize) -> io::Result<()> {
    let records = load_attention_trace_history(path)?;
    println!("Attention runtime trace history: {}", path.display());
    if records.is_empty() {
        println!("  no trace records found");
        return Ok(());
    }

    let limit = limit.max(1).min(records.len());
    println!("records: {} (showing latest {limit})", records.len());
    for (offset, record) in records.iter().rev().take(limit).enumerate() {
        let label = if offset == 0 {
            "latest".to_owned()
        } else {
            format!("latest-{offset}")
        };
        print_compact_trace_record(&label, *record);
    }

    if records.len() >= 2 {
        let latest = records[records.len() - 1];
        let previous = records[records.len() - 2];
        print_trace_delta(previous, latest);
    }

    Ok(())
}

pub(super) fn load_attention_trace_history(
    path: &Path,
) -> io::Result<Vec<AttentionRuntimeTraceRecord>> {
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut records = Vec::new();
    for (line_index, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match parse_attention_runtime_trace_jsonl(trimmed) {
            Some(record) => records.push(record),
            None => warn!(
                line = line_index + 1,
                "skipping malformed attention trace JSONL record"
            ),
        }
    }
    Ok(records)
}
