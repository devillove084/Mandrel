use std::ffi::OsStr;
use std::io::{self, BufRead, Write};
use std::process::{Command, Output, Stdio};

use mandrel_vortex_backend::{VortexCommandRunner, VortexToolchainError, VortexToolchainResult};
use tracing::{info, warn};

use crate::{Result, XtaskError};

pub(crate) const LOG_COMMAND_MAX_CHARS: usize = 4096;

pub(crate) struct XtaskCommandRunner;

impl VortexCommandRunner for XtaskCommandRunner {
    fn run(&mut self, phase: &str, command: Command) -> VortexToolchainResult<()> {
        run_checked(command, phase)
            .map_err(|error| VortexToolchainError::command_runner(phase, error.to_string()))
    }

    fn output(&mut self, phase: &str, command: Command) -> VortexToolchainResult<Output> {
        run_output_checked(command, phase)
            .map_err(|error| VortexToolchainError::command_runner(phase, error.to_string()))
    }
}

pub(crate) fn run_checked(mut command: Command, phase: &str) -> Result<()> {
    let rendered = render_command(&command);
    let logged = truncate_command_for_log(&rendered);
    info!(phase, command = %logged, "running command");
    let status = command
        .status()
        .map_err(|source| XtaskError::CommandSpawn {
            phase: phase.to_owned(),
            source,
        })?;

    if status.success() {
        info!(phase, status = %status, "command completed");
        Ok(())
    } else {
        Err(XtaskError::CommandFailed {
            phase: phase.to_owned(),
            status,
            command: rendered,
        })
    }
}

pub(crate) fn run_checked_capturing_stdout(mut command: Command, phase: &str) -> Result<String> {
    let rendered = render_command(&command);
    let logged = truncate_command_for_log(&rendered);
    info!(phase, command = %logged, "running command with stdout capture");
    command.stdout(Stdio::piped());
    let mut child = command.spawn().map_err(|source| XtaskError::CommandSpawn {
        phase: phase.to_owned(),
        source,
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        XtaskError::message(format!(
            "failed to capture stdout for phase '{phase}' after spawning command"
        ))
    })?;
    let mut reader = io::BufReader::new(stdout);
    let mut captured = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|error| {
            XtaskError::message(format!(
                "failed to read stdout for phase '{phase}': {error}"
            ))
        })?;
        if bytes == 0 {
            break;
        }
        print!("{line}");
        io::stdout().flush().map_err(|error| {
            XtaskError::message(format!(
                "failed to flush stdout for phase '{phase}': {error}"
            ))
        })?;
        captured.push_str(&line);
    }

    let status = child.wait().map_err(|source| XtaskError::CommandSpawn {
        phase: phase.to_owned(),
        source,
    })?;
    if status.success() {
        info!(phase, status = %status, "command completed");
        Ok(captured)
    } else {
        Err(XtaskError::CommandFailed {
            phase: phase.to_owned(),
            status,
            command: rendered,
        })
    }
}

pub(crate) fn run_output_checked(mut command: Command, phase: &str) -> Result<Output> {
    let rendered = render_command(&command);
    let logged = truncate_command_for_log(&rendered);
    info!(phase, command = %logged, "running command with captured output");
    let output = command
        .output()
        .map_err(|source| XtaskError::CommandSpawn {
            phase: phase.to_owned(),
            source,
        })?;

    if output.status.success() {
        info!(phase, status = %output.status, "command completed");
        Ok(output)
    } else {
        Err(XtaskError::CommandFailedWithStderr {
            phase: phase.to_owned(),
            status: output.status,
            command: rendered,
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

pub(crate) fn run_checked_with_retries<F>(
    mut make_command: F,
    phase: &str,
    attempts: u32,
) -> Result<()>
where
    F: FnMut() -> Result<Command>,
{
    let attempts = attempts.max(1);
    for attempt in 1..=attempts {
        match run_checked(make_command()?, phase) {
            Ok(()) => return Ok(()),
            Err(error) if attempt < attempts => {
                warn!(
                    phase,
                    attempt,
                    attempts,
                    error = %error,
                    "command failed; retrying"
                );
                eprintln!("{phase} failed on attempt {attempt}/{attempts}; retrying: {error}");
            }
            Err(error) => return Err(error),
        }
    }

    Err(XtaskError::message(format!("{phase} did not run")))
}

pub(crate) fn render_command(command: &Command) -> String {
    let mut parts = Vec::new();
    parts.push(shell_quote_lossy(command.get_program()));
    parts.extend(command.get_args().map(shell_quote_lossy));
    parts.join(" ")
}

pub(crate) fn truncate_command_for_log(command: &str) -> String {
    if command.len() <= LOG_COMMAND_MAX_CHARS {
        return command.to_owned();
    }

    let end = command
        .char_indices()
        .map(|(idx, _)| idx)
        .take_while(|idx| *idx <= LOG_COMMAND_MAX_CHARS)
        .last()
        .unwrap_or(0);
    format!(
        "{} ... <truncated: {} chars total>",
        &command[..end],
        command.len()
    )
}

pub(crate) fn shell_quote_lossy(value: &OsStr) -> String {
    let text = value.to_string_lossy();
    let mut quoted = String::from("'");
    for character in text.chars() {
        if character == '\'' {
            quoted.push_str("'\\''");
        } else {
            quoted.push(character);
        }
    }
    quoted.push('\'');
    quoted
}

#[cfg(test)]
mod tests {
    use super::{LOG_COMMAND_MAX_CHARS, truncate_command_for_log};

    #[test]
    fn command_log_truncation_preserves_short_commands() {
        let command = "'clang' '-c' 'attention.ll' '-o' 'attention.o'";

        assert_eq!(truncate_command_for_log(command), command);
    }

    #[test]
    fn command_log_truncation_marks_long_commands_without_splitting_utf8() {
        let command = format!("{}🚀{}", "a".repeat(LOG_COMMAND_MAX_CHARS), "b".repeat(32));
        let truncated = truncate_command_for_log(&command);

        assert!(truncated.contains("<truncated:"));
        assert!(truncated.ends_with(&format!("{} chars total>", command.len())));
        assert!(truncated.is_char_boundary(truncated.len()));
    }
}
