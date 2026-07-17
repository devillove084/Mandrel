use std::env;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt::MakeWriter;

use crate::{Result, XtaskError};

#[derive(Clone, Copy)]
enum LogFormat {
    Compact,
    Pretty,
    Json,
}

impl LogFormat {
    fn from_env() -> Result<Self> {
        let raw = env::var("MANDREL_LOG_FORMAT").unwrap_or_else(|_| "compact".to_owned());
        match raw.as_str() {
            "compact" => Ok(Self::Compact),
            "pretty" => Ok(Self::Pretty),
            "json" => Ok(Self::Json),
            other => Err(XtaskError::message(format!(
                "unsupported MANDREL_LOG_FORMAT '{other}'; use compact, pretty, or json"
            ))),
        }
    }
}

pub(crate) fn init_logging() -> Result<()> {
    let filter = env::var("MANDREL_LOG").unwrap_or_else(|_| "info".to_owned());
    let filter =
        EnvFilter::try_new(filter).map_err(|error| format!("invalid MANDREL_LOG: {error}"))?;
    let format = LogFormat::from_env()?;

    if let Some(file_path) = env::var_os("MANDREL_LOG_FILE").map(PathBuf::from) {
        let parent = non_empty_parent(&file_path);
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create log directory '{}': {error}",
                parent.display()
            )
        })?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .map_err(|error| {
                format!("failed to open log file '{}': {error}", file_path.display())
            })?;
        install_subscriber(filter, format, Mutex::new(file), false)?;
        return Ok(());
    }

    install_subscriber(filter, format, io::stdout, true)
}

fn non_empty_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn install_subscriber<W>(filter: EnvFilter, format: LogFormat, writer: W, ansi: bool) -> Result<()>
where
    W: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
    match format {
        LogFormat::Compact => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            .with_ansi(ansi)
            .compact()
            .try_init(),
        LogFormat::Pretty => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            .with_ansi(ansi)
            .pretty()
            .try_init(),
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            .with_ansi(ansi)
            .json()
            .try_init(),
    }
    .map_err(|error| XtaskError::message(format!("failed to install tracing subscriber: {error}")))
}
