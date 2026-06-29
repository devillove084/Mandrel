use super::*;

pub(crate) fn print_vortex_status(workspace_root: &Path) -> Result<()> {
    let config = VortexConfig::from_env(workspace_root)?;
    let status = VortexStatus::probe(&config);

    println!("Vortex source dir:       {}", config.source_dir.display());
    println!("Vortex build dir:        {}", config.build_dir.display());
    println!(
        "Vortex install dir:      {}",
        config.install_dir().display()
    );
    println!("Vortex tool dir:         {}", config.tool_dir.display());
    println!("Vortex env file:         {}", config.env_file.display());
    println!(
        "download proxy prefix:   {}",
        config.download_proxy_prefix.as_deref().unwrap_or("<unset>")
    );
    println!(
        "git proxy prefix:        {}",
        config.git_proxy_prefix.as_deref().unwrap_or("<unset>")
    );
    println!("fetch_retries:           {}", config.fetch_retries);
    println!(
        "toolchain_mode:          {}",
        config.toolchain_mode.as_str()
    );
    println!("checkout_exists:         {}", status.checkout_exists);
    println!("build_dir_exists:        {}", status.build_dir_exists);
    println!("install_dir_exists:      {}", status.install_dir_exists);
    println!("env_file_exists:         {}", status.env_file_exists);
    println!(
        "download_wrapper_dir:    {}",
        status.download_wrapper_dir_exists
    );
    println!("blackbox_script_exists:  {}", status.blackbox_script_exists);
    println!("simx_binary_exists:     {}", status.simx_binary_exists);
    println!(
        "runtime_pkg_config:      {}",
        status.runtime_pkg_config_exists
    );
    println!(
        "kernel_pkg_config:       {}",
        status.kernel_pkg_config_exists
    );
    println!("can_run_blackbox:        {}", status.can_run_blackbox());
    println!("can_run_simx:            {}", status.can_run_simx());
    println!(
        "can_use_installed_sdk:   {}",
        status.can_use_installed_runtime()
    );
    Ok(())
}
