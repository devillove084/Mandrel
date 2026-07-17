use std::path::Path;

use mandrel_artifact::VortexMlirKernelArtifacts;
use mandrel_experiment::ExperimentSpec;
use mandrel_vortex_backend::{
    VortexConfig, VortexMlirKernelBuildRequest, VortexToolchainMode,
    build_vortex_mlir_kernel_artifacts,
};
use mandrel_vortex_codegen::generate_vortex_attention_prefill_mlir;

use crate::Result;
use crate::command::XtaskCommandRunner;
use crate::vortex::reject_obvious_incompatible_prebuilt_tools;

use super::plan::{
    current_attention_experiment_spec, current_attention_prefill_plan, print_attention_plan,
};

pub(crate) fn generate_vortex_attention_kernel_source(workspace_root: &Path) -> Result<()> {
    let spec = current_attention_experiment_spec()?;
    generate_vortex_attention_artifacts(workspace_root, spec, true).map(|_| ())
}

pub(super) fn generate_vortex_attention_artifacts(
    workspace_root: &Path,
    spec: ExperimentSpec,
    print_source: bool,
) -> Result<VortexMlirKernelArtifacts> {
    let config = VortexConfig::from_env(workspace_root)?;
    if config.toolchain_mode == VortexToolchainMode::Skip {
        reject_obvious_incompatible_prebuilt_tools(&config)?;
    }

    let plan = current_attention_prefill_plan(spec)?;
    print_attention_plan("Vortex attention-prefill MLIR dispatch", &plan);
    match generate_vortex_attention_prefill_mlir(&plan) {
        Ok(generated) => {
            println!(
                "generated kernel: {} ({:?}) format={} ext=.{} headers={:?}",
                generated.symbol.as_str(),
                generated.implementation,
                generated.format.as_str(),
                generated.format.extension(),
                generated.required_headers
            );
            let artifacts = validate_attention_mlir_with_vortex_llvm(
                workspace_root,
                &config,
                generated.symbol.as_str(),
                &generated.source,
            )?;
            println!(
                "generated MLIR written to: {}",
                artifacts.mlir_path.display()
            );
            println!(
                "generated LLVM IR written to: {}",
                artifacts.ll_path.display()
            );
            println!(
                "generated object written to: {}",
                artifacts.obj_path.display()
            );
            println!("generated ELF written to: {}", artifacts.elf_path.display());
            println!(
                "generated vxbin written to: {}",
                artifacts.vxbin_path.display()
            );
            if print_source {
                println!("{}", generated.source);
            }
            Ok(artifacts)
        }
        Err(error) => Err(error.into()),
    }
}

fn validate_attention_mlir_with_vortex_llvm(
    workspace_root: &Path,
    config: &VortexConfig,
    symbol_name: &str,
    source: &str,
) -> Result<VortexMlirKernelArtifacts> {
    let artifacts = VortexMlirKernelArtifacts::under_output_dir(
        &workspace_root.join("target/mandrel/vortex"),
        symbol_name,
    );
    let mut runner = XtaskCommandRunner;
    build_vortex_mlir_kernel_artifacts(
        VortexMlirKernelBuildRequest {
            workspace_root,
            config,
            symbol_name,
            source,
            artifacts: &artifacts,
            phase_prefix: "attention",
        },
        &mut runner,
    )?;

    println!("validated LLVM IR: {}", artifacts.ll_path.display());
    println!("validated object: {}", artifacts.obj_path.display());
    println!("validated ELF: {}", artifacts.elf_path.display());
    println!("validated vxbin: {}", artifacts.vxbin_path.display());
    Ok(artifacts)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use mandrel_artifact::VortexMlirKernelArtifacts;
    use mandrel_vortex_backend::vortex_kernel_entry_symbol;

    #[test]
    fn attention_artifact_paths_follow_kernel_symbol() {
        let out_dir = Path::new("target/mandrel/vortex");
        let artifacts =
            VortexMlirKernelArtifacts::under_output_dir(out_dir, "attention_prefill_i8");

        assert_eq!(
            artifacts.mlir_path,
            out_dir.join("attention_prefill_i8.mlir")
        );
        assert_eq!(artifacts.ll_path, out_dir.join("attention_prefill_i8.ll"));
        assert_eq!(artifacts.obj_path, out_dir.join("attention_prefill_i8.o"));
        assert_eq!(
            artifacts.startup_probe_elf_path,
            out_dir.join("attention_prefill_i8.startup_probe.elf")
        );
        assert_eq!(
            artifacts.startup_object_path,
            out_dir.join("attention_prefill_i8.vx_start.o")
        );
        assert_eq!(artifacts.elf_path, out_dir.join("attention_prefill_i8.elf"));
        assert_eq!(
            artifacts.vxbin_path,
            out_dir.join("attention_prefill_i8.vxbin")
        );
    }

    #[test]
    fn vortex_kentry_symbol_uses_runtime_lookup_prefix() {
        assert_eq!(
            vortex_kernel_entry_symbol("attention_prefill_i8"),
            "__vx_kentry_attention_prefill_i8"
        );
    }
}
