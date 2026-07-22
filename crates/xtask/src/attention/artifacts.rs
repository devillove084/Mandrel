use std::path::Path;

use mandrel_experiment::ExperimentSpec;
use mandrel_vortex_backend::{
    VortexConfig, VortexKernelBuildOutputs, VortexMlirKernelBuildRequest,
    build_vortex_mlir_kernel_artifacts,
};
use mandrel_vortex_codegen::{
    VORTEX_HARDWARE_IDENTITY_PROBE_SYMBOL, generate_vortex_attention_prefill_mlir,
    generate_vortex_hardware_identity_probe_mlir,
};

use crate::Result;
use crate::command::XtaskCommandRunner;

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
) -> Result<VortexKernelBuildOutputs> {
    let config = VortexConfig::from_env(workspace_root)?;
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
            let outputs = validate_mlir_with_vortex_llvm(
                workspace_root,
                &config,
                generated.symbol.as_str(),
                &generated.source,
                "attention",
            )?;
            println!("generated MLIR written to: {}", outputs.mlir_path.display());
            println!(
                "generated LLVM IR written to: {}",
                outputs.ll_path.display()
            );
            println!(
                "generated object written to: {}",
                outputs.obj_path.display()
            );
            println!("generated ELF written to: {}", outputs.elf_path.display());
            println!(
                "generated vxbin written to: {}",
                outputs.vxbin_path.display()
            );
            if print_source {
                println!("{}", generated.source);
            }
            Ok(outputs)
        }
        Err(error) => Err(error.into()),
    }
}

pub(super) fn generate_vortex_hardware_identity_probe_artifacts(
    workspace_root: &Path,
    config: &VortexConfig,
) -> Result<VortexKernelBuildOutputs> {
    let source = generate_vortex_hardware_identity_probe_mlir();
    let outputs = validate_mlir_with_vortex_llvm(
        workspace_root,
        config,
        VORTEX_HARDWARE_IDENTITY_PROBE_SYMBOL,
        &source,
        "hardware_identity",
    )?;
    println!(
        "generated hardware identity probe vxbin: {}",
        outputs.vxbin_path.display()
    );
    Ok(outputs)
}

fn validate_mlir_with_vortex_llvm(
    workspace_root: &Path,
    config: &VortexConfig,
    symbol_name: &str,
    source: &str,
    phase_prefix: &str,
) -> Result<VortexKernelBuildOutputs> {
    let outputs = VortexKernelBuildOutputs::under_output_dir(
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
            outputs: &outputs,
            phase_prefix,
        },
        &mut runner,
    )?;

    println!("validated LLVM IR: {}", outputs.ll_path.display());
    println!("validated object: {}", outputs.obj_path.display());
    println!("validated ELF: {}", outputs.elf_path.display());
    println!("validated vxbin: {}", outputs.vxbin_path.display());
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use mandrel_vortex_backend::vortex_kernel_entry_symbol;

    #[test]
    fn vortex_kentry_symbol_uses_runtime_lookup_prefix() {
        assert_eq!(
            vortex_kernel_entry_symbol("attention_prefill_i8"),
            "__vx_kentry_attention_prefill_i8"
        );
    }
}
