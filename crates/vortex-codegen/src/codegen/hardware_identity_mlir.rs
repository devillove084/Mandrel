use crate::codegen::device_ir::{
    LlvmType, VortexBlock, VortexDeviceModule, VortexKernelFunction, VortexOp, VortexStructType,
    VortexTerminator, render_vortex_device_module_mlir,
};

pub const VORTEX_HARDWARE_IDENTITY_PROBE_SYMBOL: &str = "mandrel_hardware_identity_probe";
pub const VORTEX_MANDREL_CONFIG_ID_CSR: u16 = 0xFC5;

const HARDWARE_IDENTITY_PROBE_ARGS_STRUCT: &str = "mandrel_hardware_identity_probe_args_t";
const MLIR_SOURCE_NAME: &str = "mandrel generated Vortex hardware identity probe MLIR";

pub fn generate_vortex_hardware_identity_probe_mlir() -> String {
    let mut module = VortexDeviceModule::new(
        "mandrel generated vortex hardware identity probe",
        "mandrel://generated/vortex/hardware_identity_probe.mlir",
        MLIR_SOURCE_NAME,
    );
    module.add_struct(VortexStructType::new(
        HARDWARE_IDENTITY_PROBE_ARGS_STRUCT,
        vec![LlvmType::I64],
    ));

    let mut entry = VortexBlock::new("entry", VortexTerminator::RetVoid);
    entry.push_op(VortexOp::struct_field_ptr(
        "%out_addr_ptr",
        HARDWARE_IDENTITY_PROBE_ARGS_STRUCT,
        "%arg",
        0,
    ));
    entry.push_op(VortexOp::load(
        "%out_addr",
        LlvmType::I64,
        "%out_addr_ptr",
        8,
    ));
    entry.push_op(VortexOp::inttoptr("%out_ptr", LlvmType::I64, "%out_addr"));
    entry.push_op(VortexOp::csr_read_i64(
        "%config_id",
        VORTEX_MANDREL_CONFIG_ID_CSR,
    ));
    entry.push_op(VortexOp::store(LlvmType::I64, "%config_id", "%out_ptr", 8));

    let mut kernel =
        VortexKernelFunction::new(VORTEX_HARDWARE_IDENTITY_PROBE_SYMBOL, "arg", LlvmType::Ptr);
    kernel.push_block(entry);
    module.add_kernel(kernel);

    render_vortex_device_module_mlir(&module)
}

#[cfg(test)]
mod tests {
    use super::{
        VORTEX_HARDWARE_IDENTITY_PROBE_SYMBOL, VORTEX_MANDREL_CONFIG_ID_CSR,
        generate_vortex_hardware_identity_probe_mlir,
    };

    #[test]
    fn generates_discoverable_hardware_identity_probe_mlir() {
        let mlir = generate_vortex_hardware_identity_probe_mlir();

        assert_eq!(
            VORTEX_HARDWARE_IDENTITY_PROBE_SYMBOL,
            "mandrel_hardware_identity_probe"
        );
        assert_eq!(VORTEX_MANDREL_CONFIG_ID_CSR, 0xFC5);
        assert!(mlir.contains("llvm.func @mandrel_hardware_identity_probe(%arg: !llvm.ptr)"));
        assert!(mlir.contains("!llvm.struct<\"mandrel_hardware_identity_probe_args_t\", (i64)>"));
        assert!(mlir.contains("llvm.mlir.constant(4037 : i32) : i32"));
        assert!(mlir.contains("llvm.inline_asm has_side_effects \"csrr $0, $1\", \"=r,i\""));
        assert!(
            mlir.contains("llvm.store %config_id, %out_ptr {alignment = 8 : i64} : i64, !llvm.ptr")
        );
        assert!(mlir.contains("llvm.target_triple = \"riscv64-unknown-unknown-elf\""));
        assert!(mlir.contains("[\"target-cpu\", \"generic-rv64\"]"));
        assert!(mlir.contains("[\"target-features\", \"+64bit,+d,+f,+m,+relax,+xvortex"));
        assert!(mlir.contains("llvm.mlir.global appending @\"llvm.global.annotations\""));
        assert!(mlir.contains("llvm.mlir.global appending @\"llvm.used\""));
        assert!(mlir.contains("vortex-kernel"));
    }
}
