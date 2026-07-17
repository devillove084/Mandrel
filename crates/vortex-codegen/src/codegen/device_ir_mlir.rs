#![allow(dead_code)]

use crate::codegen::mlir::{
    MlirModule, mlir_string_literal, mlir_string_literal_with_nul, mlir_symbol_ref,
};
use crate::codegen::vortex_ir::VortexCtaCsr;

use super::{
    IntegerOverflowSemantics, LlvmType, PhiIncoming, RISCV_CUSTOM0, TARGET_DATALAYOUT,
    TARGET_TRIPLE, VortexBlock, VortexDeviceModule, VortexKernelFunction, VortexOp,
    VortexTerminator,
};

pub(crate) fn render_vortex_device_module_mlir(module: &VortexDeviceModule) -> String {
    let mut mlir = MlirModule::new();

    emit_mlir_header(&mut mlir, module);
    emit_mlir_kernel_metadata(&mut mlir, module);
    if !module.kernels.is_empty() {
        mlir.blank_line();
    }
    emit_mlir_kernels(&mut mlir, module);
    emit_mlir_declarations(&mut mlir);
    emit_mlir_footer(&mut mlir);

    mlir.finish()
}

fn emit_mlir_header(mlir: &mut MlirModule, module: &VortexDeviceModule) {
    mlir.push_line(format!("// ModuleID = '{}'", module.module_id));
    mlir.push_line(format!(
        "// source_filename = \"{}\"",
        module.source_filename
    ));
    mlir.push_line("module attributes {");
    mlir.push_line(format!(
        "  llvm.data_layout = \"{}\",",
        mlir_string_literal(TARGET_DATALAYOUT)
    ));
    mlir.push_line(format!(
        "  llvm.target_triple = \"{}\"",
        mlir_string_literal(TARGET_TRIPLE)
    ));
    mlir.push_line("} {");
}

fn emit_mlir_kernel_metadata(mlir: &mut MlirModule, module: &VortexDeviceModule) {
    if module.kernels.is_empty() {
        return;
    }

    emit_mlir_metadata_string(mlir, ".mandrel.vortex.kernel", "vortex.kernel");
    emit_mlir_metadata_string(mlir, ".mandrel.vortex.source", &module.source_name);
    emit_mlir_global_annotations(mlir, module);
    emit_mlir_used_kernel_list(mlir, "llvm.used", module);
    emit_mlir_used_kernel_list(mlir, "llvm.compiler.used", module);
}

fn emit_mlir_metadata_string(mlir: &mut MlirModule, symbol: &str, value: &str) {
    let len = value.len() + 1;
    mlir.push_line(format!(
        "  llvm.mlir.global private unnamed_addr constant {}(\"{}\") {{section = \"llvm.metadata\"}} : !llvm.array<{len} x i8>",
        mlir_symbol_ref(symbol),
        mlir_string_literal_with_nul(value)
    ));
}

fn emit_mlir_global_annotations(mlir: &mut MlirModule, module: &VortexDeviceModule) {
    let count = module.kernels.len();
    let annotation_ty = "!llvm.struct<(ptr, ptr, ptr, i32, ptr)>";
    let annotations_ty = format!("!llvm.array<{count} x {annotation_ty}>");

    mlir.push_line(format!(
        "  llvm.mlir.global appending {}() {{section = \"llvm.metadata\"}} : {annotations_ty} {{",
        mlir_symbol_ref("llvm.global.annotations")
    ));
    mlir.push_line(format!(
        "    %annotations0 = llvm.mlir.undef : {annotations_ty}"
    ));
    mlir.push_line(format!(
        "    %annotation = llvm.mlir.addressof {} : !llvm.ptr",
        mlir_symbol_ref(".mandrel.vortex.kernel")
    ));
    mlir.push_line(format!(
        "    %source = llvm.mlir.addressof {} : !llvm.ptr",
        mlir_symbol_ref(".mandrel.vortex.source")
    ));
    mlir.push_line("    %annotation_line = llvm.mlir.constant(1 : i32) : i32");
    mlir.push_line("    %annotation_null = llvm.mlir.zero : !llvm.ptr");

    let mut current_array = "%annotations0".to_owned();
    for (index, kernel) in module.kernels.iter().enumerate() {
        let kernel_addr = format!("%annotation_kernel{index}");
        let entry0 = format!("%annotation_entry{index}_0");
        let entry1 = format!("%annotation_entry{index}_1");
        let entry2 = format!("%annotation_entry{index}_2");
        let entry3 = format!("%annotation_entry{index}_3");
        let entry4 = format!("%annotation_entry{index}_4");
        let entry5 = format!("%annotation_entry{index}_5");
        let next_array = format!("%annotations{}", index + 1);

        mlir.push_line(format!(
            "    {kernel_addr} = llvm.mlir.addressof {} : !llvm.ptr",
            mlir_symbol_ref(&kernel.name)
        ));
        mlir.push_line(format!("    {entry0} = llvm.mlir.undef : {annotation_ty}"));
        mlir.push_line(format!(
            "    {entry1} = llvm.insertvalue {kernel_addr}, {entry0}[0] : {annotation_ty}"
        ));
        mlir.push_line(format!(
            "    {entry2} = llvm.insertvalue %annotation, {entry1}[1] : {annotation_ty}"
        ));
        mlir.push_line(format!(
            "    {entry3} = llvm.insertvalue %source, {entry2}[2] : {annotation_ty}"
        ));
        mlir.push_line(format!(
            "    {entry4} = llvm.insertvalue %annotation_line, {entry3}[3] : {annotation_ty}"
        ));
        mlir.push_line(format!(
            "    {entry5} = llvm.insertvalue %annotation_null, {entry4}[4] : {annotation_ty}"
        ));
        mlir.push_line(format!(
            "    {next_array} = llvm.insertvalue {entry5}, {current_array}[{index}] : {annotations_ty}"
        ));
        current_array = next_array;
    }

    mlir.push_line(format!(
        "    llvm.return {current_array} : {annotations_ty}"
    ));
    mlir.push_line("  }");
}

fn emit_mlir_used_kernel_list(mlir: &mut MlirModule, symbol: &str, module: &VortexDeviceModule) {
    let count = module.kernels.len();
    let used_ty = format!("!llvm.array<{count} x ptr>");
    let prefix = sanitize_ssa_component(symbol);

    mlir.push_line(format!(
        "  llvm.mlir.global appending {}() {{section = \"llvm.metadata\"}} : {used_ty} {{",
        mlir_symbol_ref(symbol)
    ));
    mlir.push_line(format!("    %{prefix}_0 = llvm.mlir.undef : {used_ty}"));

    let mut current_array = format!("%{prefix}_0");
    for (index, kernel) in module.kernels.iter().enumerate() {
        let kernel_addr = format!("%{prefix}_kernel{index}");
        let next_array = format!("%{prefix}_{}", index + 1);
        mlir.push_line(format!(
            "    {kernel_addr} = llvm.mlir.addressof {} : !llvm.ptr",
            mlir_symbol_ref(&kernel.name)
        ));
        mlir.push_line(format!(
            "    {next_array} = llvm.insertvalue {kernel_addr}, {current_array}[{index}] : {used_ty}"
        ));
        current_array = next_array;
    }

    mlir.push_line(format!("    llvm.return {current_array} : {used_ty}"));
    mlir.push_line("  }");
}

fn emit_mlir_kernels(mlir: &mut MlirModule, module: &VortexDeviceModule) {
    for kernel in &module.kernels {
        emit_mlir_kernel(mlir, module, kernel);
        mlir.blank_line();
    }
}

fn emit_mlir_kernel(
    mlir: &mut MlirModule,
    module: &VortexDeviceModule,
    kernel: &VortexKernelFunction,
) {
    mlir.push_line(format!(
        "  llvm.func {}(%{}: {}) attributes {{dso_local, no_unwind, passthrough = [\"vortex-kernel\", [\"frame-pointer\", \"all\"], [\"no-trapping-math\", \"true\"], [\"target-cpu\", \"generic-rv64\"], [\"target-features\", \"+64bit,+a,+d,+f,+m,+relax,+xvortex,+zaamo,+zalrsc,+zicond,+zicsr,+zmmul\"]]}} {{",
        mlir_symbol_ref(&kernel.name),
        kernel.param_name,
        render_mlir_type(&kernel.param_type, module)
    ));
    for (index, block) in kernel.blocks.iter().enumerate() {
        emit_mlir_block(mlir, module, kernel, block, index == 0);
    }
    mlir.push_line("  }");
}

fn emit_mlir_block(
    mlir: &mut MlirModule,
    module: &VortexDeviceModule,
    kernel: &VortexKernelFunction,
    block: &VortexBlock,
    is_entry: bool,
) {
    let block_args = render_mlir_block_arguments(block, module);
    if !is_entry {
        if block_args.is_empty() {
            mlir.push_line(format!("  ^{}:", block.label));
        } else {
            mlir.push_line(format!("  ^{}({block_args}):", block.label));
        }
    }

    let mut emitter = MlirBlockEmitter::new(mlir, module, &block.label);
    for op in &block.ops {
        if matches!(op, VortexOp::Phi { .. }) {
            continue;
        }
        emitter.emit_op(op);
    }
    emitter.emit_terminator(&block.terminator, &kernel.blocks, &block.label);
    emitter.ir.blank_line();
}

fn render_mlir_block_arguments(block: &VortexBlock, module: &VortexDeviceModule) -> String {
    let mut rendered = String::new();
    for (index, (result, ty, _incoming)) in phi_ops(block).enumerate() {
        if index != 0 {
            rendered.push_str(", ");
        }
        rendered.push_str(result);
        rendered.push_str(": ");
        rendered.push_str(&render_mlir_type(ty, module));
    }
    rendered
}

struct MlirBlockEmitter<'a, 'b> {
    ir: &'b mut MlirModule,
    module: &'a VortexDeviceModule,
    value_prefix: String,
    const_counter: usize,
}

impl<'a, 'b> MlirBlockEmitter<'a, 'b> {
    fn new(ir: &'b mut MlirModule, module: &'a VortexDeviceModule, block_label: &str) -> Self {
        Self {
            ir,
            module,
            value_prefix: sanitize_ssa_component(block_label),
            const_counter: 0,
        }
    }

    fn emit_op(&mut self, op: &VortexOp) {
        match op {
            VortexOp::StructFieldPtr {
                result,
                struct_type,
                base,
                field,
            } => {
                let struct_ty = render_mlir_struct_type(struct_type, self.module);
                self.ir.push_line(format!(
                    "    {result} = llvm.getelementptr inbounds {base}[0, {field}] : (!llvm.ptr) -> !llvm.ptr, {struct_ty}"
                ));
            }
            VortexOp::Load {
                result,
                ty,
                ptr,
                align,
            } => self.ir.push_line(format!(
                "    {result} = llvm.load {ptr} {{alignment = {align} : i64}} : !llvm.ptr -> {}",
                render_mlir_type(ty, self.module)
            )),
            VortexOp::Store {
                ty,
                value,
                ptr,
                align,
            } => {
                let value = self.ensure_value(value, ty);
                self.ir.push_line(format!(
                    "    llvm.store {value}, {ptr} {{alignment = {align} : i64}} : {}, !llvm.ptr",
                    render_mlir_type(ty, self.module)
                ));
            }
            VortexOp::IntToPtr {
                result,
                int_ty,
                value,
            } => {
                let value = self.ensure_value(value, int_ty);
                self.ir.push_line(format!(
                    "    {result} = llvm.inttoptr {value} : {} to !llvm.ptr",
                    render_mlir_type(int_ty, self.module)
                ));
            }
            VortexOp::CtaCsrI32 {
                raw64,
                raw32,
                result,
                csr,
                uniform,
            } => self.emit_cta_csr_i32(raw64, raw32.as_deref(), result, *csr, *uniform),
            VortexOp::LocalMemPtr { raw64, result } => {
                self.emit_csr_read_i64(raw64, VortexCtaCsr::LocalMemAddr);
                self.ir.push_line(format!(
                    "    {result} = llvm.inttoptr {raw64} : i64 to !llvm.ptr"
                ));
            }
            VortexOp::Barrier {
                barrier_id,
                num_warps,
            } => {
                let barrier_id = self.ensure_value(barrier_id, &LlvmType::I32);
                let num_warps = self.ensure_value(num_warps, &LlvmType::I32);
                self.ir.push_line(format!(
                    "    llvm.inline_asm has_side_effects \".insn r {RISCV_CUSTOM0}, 4, 0, x0, $0, $1\", \"r,r,~{{memory}}\" {barrier_id}, {num_warps} : (i32, i32) -> ()"
                ));
            }
            VortexOp::Binary {
                result,
                op,
                ty,
                lhs,
                rhs,
                overflow,
            } => {
                let lhs = self.ensure_value(lhs, ty);
                let rhs = self.ensure_value(rhs, ty);
                self.ir.push_line(format!(
                    "    {result} = llvm.{} {lhs}, {rhs}{} : {}",
                    op.render(),
                    render_mlir_overflow_flags(*overflow),
                    render_mlir_type(ty, self.module)
                ));
            }
            VortexOp::FloatBinary {
                result,
                op,
                ty,
                lhs,
                rhs,
            } => {
                let lhs = self.ensure_value(lhs, ty);
                let rhs = self.ensure_value(rhs, ty);
                self.ir.push_line(format!(
                    "    {result} = llvm.{} {lhs}, {rhs} : {}",
                    op.render(),
                    render_mlir_type(ty, self.module)
                ));
            }
            VortexOp::Icmp {
                result,
                predicate,
                ty,
                lhs,
                rhs,
            } => {
                let lhs = self.ensure_value(lhs, ty);
                let rhs = self.ensure_value(rhs, ty);
                self.ir.push_line(format!(
                    "    {result} = llvm.icmp \"{}\" {lhs}, {rhs} : {}",
                    predicate.render(),
                    render_mlir_type(ty, self.module)
                ));
            }
            VortexOp::Fcmp {
                result,
                predicate,
                ty,
                lhs,
                rhs,
            } => {
                let lhs = self.ensure_value(lhs, ty);
                let rhs = self.ensure_value(rhs, ty);
                self.ir.push_line(format!(
                    "    {result} = llvm.fcmp \"{}\" {lhs}, {rhs} : {}",
                    predicate.render(),
                    render_mlir_type(ty, self.module)
                ));
            }
            VortexOp::Select {
                result,
                cond,
                ty,
                true_value,
                false_value,
            } => {
                let cond = self.ensure_value(cond, &LlvmType::I1);
                let true_value = self.ensure_value(true_value, ty);
                let false_value = self.ensure_value(false_value, ty);
                self.ir.push_line(format!(
                    "    {result} = llvm.select {cond}, {true_value}, {false_value} : i1, {}",
                    render_mlir_type(ty, self.module)
                ));
            }
            VortexOp::FloatUnaryIntrinsic {
                result,
                intrinsic,
                ty,
                value,
            } => {
                let value = self.ensure_value(value, ty);
                let rendered_ty = render_mlir_type(ty, self.module);
                self.ir.push_line(format!(
                    "    {result} = {}({value}) : ({rendered_ty}) -> {rendered_ty}",
                    intrinsic.render()
                ));
            }
            VortexOp::Phi { .. } => {}
            VortexOp::Gep {
                result,
                element_ty,
                base,
                index,
                inbounds,
            } => {
                let index = self.ensure_value(index, &LlvmType::I64);
                let inbounds_text = if *inbounds { " inbounds" } else { "" };
                self.ir.push_line(format!(
                    "    {result} = llvm.getelementptr{inbounds_text} {base}[{index}] : (!llvm.ptr, i64) -> !llvm.ptr, {}",
                    render_mlir_type(element_ty, self.module)
                ));
            }
            VortexOp::Cast {
                result,
                kind,
                from_ty,
                value,
                to_ty,
            } => {
                let value = self.ensure_value(value, from_ty);
                self.ir.push_line(format!(
                    "    {result} = llvm.{} {value} : {} to {}",
                    kind.render(),
                    render_mlir_type(from_ty, self.module),
                    render_mlir_type(to_ty, self.module)
                ));
            }
        }
    }

    fn emit_cta_csr_i32(
        &mut self,
        raw64: &str,
        raw32: Option<&str>,
        result: &str,
        csr: VortexCtaCsr,
        uniform: bool,
    ) {
        self.emit_csr_read_i64(raw64, csr);
        if uniform {
            let raw32 = raw32.unwrap_or(result);
            self.ir
                .push_line(format!("    {raw32} = llvm.trunc {raw64} : i64 to i32"));
            self.ir.push_line(format!(
                "    {result} = llvm.call {}({raw32}) : (i32) -> i32",
                mlir_symbol_ref("llvm.riscv.vx.uniform.i32.i32")
            ));
        } else {
            self.ir
                .push_line(format!("    {result} = llvm.trunc {raw64} : i64 to i32"));
        }
    }

    fn emit_csr_read_i64(&mut self, result: &str, csr: VortexCtaCsr) {
        let csr = self.ensure_value(&csr.encoded().to_string(), &LlvmType::I32);
        self.ir.push_line(format!(
            "    {result} = llvm.inline_asm has_side_effects \"csrr $0, $1\", \"=r,i\" {csr} : (i32) -> i64"
        ));
    }

    fn emit_terminator(
        &mut self,
        terminator: &VortexTerminator,
        blocks: &[VortexBlock],
        current_label: &str,
    ) {
        match terminator {
            VortexTerminator::Br { label } => {
                let successor = self.render_successor(blocks, current_label, label);
                self.ir.push_line(format!("    llvm.br {successor}"));
            }
            VortexTerminator::CondBr {
                cond,
                then_label,
                else_label,
            } => {
                let cond = self.ensure_value(cond, &LlvmType::I1);
                let then_successor = self.render_successor(blocks, current_label, then_label);
                let else_successor = self.render_successor(blocks, current_label, else_label);
                self.ir.push_line(format!(
                    "    llvm.cond_br {cond}, {then_successor}, {else_successor}"
                ));
            }
            VortexTerminator::RetVoid => self.ir.push_line("    llvm.return"),
        }
    }

    fn render_successor(
        &mut self,
        blocks: &[VortexBlock],
        current_label: &str,
        target_label: &str,
    ) -> String {
        let target = blocks
            .iter()
            .find(|block| block.label == target_label)
            .unwrap_or_else(|| panic!("unknown Vortex MLIR successor block: {target_label}"));
        let mut values = Vec::new();
        let mut types = Vec::new();

        for (_result, ty, incoming) in phi_ops(target) {
            let value = incoming
                .iter()
                .find(|item| item.label == current_label)
                .unwrap_or_else(|| {
                    panic!(
                        "missing Vortex MLIR phi incoming from {current_label} to {target_label}"
                    )
                });
            values.push(self.ensure_value(&value.value, ty));
            types.push(render_mlir_type(ty, self.module));
        }

        if values.is_empty() {
            format!("^{target_label}")
        } else {
            format!(
                "^{target_label}({} : {})",
                values.join(", "),
                types.join(", ")
            )
        }
    }

    fn ensure_value(&mut self, value: &str, ty: &LlvmType) -> String {
        if !is_literal_for_type(value, ty) {
            return value.to_owned();
        }

        let name = format!(
            "%{}_c{}_{}",
            self.value_prefix,
            self.const_counter,
            mlir_type_suffix(ty)
        );
        self.const_counter += 1;
        self.ir.push_line(format!(
            "    {name} = llvm.mlir.constant({value} : {}) : {}",
            render_mlir_constant_attr_type(ty),
            render_mlir_type(ty, self.module)
        ));
        name
    }
}

fn emit_mlir_declarations(mlir: &mut MlirModule) {
    mlir.push_line(format!(
        "  llvm.func {}(i32) -> i32 attributes {{no_unwind}}",
        mlir_symbol_ref("llvm.riscv.vx.uniform.i32.i32")
    ));
}

fn emit_mlir_footer(mlir: &mut MlirModule) {
    mlir.push_line("}");
}

fn phi_ops(block: &VortexBlock) -> impl Iterator<Item = (&String, &LlvmType, &Vec<PhiIncoming>)> {
    block.ops.iter().filter_map(|op| match op {
        VortexOp::Phi {
            result,
            ty,
            incoming,
        } => Some((result, ty, incoming)),
        _ => None,
    })
}

fn is_literal_for_type(value: &str, ty: &LlvmType) -> bool {
    match ty {
        LlvmType::F32 => is_float_literal(value),
        _ => is_integer_literal(value),
    }
}

fn is_integer_literal(value: &str) -> bool {
    !value.starts_with('%') && value.parse::<i64>().is_ok()
}

fn is_float_literal(value: &str) -> bool {
    !value.starts_with('%')
        && (value.contains('.') || value.contains('e') || value.contains('E'))
        && value.parse::<f32>().is_ok()
}

fn render_mlir_type(ty: &LlvmType, module: &VortexDeviceModule) -> String {
    match ty {
        LlvmType::I1 => "i1".to_owned(),
        LlvmType::I8 => "i8".to_owned(),
        LlvmType::I32 => "i32".to_owned(),
        LlvmType::I64 => "i64".to_owned(),
        LlvmType::F32 => "f32".to_owned(),
        LlvmType::Ptr => "!llvm.ptr".to_owned(),
        LlvmType::Struct(name) => render_mlir_struct_type(name, module),
    }
}

fn render_mlir_struct_type(name: &str, module: &VortexDeviceModule) -> String {
    if let Some(struct_type) = module.structs.iter().find(|item| item.name == name) {
        let fields = render_mlir_type_list(&struct_type.fields, module);
        format!(
            "!llvm.struct<\"{}\", ({fields})>",
            mlir_string_literal(name)
        )
    } else {
        format!("!llvm.struct<\"{}\">", mlir_string_literal(name))
    }
}

fn render_mlir_type_list(types: &[LlvmType], module: &VortexDeviceModule) -> String {
    let mut rendered = String::new();
    for (index, ty) in types.iter().enumerate() {
        if index != 0 {
            rendered.push_str(", ");
        }
        rendered.push_str(&render_mlir_type(ty, module));
    }
    rendered
}

fn render_mlir_constant_attr_type(ty: &LlvmType) -> &'static str {
    match ty {
        LlvmType::I1 => "i1",
        LlvmType::I8 => "i8",
        LlvmType::I32 => "i32",
        LlvmType::I64 => "i64",
        LlvmType::F32 => "f32",
        LlvmType::Ptr | LlvmType::Struct(_) => "i64",
    }
}

fn mlir_type_suffix(ty: &LlvmType) -> &'static str {
    match ty {
        LlvmType::I1 => "i1",
        LlvmType::I8 => "i8",
        LlvmType::I32 => "i32",
        LlvmType::I64 => "i64",
        LlvmType::F32 => "f32",
        LlvmType::Ptr => "ptr",
        LlvmType::Struct(_) => "struct",
    }
}

fn render_mlir_overflow_flags(overflow: IntegerOverflowSemantics) -> &'static str {
    match overflow {
        IntegerOverflowSemantics::None => "",
        IntegerOverflowSemantics::Nsw => " overflow<nsw>",
    }
}

fn sanitize_ssa_component(value: &str) -> String {
    let mut sanitized = String::from("v");
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    sanitized
}
