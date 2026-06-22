use crate::codegen::llvm_ir::{LlvmIrModule, emit_private_metadata_string};
use crate::codegen::vortex_ir::VortexCtaCsr;

const TARGET_DATALAYOUT: &str = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
const TARGET_TRIPLE: &str = "riscv64-unknown-unknown-elf";
const VORTEX_UNIFORM_INTRINSIC: &str = "@llvm.riscv.vx.uniform.i32.i32";
const RISCV_CUSTOM0: u32 = 0x0B;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexDeviceModule {
    module_id: String,
    source_filename: String,
    source_name: String,
    structs: Vec<VortexStructType>,
    kernels: Vec<VortexKernelFunction>,
}

impl VortexDeviceModule {
    pub fn new(
        module_id: impl Into<String>,
        source_filename: impl Into<String>,
        source_name: impl Into<String>,
    ) -> Self {
        Self {
            module_id: module_id.into(),
            source_filename: source_filename.into(),
            source_name: source_name.into(),
            structs: Vec::new(),
            kernels: Vec::new(),
        }
    }

    pub fn add_struct(&mut self, struct_type: VortexStructType) {
        self.structs.push(struct_type);
    }

    pub fn add_kernel(&mut self, kernel: VortexKernelFunction) {
        self.kernels.push(kernel);
    }

    pub fn kernels(&self) -> &[VortexKernelFunction] {
        &self.kernels
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexStructType {
    name: String,
    fields: Vec<LlvmType>,
}

impl VortexStructType {
    pub fn new(name: impl Into<String>, fields: Vec<LlvmType>) -> Self {
        Self {
            name: name.into(),
            fields,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexKernelFunction {
    name: String,
    param_name: String,
    param_type: LlvmType,
    blocks: Vec<VortexBlock>,
}

impl VortexKernelFunction {
    pub fn new(
        name: impl Into<String>,
        param_name: impl Into<String>,
        param_type: LlvmType,
    ) -> Self {
        Self {
            name: name.into(),
            param_name: param_name.into(),
            param_type,
            blocks: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn push_block(&mut self, block: VortexBlock) {
        self.blocks.push(block);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexBlock {
    label: String,
    ops: Vec<VortexOp>,
    terminator: VortexTerminator,
}

impl VortexBlock {
    pub fn new(label: impl Into<String>, terminator: VortexTerminator) -> Self {
        Self {
            label: label.into(),
            ops: Vec::new(),
            terminator,
        }
    }

    pub fn push_op(&mut self, op: VortexOp) {
        self.ops.push(op);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlvmType {
    I1,
    I8,
    I32,
    I64,
    Ptr,
    Struct(String),
}

impl LlvmType {
    pub fn struct_named(name: impl Into<String>) -> Self {
        Self::Struct(name.into())
    }

    fn render(&self) -> String {
        match self {
            Self::I1 => "i1".to_owned(),
            Self::I8 => "i8".to_owned(),
            Self::I32 => "i32".to_owned(),
            Self::I64 => "i64".to_owned(),
            Self::Ptr => "ptr".to_owned(),
            Self::Struct(name) => format!("%struct.{name}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VortexOp {
    StructFieldPtr {
        result: String,
        struct_type: String,
        base: String,
        field: u32,
    },
    Load {
        result: String,
        ty: LlvmType,
        ptr: String,
        align: u32,
    },
    Store {
        ty: LlvmType,
        value: String,
        ptr: String,
        align: u32,
    },
    IntToPtr {
        result: String,
        int_ty: LlvmType,
        value: String,
    },
    CtaCsrI32 {
        raw64: String,
        raw32: Option<String>,
        result: String,
        csr: VortexCtaCsr,
        uniform: bool,
    },
    LocalMemPtr {
        raw64: String,
        result: String,
    },
    Barrier {
        barrier_id: String,
        num_warps: String,
    },
    Binary {
        result: String,
        op: VortexBinaryOp,
        ty: LlvmType,
        lhs: String,
        rhs: String,
        overflow: IntegerOverflowSemantics,
    },
    Icmp {
        result: String,
        predicate: IcmpPredicate,
        ty: LlvmType,
        lhs: String,
        rhs: String,
    },
    Phi {
        result: String,
        ty: LlvmType,
        incoming: Vec<PhiIncoming>,
    },
    Gep {
        result: String,
        element_ty: LlvmType,
        base: String,
        index: String,
        inbounds: bool,
    },
    Cast {
        result: String,
        kind: CastKind,
        from_ty: LlvmType,
        value: String,
        to_ty: LlvmType,
    },
}

impl VortexOp {
    pub fn struct_field_ptr(
        result: impl Into<String>,
        struct_type: impl Into<String>,
        base: impl Into<String>,
        field: u32,
    ) -> Self {
        Self::StructFieldPtr {
            result: result.into(),
            struct_type: struct_type.into(),
            base: base.into(),
            field,
        }
    }

    pub fn load(
        result: impl Into<String>,
        ty: LlvmType,
        ptr: impl Into<String>,
        align: u32,
    ) -> Self {
        Self::Load {
            result: result.into(),
            ty,
            ptr: ptr.into(),
            align,
        }
    }

    pub fn store(
        ty: LlvmType,
        value: impl Into<String>,
        ptr: impl Into<String>,
        align: u32,
    ) -> Self {
        Self::Store {
            ty,
            value: value.into(),
            ptr: ptr.into(),
            align,
        }
    }

    pub fn inttoptr(result: impl Into<String>, int_ty: LlvmType, value: impl Into<String>) -> Self {
        Self::IntToPtr {
            result: result.into(),
            int_ty,
            value: value.into(),
        }
    }

    pub fn cta_csr_i32(
        result: impl Into<String>,
        raw64: impl Into<String>,
        csr: VortexCtaCsr,
    ) -> Self {
        Self::CtaCsrI32 {
            raw64: raw64.into(),
            raw32: None,
            result: result.into(),
            csr,
            uniform: false,
        }
    }

    pub fn uniform_cta_csr_i32(
        result: impl Into<String>,
        raw64: impl Into<String>,
        raw32: impl Into<String>,
        csr: VortexCtaCsr,
    ) -> Self {
        Self::CtaCsrI32 {
            raw64: raw64.into(),
            raw32: Some(raw32.into()),
            result: result.into(),
            csr,
            uniform: true,
        }
    }

    pub fn local_mem_ptr(result: impl Into<String>, raw64: impl Into<String>) -> Self {
        Self::LocalMemPtr {
            raw64: raw64.into(),
            result: result.into(),
        }
    }

    pub fn barrier(barrier_id: impl Into<String>, num_warps: impl Into<String>) -> Self {
        Self::Barrier {
            barrier_id: barrier_id.into(),
            num_warps: num_warps.into(),
        }
    }

    pub fn binary(
        result: impl Into<String>,
        op: VortexBinaryOp,
        ty: LlvmType,
        lhs: impl Into<String>,
        rhs: impl Into<String>,
    ) -> Self {
        Self::Binary {
            result: result.into(),
            op,
            ty,
            lhs: lhs.into(),
            rhs: rhs.into(),
            overflow: IntegerOverflowSemantics::None,
        }
    }

    pub fn binary_nsw(
        result: impl Into<String>,
        op: VortexBinaryOp,
        ty: LlvmType,
        lhs: impl Into<String>,
        rhs: impl Into<String>,
    ) -> Self {
        Self::Binary {
            result: result.into(),
            op,
            ty,
            lhs: lhs.into(),
            rhs: rhs.into(),
            overflow: IntegerOverflowSemantics::Nsw,
        }
    }

    pub fn icmp(
        result: impl Into<String>,
        predicate: IcmpPredicate,
        ty: LlvmType,
        lhs: impl Into<String>,
        rhs: impl Into<String>,
    ) -> Self {
        Self::Icmp {
            result: result.into(),
            predicate,
            ty,
            lhs: lhs.into(),
            rhs: rhs.into(),
        }
    }

    pub fn phi(result: impl Into<String>, ty: LlvmType, incoming: Vec<PhiIncoming>) -> Self {
        Self::Phi {
            result: result.into(),
            ty,
            incoming,
        }
    }

    pub fn gep(
        result: impl Into<String>,
        element_ty: LlvmType,
        base: impl Into<String>,
        index: impl Into<String>,
        inbounds: bool,
    ) -> Self {
        Self::Gep {
            result: result.into(),
            element_ty,
            base: base.into(),
            index: index.into(),
            inbounds,
        }
    }

    pub fn cast(
        result: impl Into<String>,
        kind: CastKind,
        from_ty: LlvmType,
        value: impl Into<String>,
        to_ty: LlvmType,
    ) -> Self {
        Self::Cast {
            result: result.into(),
            kind,
            from_ty,
            value: value.into(),
            to_ty,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexBinaryOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Udiv,
    Urem,
}

impl VortexBinaryOp {
    const fn render(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::And => "and",
            Self::Or => "or",
            Self::Udiv => "udiv",
            Self::Urem => "urem",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerOverflowSemantics {
    None,
    Nsw,
}

impl IntegerOverflowSemantics {
    const fn render(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Nsw => " nsw",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcmpPredicate {
    Ult,
    Uge,
}

impl IcmpPredicate {
    const fn render(self) -> &'static str {
        match self {
            Self::Ult => "ult",
            Self::Uge => "uge",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiIncoming {
    value: String,
    label: String,
}

impl PhiIncoming {
    pub fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    Sext,
    Zext,
}

impl CastKind {
    const fn render(self) -> &'static str {
        match self {
            Self::Sext => "sext",
            Self::Zext => "zext",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VortexTerminator {
    Br {
        label: String,
    },
    CondBr {
        cond: String,
        then_label: String,
        else_label: String,
    },
    RetVoid,
}

impl VortexTerminator {
    pub fn br(label: impl Into<String>) -> Self {
        Self::Br {
            label: label.into(),
        }
    }

    pub fn cond_br(
        cond: impl Into<String>,
        then_label: impl Into<String>,
        else_label: impl Into<String>,
    ) -> Self {
        Self::CondBr {
            cond: cond.into(),
            then_label: then_label.into(),
            else_label: else_label.into(),
        }
    }
}

pub(crate) fn render_vortex_device_module_llvm_ir(module: &VortexDeviceModule) -> String {
    let mut ir = LlvmIrModule::new();

    emit_header(&mut ir, module);
    emit_structs(&mut ir, module);
    emit_kernel_metadata(&mut ir, module);
    ir.blank_line();
    emit_kernels(&mut ir, module);
    ir.blank_line();
    emit_declarations_and_attributes(&mut ir);
    ir.blank_line();
    emit_module_flags(&mut ir);

    ir.finish()
}

fn emit_header(ir: &mut LlvmIrModule, module: &VortexDeviceModule) {
    ir.push_line(format!("; ModuleID = '{}'", module.module_id));
    ir.push_line(format!("source_filename = \"{}\"", module.source_filename));
    ir.push_line(format!("target datalayout = \"{TARGET_DATALAYOUT}\""));
    ir.push_line(format!("target triple = \"{TARGET_TRIPLE}\""));
    ir.blank_line();
}

fn emit_structs(ir: &mut LlvmIrModule, module: &VortexDeviceModule) {
    for struct_type in &module.structs {
        let fields = render_type_list(&struct_type.fields);
        ir.push_line(format!(
            "%struct.{} = type {{ {fields} }}",
            struct_type.name
        ));
    }
    if !module.structs.is_empty() {
        ir.blank_line();
    }
}

fn emit_kernel_metadata(ir: &mut LlvmIrModule, module: &VortexDeviceModule) {
    if module.kernels.is_empty() {
        return;
    }

    emit_private_metadata_string(ir, ".mandrel.vortex.kernel", "vortex.kernel");
    emit_private_metadata_string(ir, ".mandrel.vortex.source", &module.source_name);
    emit_global_annotations(ir, module);
    emit_used_kernel_list(ir, "@llvm.used", module);
    emit_used_kernel_list(ir, "@llvm.compiler.used", module);
}

fn emit_global_annotations(ir: &mut LlvmIrModule, module: &VortexDeviceModule) {
    let count = module.kernels.len();
    let mut line = format!(
        "@llvm.global.annotations = appending global [{count} x {{ ptr, ptr, ptr, i32, ptr }}] ["
    );
    for (index, kernel) in module.kernels.iter().enumerate() {
        if index != 0 {
            line.push_str(", ");
        }
        line.push_str(&format!(
            "{{ ptr, ptr, ptr, i32, ptr }} {{ ptr @{}, ptr @.mandrel.vortex.kernel, ptr @.mandrel.vortex.source, i32 1, ptr null }}",
            kernel.name
        ));
    }
    line.push_str("], section \"llvm.metadata\"");
    ir.push_line(line);
}

fn emit_used_kernel_list(ir: &mut LlvmIrModule, symbol: &str, module: &VortexDeviceModule) {
    let count = module.kernels.len();
    let mut line = format!("{symbol} = appending global [{count} x ptr] [");
    for (index, kernel) in module.kernels.iter().enumerate() {
        if index != 0 {
            line.push_str(", ");
        }
        line.push_str(&format!("ptr @{}", kernel.name));
    }
    line.push_str("], section \"llvm.metadata\"");
    ir.push_line(line);
}

fn emit_kernels(ir: &mut LlvmIrModule, module: &VortexDeviceModule) {
    for kernel in &module.kernels {
        emit_kernel(ir, kernel);
        ir.blank_line();
    }
}

fn emit_kernel(ir: &mut LlvmIrModule, kernel: &VortexKernelFunction) {
    ir.push_line(format!(
        "define dso_local void @{}({} %{}) #0 {{",
        kernel.name,
        kernel.param_type.render(),
        kernel.param_name
    ));
    for block in &kernel.blocks {
        emit_block(ir, block);
    }
    ir.push_line("}");
}

fn emit_block(ir: &mut LlvmIrModule, block: &VortexBlock) {
    ir.push_line(format!("{}:", block.label));
    for op in &block.ops {
        emit_op(ir, op);
    }
    emit_terminator(ir, &block.terminator);
    ir.blank_line();
}

fn emit_op(ir: &mut LlvmIrModule, op: &VortexOp) {
    match op {
        VortexOp::StructFieldPtr {
            result,
            struct_type,
            base,
            field,
        } => ir.push_line(format!(
            "  {result} = getelementptr inbounds %struct.{struct_type}, ptr {base}, i32 0, i32 {field}"
        )),
        VortexOp::Load {
            result,
            ty,
            ptr,
            align,
        } => ir.push_line(format!(
            "  {result} = load {}, ptr {ptr}, align {align}",
            ty.render()
        )),
        VortexOp::Store {
            ty,
            value,
            ptr,
            align,
        } => ir.push_line(format!(
            "  store {} {value}, ptr {ptr}, align {align}",
            ty.render()
        )),
        VortexOp::IntToPtr {
            result,
            int_ty,
            value,
        } => ir.push_line(format!(
            "  {result} = inttoptr {} {value} to ptr",
            int_ty.render()
        )),
        VortexOp::CtaCsrI32 {
            raw64,
            raw32,
            result,
            csr,
            uniform,
        } => emit_cta_csr_i32(ir, raw64, raw32.as_deref(), result, *csr, *uniform),
        VortexOp::LocalMemPtr { raw64, result } => {
            emit_csr_read_i64(ir, raw64, VortexCtaCsr::LocalMemAddr);
            ir.push_line(format!("  {result} = inttoptr i64 {raw64} to ptr"));
        }
        VortexOp::Barrier {
            barrier_id,
            num_warps,
        } => ir.push_line(format!(
            "  call void asm sideeffect \".insn r {RISCV_CUSTOM0}, 4, 0, x0, $0, $1\", \"r,r,~{{memory}}\"(i32 {barrier_id}, i32 {num_warps}) #2"
        )),
        VortexOp::Binary {
            result,
            op,
            ty,
            lhs,
            rhs,
            overflow,
        } => ir.push_line(format!(
            "  {result} = {}{} {} {lhs}, {rhs}",
            op.render(),
            overflow.render(),
            ty.render()
        )),
        VortexOp::Icmp {
            result,
            predicate,
            ty,
            lhs,
            rhs,
        } => ir.push_line(format!(
            "  {result} = icmp {} {} {lhs}, {rhs}",
            predicate.render(),
            ty.render()
        )),
        VortexOp::Phi {
            result,
            ty,
            incoming,
        } => ir.push_line(format!(
            "  {result} = phi {} {}",
            ty.render(),
            render_phi_incoming(incoming)
        )),
        VortexOp::Gep {
            result,
            element_ty,
            base,
            index,
            inbounds,
        } => {
            let inbounds_text = if *inbounds { " inbounds" } else { "" };
            ir.push_line(format!(
                "  {result} = getelementptr{inbounds_text} {}, ptr {base}, i64 {index}",
                element_ty.render()
            ));
        }
        VortexOp::Cast {
            result,
            kind,
            from_ty,
            value,
            to_ty,
        } => ir.push_line(format!(
            "  {result} = {} {} {value} to {}",
            kind.render(),
            from_ty.render(),
            to_ty.render()
        )),
    }
}

fn emit_cta_csr_i32(
    ir: &mut LlvmIrModule,
    raw64: &str,
    raw32: Option<&str>,
    result: &str,
    csr: VortexCtaCsr,
    uniform: bool,
) {
    emit_csr_read_i64(ir, raw64, csr);
    if uniform {
        let raw32 = raw32.unwrap_or(result);
        ir.push_line(format!("  {raw32} = trunc i64 {raw64} to i32"));
        ir.push_line(format!(
            "  {result} = call i32 {VORTEX_UNIFORM_INTRINSIC}(i32 {raw32}) #1"
        ));
    } else {
        ir.push_line(format!("  {result} = trunc i64 {raw64} to i32"));
    }
}

fn emit_csr_read_i64(ir: &mut LlvmIrModule, result: &str, csr: VortexCtaCsr) {
    ir.push_line(format!(
        "  {result} = call i64 asm sideeffect \"csrr $0, $1\", \"=r,i\"(i32 {}) #2",
        csr.encoded()
    ));
}

fn emit_terminator(ir: &mut LlvmIrModule, terminator: &VortexTerminator) {
    match terminator {
        VortexTerminator::Br { label } => ir.push_line(format!("  br label %{label}")),
        VortexTerminator::CondBr {
            cond,
            then_label,
            else_label,
        } => ir.push_line(format!(
            "  br i1 {cond}, label %{then_label}, label %{else_label}"
        )),
        VortexTerminator::RetVoid => ir.push_line("  ret void"),
    }
}

fn render_type_list(types: &[LlvmType]) -> String {
    let mut rendered = String::new();
    for (index, ty) in types.iter().enumerate() {
        if index != 0 {
            rendered.push_str(", ");
        }
        rendered.push_str(&ty.render());
    }
    rendered
}

fn render_phi_incoming(incoming: &[PhiIncoming]) -> String {
    let mut rendered = String::new();
    for (index, item) in incoming.iter().enumerate() {
        if index != 0 {
            rendered.push_str(", ");
        }
        rendered.push_str(&format!("[ {}, %{} ]", item.value, item.label));
    }
    rendered
}

fn emit_declarations_and_attributes(ir: &mut LlvmIrModule) {
    ir.push_line(format!("declare i32 {VORTEX_UNIFORM_INTRINSIC}(i32) #1"));
    ir.blank_line();
    ir.push_line("attributes #0 = { nounwind \"vortex-kernel\" \"frame-pointer\"=\"all\" \"no-trapping-math\"=\"true\" \"target-cpu\"=\"generic-rv64\" \"target-features\"=\"+64bit,+a,+d,+f,+m,+relax,+xvortex,+zaamo,+zalrsc,+zicond,+zicsr,+zmmul\" }");
    ir.push_line("attributes #1 = { nounwind memory(none) }");
    ir.push_line("attributes #2 = { nounwind }");
}

fn emit_module_flags(ir: &mut LlvmIrModule) {
    ir.push_line("!llvm.module.flags = !{!0, !1, !2, !4, !5, !6}");
    ir.push_line("!llvm.ident = !{!7}");
    ir.blank_line();
    ir.push_line("!0 = !{i32 1, !\"wchar_size\", i32 4}");
    ir.push_line("!1 = !{i32 1, !\"target-abi\", !\"lp64d\"}");
    ir.push_line("!2 = !{i32 6, !\"riscv-isa\", !3}");
    ir.push_line("!3 = !{!\"rv64i2p1_m2p0_a2p1_f2p2_d2p2_zicond1p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_xvortex1p0\"}");
    ir.push_line("!4 = !{i32 1, !\"Code Model\", i32 3}");
    ir.push_line("!5 = !{i32 7, !\"frame-pointer\", i32 2}");
    ir.push_line("!6 = !{i32 8, !\"SmallDataLimit\", i32 0}");
    ir.push_line("!7 = !{!\"mandrel generated Vortex LLVM IR\"}");
}

#[cfg(test)]
mod tests {
    use super::{
        LlvmType, VortexBlock, VortexDeviceModule, VortexKernelFunction, VortexOp,
        VortexStructType, VortexTerminator, render_vortex_device_module_llvm_ir,
    };
    use crate::codegen::vortex_ir::{VortexAxis, VortexCtaCsr};

    #[test]
    fn renders_kernel_metadata_and_cta_csr_ops() {
        let mut module =
            VortexDeviceModule::new("test module", "mandrel://test.ll", "mandrel device-ir test");
        module.add_struct(VortexStructType::new("args_t", vec![LlvmType::I64]));

        let mut entry = VortexBlock::new("entry", VortexTerminator::RetVoid);
        entry.push_op(VortexOp::uniform_cta_csr_i32(
            "%block_x",
            "%block_x64",
            "%block_x_raw",
            VortexCtaCsr::BlockId(VortexAxis::X),
        ));
        entry.push_op(VortexOp::cta_csr_i32(
            "%thread_x",
            "%thread_x64",
            VortexCtaCsr::ThreadId(VortexAxis::X),
        ));

        let mut kernel = VortexKernelFunction::new("demo_kernel", "arg", LlvmType::Ptr);
        kernel.push_block(entry);
        module.add_kernel(kernel);

        let ir = render_vortex_device_module_llvm_ir(&module);

        assert!(ir.contains("%struct.args_t = type { i64 }"));
        assert!(ir.contains("@llvm.global.annotations"));
        assert!(ir.contains("vortex.kernel"));
        assert!(ir.contains("define dso_local void @demo_kernel(ptr %arg) #0"));
        assert!(ir.contains("@llvm.riscv.vx.uniform.i32.i32"));
        assert!(ir.contains("csrr $0, $1"));
        assert!(ir.contains("i32 3286"));
        assert!(ir.contains("i32 3283"));
    }

    #[test]
    fn lowers_local_memory_and_barrier_semantics() {
        let mut module = VortexDeviceModule::new(
            "barrier module",
            "mandrel://barrier.ll",
            "mandrel barrier test",
        );
        let mut entry = VortexBlock::new("entry", VortexTerminator::RetVoid);
        entry.push_op(VortexOp::uniform_cta_csr_i32(
            "%cta_id",
            "%cta_id64",
            "%cta_id_raw",
            VortexCtaCsr::Id,
        ));
        entry.push_op(VortexOp::uniform_cta_csr_i32(
            "%cta_size",
            "%cta_size64",
            "%cta_size_raw",
            VortexCtaCsr::Size,
        ));
        entry.push_op(VortexOp::local_mem_ptr("%local", "%local64"));
        entry.push_op(VortexOp::barrier("%cta_id", "%cta_size"));

        let mut kernel = VortexKernelFunction::new("barrier_kernel", "arg", LlvmType::Ptr);
        kernel.push_block(entry);
        module.add_kernel(kernel);

        let ir = render_vortex_device_module_llvm_ir(&module);

        assert!(ir.contains("i32 3280"));
        assert!(ir.contains("i32 3282"));
        assert!(ir.contains("i32 3295"));
        assert!(ir.contains(".insn r 11, 4, 0, x0, $0, $1"));
        assert!(ir.contains("inttoptr i64 %local64 to ptr"));
    }
}
