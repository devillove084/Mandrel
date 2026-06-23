#![allow(dead_code, unused_imports)]

use crate::codegen::vortex_ir::VortexCtaCsr;

const TARGET_DATALAYOUT: &str = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
const TARGET_TRIPLE: &str = "riscv64-unknown-unknown-elf";
const RISCV_CUSTOM0: u32 = 0x0B;

#[path = "device_ir_mlir.rs"]
mod device_ir_mlir;
pub(crate) use device_ir_mlir::render_vortex_device_module_mlir;

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
    F32,
    Ptr,
    Struct(String),
}

impl LlvmType {
    pub fn struct_named(name: impl Into<String>) -> Self {
        Self::Struct(name.into())
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
    FloatBinary {
        result: String,
        op: VortexFloatBinaryOp,
        ty: LlvmType,
        lhs: String,
        rhs: String,
    },
    Icmp {
        result: String,
        predicate: IcmpPredicate,
        ty: LlvmType,
        lhs: String,
        rhs: String,
    },
    Fcmp {
        result: String,
        predicate: FcmpPredicate,
        ty: LlvmType,
        lhs: String,
        rhs: String,
    },
    Select {
        result: String,
        cond: String,
        ty: LlvmType,
        true_value: String,
        false_value: String,
    },
    FloatUnaryIntrinsic {
        result: String,
        intrinsic: VortexFloatUnaryIntrinsic,
        ty: LlvmType,
        value: String,
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

    pub fn float_binary(
        result: impl Into<String>,
        op: VortexFloatBinaryOp,
        ty: LlvmType,
        lhs: impl Into<String>,
        rhs: impl Into<String>,
    ) -> Self {
        Self::FloatBinary {
            result: result.into(),
            op,
            ty,
            lhs: lhs.into(),
            rhs: rhs.into(),
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

    pub fn fcmp(
        result: impl Into<String>,
        predicate: FcmpPredicate,
        ty: LlvmType,
        lhs: impl Into<String>,
        rhs: impl Into<String>,
    ) -> Self {
        Self::Fcmp {
            result: result.into(),
            predicate,
            ty,
            lhs: lhs.into(),
            rhs: rhs.into(),
        }
    }

    pub fn select(
        result: impl Into<String>,
        cond: impl Into<String>,
        ty: LlvmType,
        true_value: impl Into<String>,
        false_value: impl Into<String>,
    ) -> Self {
        Self::Select {
            result: result.into(),
            cond: cond.into(),
            ty,
            true_value: true_value.into(),
            false_value: false_value.into(),
        }
    }

    pub fn float_unary_intrinsic(
        result: impl Into<String>,
        intrinsic: VortexFloatUnaryIntrinsic,
        ty: LlvmType,
        value: impl Into<String>,
    ) -> Self {
        Self::FloatUnaryIntrinsic {
            result: result.into(),
            intrinsic,
            ty,
            value: value.into(),
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
pub enum VortexFloatBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl VortexFloatBinaryOp {
    const fn render(self) -> &'static str {
        match self {
            Self::Add => "fadd",
            Self::Sub => "fsub",
            Self::Mul => "fmul",
            Self::Div => "fdiv",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexFloatUnaryIntrinsic {
    Exp,
}

impl VortexFloatUnaryIntrinsic {
    const fn render(self) -> &'static str {
        match self {
            Self::Exp => "llvm.intr.exp",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerOverflowSemantics {
    None,
    Nsw,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FcmpPredicate {
    Ogt,
    Olt,
}

impl FcmpPredicate {
    const fn render(self) -> &'static str {
        match self {
            Self::Ogt => "ogt",
            Self::Olt => "olt",
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
    Trunc,
    Sitofp,
    Fptosi,
}

impl CastKind {
    const fn render(self) -> &'static str {
        match self {
            Self::Sext => "sext",
            Self::Zext => "zext",
            Self::Trunc => "trunc",
            Self::Sitofp => "sitofp",
            Self::Fptosi => "fptosi",
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

#[cfg(test)]
mod tests {
    use super::{
        LlvmType, PhiIncoming, VortexBlock, VortexDeviceModule, VortexKernelFunction, VortexOp,
        VortexStructType, VortexTerminator, render_vortex_device_module_mlir,
    };
    use crate::codegen::vortex_ir::{VortexAxis, VortexCtaCsr};

    #[test]
    fn renders_llvm_dialect_mlir_with_block_arguments_and_vortex_contracts() {
        let mut module = VortexDeviceModule::new(
            "mlir module",
            "mandrel://test.mlir",
            "mandrel device-ir mlir test",
        );
        module.add_struct(VortexStructType::new("args_t", vec![LlvmType::I64]));

        let mut entry = VortexBlock::new("entry", VortexTerminator::br("loop"));
        entry.push_op(VortexOp::uniform_cta_csr_i32(
            "%block_x",
            "%block_x64",
            "%block_x_raw",
            VortexCtaCsr::BlockId(VortexAxis::X),
        ));

        let mut loop_block = VortexBlock::new("loop", VortexTerminator::RetVoid);
        loop_block.push_op(VortexOp::phi(
            "%depth",
            LlvmType::I32,
            vec![PhiIncoming::new("0", "entry")],
        ));

        let mut kernel = VortexKernelFunction::new("demo_kernel", "arg", LlvmType::Ptr);
        kernel.push_block(entry);
        kernel.push_block(loop_block);
        module.add_kernel(kernel);

        let mlir = render_vortex_device_module_mlir(&module);

        assert!(mlir.contains("llvm.target_triple = \"riscv64-unknown-unknown-elf\""));
        assert!(mlir.contains("llvm.mlir.global appending @\"llvm.global.annotations\""));
        assert!(mlir.contains("vortex-kernel"));
        assert!(mlir.contains("llvm.func @demo_kernel(%arg: !llvm.ptr)"));
        assert!(mlir.contains("llvm.inline_asm has_side_effects \"csrr"));
        assert!(mlir.contains("llvm.call @\"llvm.riscv.vx.uniform.i32.i32\""));
        assert!(mlir.contains("llvm.br ^loop(%ventry_c1_i32 : i32)"));
        assert!(mlir.contains("^loop(%depth: i32):"));
    }
}
