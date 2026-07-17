# Mission

Mandrel is a serving-driven RISC-V inference hardware/software codesign lab.

Its mission is to make a narrow but important class of research experiments executable and reproducible:

> Start from attention and KV-cache behavior observed in LLM serving, vary software and realizable Vortex/RISC-V hardware together, and produce enough correctness, provenance, artifact, counter, RTL, and PPA evidence for a researcher to defend a codesign conclusion.

## Why Mandrel exists

Open RISC-V accelerator projects provide CPUs, GPUs, RTL, simulators, compilers, FPGA flows, and synthesis tools. LLM serving systems provide demanding workloads and sophisticated attention/KV strategies. What remains difficult is running a controlled experiment that binds all of these facts at once:

- exact workload semantics and shape;
- schedule and kernel lowering;
- compiler target and executable binary;
- hardware source, configuration, RTL/build identity, and backend;
- correctness and failure outcome;
- evidence class and metric provenance;
- artifacts that another researcher can rebuild and inspect.

Mandrel focuses on that binding.

## Scope

The research scope spans two branches:

```text
software:
  workload -> schedule -> kernel IR -> MLIR -> LLVM -> RISC-V binary

hardware:
  design spec -> Vortex configuration -> SimX / RTL / FPGA / netlist
```

The branches meet at a versioned experiment, not through an imagined LLVM-to-Verilog lowering.

The initial workload family is attention and KV cache because it exposes computation, reduction, data movement, irregular memory, serving phase, and hardware scheduling choices in one place. The project may use narrow MoE or framework probes, but they do not displace attention/KV as the primary codesign axis.

## Current foothold

Today Mandrel executes one dense `attention_prefill_i8` baseline on Vortex SimX. It generates LLVM-dialect MLIR and RISC-V artifacts, launches the `.vxbin`, exact-compares against a Rust reference, prints SimX counter statistics, and writes JSON/CSV results.

The current kernel is scalar, two-pass, direct-global, and uses no local-memory staging. It is infrastructure evidence, not a production-serving or hardware-performance result.

## Principles

1. **One trustworthy vertical slice before breadth.** Keep the attention path correct and executable while extending compiler and hardware depth.
2. **Serving semantics before benchmark theater.** Causal behavior, prefill/decode, head structure, GQA/MQA, paged KV, and replay shape must become explicit.
3. **Hardware facts have identities.** Source revision, resolved configuration, build, compiler target, and execution backend are part of the experiment.
4. **Legality is not identity.** A kernel may run on a non-identical target; exact target matching and requirement satisfaction are separate contracts.
5. **Evidence classes never collapse.** Static estimates, SimX, RTL, FPGA, synthesis, and silicon mean different things.
6. **LLVM owns target machinery, not model policy.** ISA, ABI, intrinsics, instruction selection, registers, MC support, and scheduling models belong in LLVM; attention/KV schedule policy remains in Mandrel.
7. **Use upstream hardware before inventing hardware.** Validate Vortex TCU and DXA, then add a narrow RTL primitive only when workload evidence justifies it.
8. **Experiments are human decisions.** Mandrel generates artifacts and reports; it does not automatically choose baselines, rank research ideas, or prescribe the next design.
9. **A result includes failures.** Unsupported target requirements, compilation errors, runtime errors, and correctness failures are first-class outcomes.
10. **No novelty by adjective.** Claims must be bounded by the measured workload, target, backend, evidence class, and related work.

## What success looks like

A successful Mandrel study can answer:

- Which software and hardware variables changed?
- Was the same semantic workload executed?
- Did every design point satisfy the same correctness policy?
- Which source/configuration/binary/netlist identities produced the result?
- Are metrics static, SimX, RTL, FPGA, synthesis, or silicon evidence?
- Does a matched software+hardware design beat controlled software-only and hardware-only alternatives?
- What bottleneck was removed, and what new bottleneck appeared?
- Can another researcher rebuild the design point and inspect its artifacts?

## Non-goals

Mandrel is not currently:

- a production vLLM/SGLang replacement;
- a full model compiler or serving scheduler;
- an automatic search/feedback system;
- a CUDA/Triton compatibility layer;
- an LLVM-to-Verilog compiler;
- a monolithic attention accelerator project;
- a source of FPGA, PPA, energy, or silicon claims before those evidence paths exist;
- a claim of being the first or uniquely complete system in the field.

The ambition is not to own every layer. It is to make the critical interfaces between workload, compiler, RISC-V GPU RTL, chip configuration, and evidence precise enough to support serious codesign.
