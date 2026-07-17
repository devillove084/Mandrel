# Mandrel

> A serving-driven RISC-V inference hardware/software codesign lab.

Mandrel is building an open laboratory for studying how LLM-serving workloads should shape RISC-V software, compiler targets, Verilog/RTL features, and chip configurations. Its job is to turn a workload, a software design, and a hardware design into one reproducible experiment with explicit binaries, hardware identity, correctness, counters, artifacts, and evidence class.

The long-term goal is ambitious but narrow: make attention and KV-cache workloads executable across an open Vortex-based stack, then use controlled experiments to design better schedules, LLVM support, RTL primitives, memory systems, and chips.

Mandrel is not a production serving framework, an automatic optimizer, or a claim of leading performance. Researchers choose hypotheses and design points; Mandrel makes those choices executable and auditable.

![Mandrel serving-driven RISC-V inference codesign architecture](docs/assets/mandrel-codesign-architecture.svg)

## Current reality

Mandrel currently has one end-to-end executable baseline:

- workload: dense, serving-motivated `attention_prefill_i8`;
- schedule: `dense_scalar_two_pass_4x1x64`;
- structure: `query_tile=4`, scalar `key_tile=1`, `head_dim_tile=64`;
- memory: direct global-memory access and `0 B` local memory per workgroup;
- code path: Rust plan → textual LLVM-dialect MLIR → LLVM IR → RISC-V object → startup-aware ELF → `.vxbin`;
- execution: Vortex SimX through the Vortex runtime;
- validation: exact comparison against a Rust host reference;
- evidence: launch/transfer events and SimX `PERF` instructions, cycles, and IPC;
- outputs: a versioned JSON result and a one-row CSV summary.

This baseline proves the executable spine and correctness contract. It is not yet serving-faithful prefill/decode, paged attention, a structural key-tiled online-softmax kernel, an RTL result, an FPGA measurement, or a PPA result. SimX cycles are simulator observations, not chip performance.

## The focused research question

Mandrel asks one recurring question:

> For a serving workload and a fixed correctness contract, which combination of software lowering and realizable RISC-V/Vortex hardware produces the best measured tradeoff, and why?

That question is studied across four coupled surfaces:

1. **Workload and operator semantics** — prefill, decode, causal masking, GQA/MQA, paged KV, quantization, and serving replay.
2. **Software design** — schedule, tiling, work assignment, memory movement, kernel IR, MLIR lowering, LLVM target support, and runtime behavior.
3. **Hardware design** — Vortex parameters, TCU, DXA, local memory, caches, new RTL primitives, RTLSim, FPGA, and synthesis/PPA.
4. **Evidence** — exact correctness, requested/realized/observed target facts, source/build identities, artifacts, counters, events, and controlled ablations.

## Codesign architecture

Mandrel has one experiment plane and two artifact-producing branches:

```text
software branch:
  serving workload
    -> target-aware schedule
    -> executable kernel IR
    -> structured MLIR
    -> LLVM / Vortex target support
    -> RISC-V object / ELF / vxbin

hardware branch:
  hardware design spec
    -> resolved Vortex configuration
    -> generated Verilog configuration
    -> SimX / RTL simulation / FPGA / synthesized netlist

experiment plane:
  workload + software design + hardware design + toolchain identity
    -> compatibility and correctness gates
    -> counters, events, artifacts, provenance, CSV/JSON report
    -> human analysis and the next explicit experiment
```

LLVM does not lower directly into Verilog. LLVM owns target-facing ISA, ABI, intrinsics, instruction selection, register constraints, and scheduling models. RTL and chip design are a separate branch resolved from the same experiment manifest. The executable binary and matching realized hardware meet at execution and evidence collection.

See [`docs/codesign-architecture.md`](docs/codesign-architecture.md) for the full design.

## Near-term proof point

The first codesign study is deliberately conservative:

1. materialize tracked Vortex hardware configurations;
2. run the same attention baseline through matching SimX and RTL configurations;
3. enable and expose existing Vortex TCU and DXA mechanisms;
4. compare software-only, hardware-only, and matched software+hardware design points;
5. add formal LLVM support only after helper-call/inline-assembly paths validate semantics;
6. use the evidence to choose one narrow Mandrel-specific RTL primitive, such as a reduction or packed-dot operation.

This sequence avoids building a monolithic attention accelerator before the experiment and compiler contracts are credible.

## Experiment output

`cargo vortex-run-attention` prints a `perf stat`-style terminal summary and writes:

```text
target/mandrel/vortex/attention_prefill_i8.experiment.json
target/mandrel/vortex/attention_prefill_i8.experiment.csv
```

The JSON result is the complete machine-readable record. The CSV contains one row of core fields for scripts, notebooks, and manually curated comparisons. Mandrel does not automatically choose a historical baseline or infer the next optimization.

Evidence sources remain distinct:

| Evidence class | Meaning |
|---|---|
| Static model | Logical/lowered work and traffic estimates. |
| SimX observation | Functional simulator counters and runtime events. |
| RTL simulation | Cycle/event evidence from matching RTL. Planned. |
| FPGA measurement | Measurements from a versioned bitstream and board setup. Planned. |
| Synthesis estimate | Timing, area, and power methodology tied to a netlist. Planned. |
| Silicon measurement | Not currently available. |

## Quick start

Basic workspace validation:

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo no-std-check
```

Inspect and run the current attention path:

```sh
cargo vortex-plan-attention
cargo vortex-generate-attention
cargo vortex-run-attention
```

The Vortex commands require the configured Vortex checkout, LLVM toolchain, runtime, and SimX build. Use the setup/status commands when preparing a new environment:

```sh
cargo vortex-fetch
cargo vortex-system-tools
cargo vortex-install
cargo vortex-env
cargo vortex-status
```

The primary integration gate is `cargo vortex-run-attention`: it regenerates the device artifact, validates plan/ABI constraints, launches SimX, exact-compares the output, prints counter statistics, and writes JSON/CSV evidence.

## Repository map

```text
crates/
  model-ir/           workload and operator semantics
  schedule/           target-aware schedule selection
  kernel-ir/          kernel catalog, ABI, and launch descriptors
  target-ir/          canonical target facts and kernel requirements
  compiler/           workload/schedule to executable Vortex plans
  vortex-codegen/     plan validation, device IR, and MLIR generation
  artifact/           artifact identities, sets, paths, and registries
  hardware/           hardware design and realization schemas
  vortex-backend/     Vortex runtime, execution, FFI, and host toolchain driver
  experiment/         experiment specs, events, correctness, and results
  profiler/           static estimates and runtime counter parsing
  runtime/            host/runtime scaffolding
  kernels/            host reference kernels
  ggml-adapter/       narrow integration probe
  xtask/              reproducible project and Vortex commands

hardware/vortex/
  source.lock.toml     pinned Vortex and LLVM-Vortex source identities
  configs/             tracked hardware design points
  patches/             reviewed Mandrel-specific RTL/simulator patches

experiments/           human-authored study manifests and publication bundles
docs/                  mission, architecture, roadmap, toolchain notes, survey
external/              materialized upstream checkouts and builds
target/mandrel/         generated artifacts and experiment outputs
```

The current `kernel-ir` is still mostly interface/ABI/launch IR; computation is generated directly from compiler plans into an internal Vortex device IR and textual LLVM-dialect MLIR. A structured compute IR and structured MLIR pipeline are roadmap work, not current claims.

## Design rules

- Preserve one exact-correct executable attention spine while changing the stack around it.
- Derive compiler-facing target facts from a tracked hardware design, not duplicated constants.
- Record requested, realized, and observed target identities separately.
- Separate exact hardware identity from the weaker question “can this kernel legally run?”
- Keep attention/KV policy in Mandrel; keep ISA/ABI/instruction machinery in LLVM.
- Validate upstream TCU/DXA before adding new RTL.
- Do not compare static, SimX, RTL, FPGA, synthesis, and silicon evidence as if they were interchangeable.
- Do not automate research judgment. Generate evidence that a researcher can inspect and defend.

## Documentation

- [`docs/mission.md`](docs/mission.md) — stable mission, principles, and non-goals.
- [`docs/codesign-architecture.md`](docs/codesign-architecture.md) — software/hardware architecture and experiment contract.
- [`docs/roadmap.md`](docs/roadmap.md) — phased implementation and evidence gates.
- [`docs/mlir.md`](docs/mlir.md) — current MLIR/LLVM/Vortex artifact path.
- [`docs/llm-serving-kv-attention-survey.md`](docs/llm-serving-kv-attention-survey.md) — literature survey and design hypotheses.
- [`hardware/vortex/README.md`](hardware/vortex/README.md) — tracked Vortex hardware inputs.
- [`experiments/README.md`](experiments/README.md) — experiment and report policy.
