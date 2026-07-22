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
- code path: Rust plan → textual LLVM-dialect MLIR → LLVM IR → RV64 object → startup-aware ELF → `.vxbin`;
- compatibility gate: final ELF XLEN and ISA attributes must be a subset of the materialized RTL `VX_CFG_EXT_*` capabilities before `.vxbin` packaging;
- hardware identity: the tracked design subset is checked against every define resolved for the fixed Verilator RTLSim profile (`-DSIMULATION -DSV_DPI`), whose canonical SHA-256 is materialized into a manifest and a 64-bit RTL association tag;
- execution: an infrastructure MLIR probe reads that tag from read-only Vortex CSR `0xFC5` before attention runs through the pinned project-local Verilator RTLSim and Vortex runtime;
- validation: Rust recomputes the manifest digest, verifies both sidecars and the RTLSim profile, rejects a realized/observed association-tag mismatch before accepting performance evidence, then exact-compares attention against a Rust host reference;
- evidence: hardware identity, launch/transfer events, and RTL `PERF` instructions, cycles, and IPC;
- outputs: a versioned JSON result and a one-row CSV summary, both labeled with `rtl_simulation` evidence.

This baseline proves that the executable spine, runtime hardware-identity gate, and correctness contract pass through Vortex RTL end to end. The full SHA-256 remains the host-side resolved-configuration identity; the CSR exposes only its first 64 bits as a runtime association tag, not as a complete cryptographic proof. The baseline is not yet serving-faithful prefill/decode, paged attention, a structural key-tiled online-softmax kernel, an FPGA measurement, or a PPA result. Verilator RTLSim cycles are RTL-simulation observations, not FPGA or chip performance.

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
    -> Verilator RTLSim / FPGA / synthesized netlist

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

1. preserve the exact-correct Verilator RTLSim attention path as the control;
2. materialize additional tracked Vortex hardware configurations with matching compiler and RTL identities;
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

The JSON result is the complete machine-readable record. Its `hardware_identity` object records the full resolved-configuration SHA-256, realized 64-bit tag, RTL-observed tag, and `association_tag_match` decision. The CSV exposes the same core identity fields alongside correctness and profiling values for scripts, notebooks, and manually curated comparisons. Mandrel does not automatically choose a historical baseline or infer the next optimization.

Evidence sources remain distinct:

| Evidence class | Meaning |
|---|---|
| Static model | Logical/lowered work and traffic estimates. |
| RTL simulation | Instructions, cycles, IPC, and runtime events from the matching SystemVerilog RTL through pinned Verilator RTLSim. Current baseline. |
| FPGA measurement | Measurements from a versioned bitstream and board setup. Planned. |
| Synthesis estimate | Timing, area, and power methodology tied to a netlist. Planned. |
| Silicon measurement | Not currently available. |

## Quick start

The root `Makefile` is the recommended orchestration entry point:

```sh
make help
make install
make run
```

`make install` materializes the complete pinned project-local environment, validates the tracked Vortex design subset, generates the canonical RTLSim-profile manifest/tag and RTL header, and builds RTLSim with the same fixed profile. `make run` recomputes and validates that manifest, regenerates the identity-probe and operator artifacts, checks plan/ABI and ELF/RTL ISA compatibility, requires the RTL-observed config tag to match, launches attention through Vortex SystemVerilog, exact-compares the output, prints RTL `PERF` counters, and writes JSON/CSV with `rtl_simulation` evidence. The probe uses a separate backend/device instance, so its instructions and cycles do not contaminate the attention counters.

Run setup and the RTL integration gate in one command:

```sh
make e2e
```

Common targets:

| Target | Purpose |
|---|---|
| `make install` / `make setup` | Run the complete uv, Verilator, LLVM-Vortex, Vortex, and environment-check flow. |
| `make setup-python` | Create the frozen uv-managed Python environment. |
| `make setup-verilator` | Build the pinned project-local Verilator. |
| `make setup-llvm` | Build or verify LLVM-Vortex and compiler-rt. |
| `make setup-vortex` | Fetch and patch Vortex, materialize its resolved config identity, and build the matching RTLSim runtime. |
| `make env-check` | Verify revisions, tools, libraries, non-RVC libc, and generated Vortex config identity integrity. |
| `make plan` | Print the typed attention launch plan. |
| `make generate` | Generate and validate MLIR, LLVM IR, object, ELF, and `.vxbin` artifacts. |
| `make run` / `make profile` | Gate RTL config identity, run exact RTLSim correctness, and emit terminal, JSON, and CSV profiles. |
| `make validate` | Run formatting, workspace check, Clippy, tests, and the RISC-V `no_std` check. |
| `make verify` | Run the environment check, full Rust validation, and the RTL integration gate. |

The Makefile is intentionally only an orchestration layer. Bash under `scripts/env/` owns the frozen `uv` environment, pinned checkouts, reviewed patches, toolchain and RTLSim builds, and exported environment. Rust `xtask` owns operator artifact generation, typed launch planning, runtime execution, exact correctness, and profile/report emission.

For manual debugging, the underlying commands remain available:

```sh
./scripts/env/setup.sh all
source scripts/env/vortex-env.sh
cargo vortex-plan-attention
cargo vortex-generate-attention
cargo vortex-run-attention
```

`make env` prints the activation command because a child `make` process cannot modify its parent shell. `make shell` instead opens an interactive Bash with the project environment already active.

## Repository map

```text
crates/
  model-ir/           workload and operator semantics
  schedule/           target-aware schedule selection
  kernel-ir/          kernel catalog, ABI, and launch descriptors
  target-ir/          canonical target facts and kernel requirements
  compiler/           workload/schedule to executable Vortex plans
  vortex-codegen/     plan validation, device IR, and MLIR generation
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
Makefile                environment, validation, and RTLSim workflow entry point
```

The current `kernel-ir` is still mostly interface/ABI/launch IR; computation is generated directly from compiler plans into an internal Vortex device IR and textual LLVM-dialect MLIR. A structured compute IR and structured MLIR pipeline are roadmap work, not current claims.

## Design rules

- Preserve one exact-correct executable attention spine while changing the stack around it.
- Derive compiler-facing target facts from a tracked hardware design, not duplicated constants.
- Record requested, realized, and observed target identities separately.
- Separate exact hardware identity from the weaker question “can this kernel legally run?”
- Keep attention/KV policy in Mandrel; keep ISA/ABI/instruction machinery in LLVM.
- Validate upstream TCU/DXA before adding new RTL.
- Do not compare static, RTL-simulation, FPGA, synthesis, and silicon evidence as if they were interchangeable.
- Do not automate research judgment. Generate evidence that a researcher can inspect and defend.

## Documentation

- [`docs/mission.md`](docs/mission.md) — stable mission, principles, and non-goals.
- [`docs/codesign-architecture.md`](docs/codesign-architecture.md) — software/hardware architecture and experiment contract.
- [`docs/roadmap.md`](docs/roadmap.md) — phased implementation and evidence gates.
- [`docs/mlir.md`](docs/mlir.md) — current MLIR/LLVM/Vortex artifact path.
- [`docs/llm-serving-kv-attention-survey.md`](docs/llm-serving-kv-attention-survey.md) — literature survey and design hypotheses.
- [`hardware/vortex/README.md`](hardware/vortex/README.md) — tracked Vortex hardware inputs.
- [`experiments/README.md`](experiments/README.md) — experiment and report policy.
