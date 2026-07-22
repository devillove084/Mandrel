# Roadmap

Mandrel's roadmap is organized around **evidence gates**, not crate count or kernel count. A phase is complete only when its design variables, artifacts, correctness, provenance, backend, and evidence class are captured by a reproducible experiment.

The current anchor is one exact-correct dense attention baseline running Vortex SystemVerilog RTL through the pinned project-local Verilator RTLSim, with a canonical resolved-config identity observed through RTL before performance evidence is accepted. The next objective is to complete source/build provenance and derive all compiler-facing target facts from that materialization, then progressively strengthen software semantics and the FPGA/synthesis evidence ladder.

## Operating constraint

`cargo vortex-run-attention` remains the integration gate throughout the roadmap. New work must preserve:

- device artifact generation;
- attention plan and ABI validation;
- matching target requirements;
- Vortex execution;
- exact host-reference correctness;
- concise terminal counters;
- a versioned JSON result and CSV summary.

Mandrel does not maintain an automatic history/delta feedback loop. Researchers define comparisons and assemble study-level reports explicitly.

Bash under `scripts/env/` owns `uv`, checkouts, patching, builds, and environment export. Rust `xtask` owns operator artifact generation, launch planning, runtime execution, exact correctness, and profiling/reporting.

## Phase 0 — Frozen reproducible baseline

**Status: implemented for Vortex SystemVerilog through Verilator RTLSim.**

Baseline:

- dense `attention_prefill_i8`;
- schedule `dense_scalar_two_pass_4x1x64`;
- `query_tile=4`, `key_tile=1`, `head_dim_tile=64`;
- two-pass stable softmax;
- direct global-memory Q/K/V/O accesses;
- `0 B` local memory per workgroup;
- textual LLVM-dialect MLIR → LLVM IR → RV64 object → startup-aware ELF → `.vxbin`;
- Vortex SystemVerilog execution through the pinned project-local Verilator RTLSim;
- a pre-execution MLIR/LLVM probe of read-only config-identity CSR `0xFC5` with required realized/observed association-tag matching;
- exact Rust reference comparison;
- RTL `PERF` instructions/cycles/IPC and runtime transfer events, isolated from the probe's device instance;
- JSON/CSV output labeled with `rtl_simulation` evidence.

Exit criteria:

- default and shape-override smokes pass exactly;
- logical work, lowered work, static traffic, and RTL counters remain distinct;
- generated reports identify the baseline as `rtl_simulation`, not FPGA or silicon evidence.

## Phase 1 — Canonical target, artifact, and hardware schemas

**Status: initial schemas plus live resolved-config identity are implemented.**

Delivered boundaries:

- `mandrel-target-ir` owns backend/capability/target contracts, target constraints, operation capabilities, and kernel requirements;
- `mandrel-experiment` owns the current lightweight software-output references used by reports;
- `mandrel-vortex-backend` owns Vortex build-output paths and runtime kernel-image lookup;
- `mandrel-hardware` owns Vortex hardware parameters, compiler target, realization kind, and realized hardware manifest;
- `mandrel-vortex-codegen` is separate from the Vortex runtime backend;
- `hardware/vortex/source.lock.toml` pins Vortex and LLVM-Vortex sources;
- `hardware/vortex/configs/current-default.toml` tracks the current hardware design point;
- the resolved Vortex define set is canonicalized and bound to a full SHA-256 plus a 64-bit RTL association tag;
- experiment JSON/CSV records realized/observed hardware identity, with no legacy history path.

Remaining exit criteria:

- replace the current lightweight output list with typed software-build and complete hardware-realization manifests carrying content identities;
- bind source revision, patch series, build command/environment, generated-header identities, and tool versions into the live result;
- derive all compiler-facing target facts from the materialized Vortex configuration;
- represent compile/runtime/correctness failures in the same result schema.

## Phase 2 — Hardware materialization and evidence ladder

**Status: the first configuration-identity vertical slice is implemented through pinned Verilator RTLSim; complete build provenance, synthesis, and FPGA rungs remain.**

Delivered without changing attention semantics:

1. Parse `hardware/vortex/configs/current-default.toml` and map each tracked field to its upstream resolved define.
2. Invoke the pinned upstream config generator with the same fixed `-DSIMULATION -DSV_DPI` profile used by RTLSim, reject requested/resolved subset drift, and canonicalize the complete resolved define set plus XLEN.
3. Materialize a full SHA-256 manifest, 64-bit association tag, and generated `VX_mandrel.vh`/`VX_mandrel.h` headers in the isolated Vortex build tree.
4. Recompute and validate the manifest digest/profile/sidecars in Rust, compile the tag into read-only Vortex CSR `0xFC5`, read it with a generated MLIR/LLVM infrastructure probe, and reject a tag mismatch before attention performance is accepted.
5. Run the probe in a separate backend/device instance so attention `PERF` counters remain the control measurement.
6. Record full configuration SHA-256 plus realized/observed tags in JSON and CSV evidence.

Remaining work:

1. Turn tracked hardware inputs into deliberate upstream config overrides rather than only validating the current pinned upstream resolution.
2. Capture generated upstream `VX_config.vh` and `VX_config.h`, source revision, patch-series identity, build command/environment, and tool versions in a complete realization manifest.
3. Derive `TargetSpec` from the resolved configuration instead of duplicated target constants.
4. Keep the same binary/config pair exact-correct through pinned Verilator RTLSim as configuration materialization evolves.
5. Add Yosys synthesis artifact and report collection.
6. Add FPGA realization only after matching RTL correctness is preserved.

Exit criteria:

- requested, realized, and observed target facts are all recorded;
- target drift rejects an experiment before performance evidence is accepted;
- the dense baseline exact-passes on the matching SystemVerilog RTL through Verilator RTLSim;
- synthesis reports are labeled as estimates and tied to constraints/netlist identity.

## Phase 3 — Serving-faithful dense software baseline

The current symbol is serving-motivated but not serving-faithful. Before paged KV, make dense semantics credible.

Work:

- explicit scale and causal masking;
- batch, query heads, KV heads, GQA/MQA metadata;
- separate prefill and decode workload forms;
- ragged sequence and tail legality;
- quantization semantics rather than only i8 storage;
- key-tiled online `(max, sum, accumulator)` state;
- structural `key_tile > 1` loops;
- K/V local-memory staging and barriers;
- target-driven occupancy/local-memory legality;
- structured kernel compute IR and structured MLIR stages.

Exit criteria:

- dense prefill and decode match trusted references across edge cases;
- tile and memory choices alter actual generated control/data movement;
- the current scalar/two-pass kernel remains available as the control design;
- reports distinguish semantic, schedule, and lowering identities.

## Phase 4 — Existing Vortex TCU and DXA integration

Use existing upstream hardware before adding new RTL.

### TCU track

- define target MLIR operations for supported WMMA/WGMMA forms;
- begin with helper calls or inline assembly;
- validate i8 and selected low-precision types;
- expose capability/requirement gates;
- collect TCU counters from Verilator RTLSim;
- compare scalar and TCU-aware software on controlled hardware configurations.

### DXA track

- define target operations for GMEM→LMEM tiled async copy;
- represent transpose, multicast, barriers, and completion semantics;
- stage attention K/V tiles through DXA;
- collect copy/queue/stall counters;
- compare synchronous and asynchronous movement with matched schedules.

Exit criteria:

- Verilator RTLSim preserves exact host-reference correctness and exposes the expected instruction/event semantics;
- software requesting TCU/DXA cannot silently run on hardware without the feature;
- hardware-only, software-only, and matched designs are all explicitly represented;
- performance claims identify whether the gain came from compute, movement, overlap, or configuration.

## Phase 5 — Formal LLVM target support

Only after target operations validate the full loop, replace helper/inline-assembly exposure with maintainable compiler support.

Work in the LLVM-Vortex fork:

- RISC-V intrinsics for selected TCU/DXA operations;
- TableGen instruction definitions and feature predicates;
- operand/register constraints, including grouped registers where required;
- instruction selection and legalization;
- target scheduling/latency/resource models;
- MC assembler/disassembler support;
- feature strings and diagnostics;
- compiler tests tied to Mandrel-generated kernels.

Mandrel continues to own:

- serving/operator semantics;
- schedule and memory policy;
- target requirement selection;
- experiment design and evidence.

Exit criteria:

- generated kernels no longer require opaque inline `.insn` for the selected path;
- unsupported features fail with precise compiler diagnostics;
- LLVM artifacts are reproducible against the pinned fork revision;
- Verilator RTLSim correctness remains exact.

## Phase 6 — First Mandrel-specific RTL primitive

Do not begin with a standalone attention accelerator. Select one narrow primitive from measured bottlenecks.

Candidate classes:

- warp/subgroup max or sum reduction for online softmax;
- packed int8 dot/accumulate not efficiently covered by the chosen TCU path;
- a KV-oriented gather/layout primitive;
- a small synchronization or movement primitive exposed by attention stalls.

Required implementation surface:

- ISA/encoding or memory-mapped contract;
- Vortex decode/execute integration;
- SystemVerilog RTL implementation and counters exercised through Verilator RTLSim;
- compiler exposure;
- requirement/capability representation;
- correctness tests;
- synthesis timing/area impact;
- software fallback/control design.

Exit criteria:

- the primitive is justified by a baseline bottleneck;
- matched software+hardware gains survive correctness and controlled ablations;
- area/timing costs and evidence class are explicit;
- the result does not depend on comparing unlike workloads or backends.

## Phase 7 — Factorial studies and Pareto frontiers

A codesign conclusion requires more than “new hardware plus new software is faster.”

For each claimed interaction, define a software factor and a hardware factor. Record the complete 2×2 outcome matrix:

| | Hardware H0 | Hardware H1 |
|---|---|---|
| Software S0 | control | hardware-only effect |
| Software S1 | software-only effect or explicit unsupported outcome | matched codesign effect |

If S1/H0 is illegal, that unsupported outcome is still recorded, but it is not enough for a numeric interaction claim. A publishable interaction estimate needs four legal comparable points, a controlled emulation/fallback definition, or a clearly stated alternative causal argument.

Report dimensions:

- correctness and unsupported/failure outcomes;
- instructions, cycles, IPC, stalls, movement, occupancy;
- RTL cycles where available;
- FPGA latency/throughput where available;
- synthesis frequency, area, and power methodology;
- software artifact size/complexity;
- Pareto frontiers over performance, area, energy, and programmability.

Exit criteria:

- study manifests identify controlled and varying factors;
- JSON results and CSV summaries can be aggregated without hidden defaults;
- study-level plots/tables are generated from curated result sets, not implicit harness history;
- conclusions remain bounded to workload and evidence class.

## Phase 8 — Paged KV and serving replay

Paged KV follows a trustworthy dense baseline and hardware evidence path.

Work:

- page table and page-size semantics;
- page/stride/alignment legality before lowering;
- paged prefill and decode;
- GQA/MQA-aware KV sharing;
- ragged tails and variable-length batches;
- sequence splitting and work queues;
- KV compression/quantization policies;
- serving trace replay with explicit arrival and shape data;
- narrow llama.cpp/ggml or SGLang-class integration probes.

Exit criteria:

- dense and paged forms share semantic correctness tests;
- page/work-assignment policies are explicit software factors;
- hardware studies isolate gather, cache, local-memory, and movement effects;
- TTFT/TPOT/goodput are reported only when the runtime actually models the required serving behavior.

## Validation by phase

Core validation:

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo no-std-check
cargo deny check
cargo audit --deny warnings
```

Current integration validation:

```sh
cargo vortex-plan-attention
cargo vortex-generate-attention
cargo vortex-run-attention
```

Future hardware gates extend the current Verilator RTLSim exact-correct config-identity gate with complete build provenance, synthesis checks, and FPGA tests.
