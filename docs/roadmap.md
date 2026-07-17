# Roadmap

Mandrel's roadmap is organized around **evidence gates**, not crate count or kernel count. A phase is complete only when its design variables, artifacts, correctness, provenance, backend, and evidence class are captured by a reproducible experiment.

The current anchor is one exact-correct dense attention baseline on Vortex SimX. The next objective is to make software and hardware design points refer to the same tracked Vortex configuration, then progressively strengthen software semantics and hardware evidence.

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

## Phase 0 — Frozen reproducible baseline

**Status: implemented for Vortex SimX.**

Baseline:

- dense `attention_prefill_i8`;
- schedule `dense_scalar_two_pass_4x1x64`;
- `query_tile=4`, `key_tile=1`, `head_dim_tile=64`;
- two-pass stable softmax;
- direct global-memory Q/K/V/O accesses;
- `0 B` local memory per workgroup;
- textual LLVM-dialect MLIR;
- Vortex object, startup-aware ELF, and `.vxbin`;
- exact Rust reference comparison;
- SimX instructions/cycles/IPC and runtime transfer events.

Exit criteria:

- default and shape-override smokes pass exactly;
- logical work, lowered work, static traffic, and SimX counters remain distinct;
- generated reports state that SimX is not RTL/FPGA/silicon evidence.

## Phase 1 — Canonical target, artifact, and hardware schemas

**Status: initial implementation in this reorganization.**

Delivered boundaries:

- `mandrel-target-ir` owns backend/capability/target contracts, target constraints, operation capabilities, and kernel requirements;
- `mandrel-artifact` owns artifact kind, path, digest identity, sets, Vortex path bundles, and runtime artifact registry;
- `mandrel-hardware` owns Vortex hardware parameters, compiler target, realization kind, and realized hardware manifest;
- `mandrel-vortex-codegen` is separate from the Vortex runtime backend;
- `hardware/vortex/source.lock.toml` pins Vortex and LLVM-Vortex sources;
- `hardware/vortex/configs/current-default.toml` tracks the current hardware design point;
- experiment output is JSON + CSV, with no legacy history path.

Remaining exit criteria:

- compute content digests for emitted artifacts;
- bind resolved config and build identities into the live result instead of placeholder strings;
- derive all compiler-facing target facts from the materialized Vortex configuration;
- represent compile/runtime/correctness failures in the same result schema.

## Phase 2 — Hardware materialization and evidence ladder

Build the hardware branch without changing operator semantics first.

Work:

1. Parse tracked `HardwareDesignSpec`/Vortex config inputs.
2. Materialize upstream `VX_config.toml` overrides in an isolated build directory.
3. Capture generated `VX_config.vh` and `VX_config.h` as artifacts.
4. Record source SHA, config digest, build command/environment, and tool versions.
5. Derive `TargetSpec` from the resolved configuration.
6. Run the same binary/config pair through SimX and RTLSim.
7. Add Yosys synthesis artifact and report collection.
8. Add FPGA realization only after SimX/RTL correctness parity.

Exit criteria:

- requested, realized, and observed target facts are all recorded;
- target drift rejects an experiment before performance evidence is accepted;
- the dense baseline exact-passes on SimX and matching RTL;
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
- collect TCU counters in SimX and RTL;
- compare scalar and TCU-aware software on controlled hardware configurations.

### DXA track

- define target operations for GMEM→LMEM tiled async copy;
- represent transpose, multicast, barriers, and completion semantics;
- stage attention K/V tiles through DXA;
- collect copy/queue/stall counters;
- compare synchronous and asynchronous movement with matched schedules.

Exit criteria:

- SimX and RTL agree on correctness and instruction/event semantics;
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
- SimX/RTL correctness remains exact.

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
- SimX functional model;
- RTL implementation and counters;
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

Future hardware gates add configuration materialization, RTLSim parity, synthesis checks, and FPGA tests without weakening the current SimX exact-correct gate.
