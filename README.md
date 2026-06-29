# Mandrel

> **Quantify, analyze, and optimize the full hardware/software stack for efficient LLM serving.**
>
> Mandrel uses LLM serving workloads, starting from attention and KV cache, to turn hardware, memory, runtime, compiler, layout, scheduling, and kernel design choices into measurable experiments on RISC-V/Vortex-like hardware.

Mandrel is not just an attention-kernel demo and not merely a benchmark harness. Its goal is to make the full LLM-serving stack optimizable: define design variables, run executable experiments, collect correctness and trace evidence, analyze bottlenecks, and use the results to drive more efficient hardware/software designs. The current Vortex attention path is the first narrow but real measurement spine.

```text
Measure the stack. Explain the bottleneck. Optimize the design.
```

## Mission

Mandrel exists to quantify, analyze, and maximize LLM-serving efficiency across the hardware/software stack — chip architecture, memory hierarchy, KV layout, data movement, runtime/driver behavior, compiler lowering, scheduling policy, and kernels.

The project currently focuses on Vortex/RISC-V GPGPU because it provides an open, inspectable target where design ideas can be made executable. The long-term goal is a reproducible optimization loop where LLM-serving workloads drive design-space exploration and full-stack efficiency decisions with correctness checks, measurable traces, and experiment reports.

Read the full mission: [`docs/mission.md`](docs/mission.md).

## Quantitative architecture method for AI workloads

Mandrel follows the spirit of the quantitative approach to computer architecture: optimize systems by measuring real workloads, making design alternatives explicit, comparing them with objective metrics, and using the results to guide the next design.

For AI serving, that loop becomes:

```text
AI workload and algorithm shape
  -> measurable design variables
  -> hardware, memory, runtime, compiler, schedule, and kernel choices
  -> executable artifact or calibrated model
  -> correctness, counters, traces, and derived metrics
  -> bottleneck and sensitivity analysis
  -> hardware guidance and algorithm feedback
  -> next co-designed workload/hardware iteration
```

This means Mandrel should eventually answer questions in both directions:

- **Hardware direction:** which ISA, memory hierarchy, local-memory, cache, page-table, DMA/copy, barrier, or runtime features improve serving efficiency?
- **Software/runtime direction:** which layouts, schedules, batching policies, compression policies, and lowering strategies best fit a target?
- **Algorithm direction:** which model/attention/KV choices are hardware-friendly enough to justify algorithmic adoption or redesign?

## Why Mandrel

Modern LLM serving is dominated by CUDA-centric kernel infrastructure and framework-specific heuristics. To maximize efficiency on open hardware, we need more than isolated microbenchmarks: we need realistic workload paths that expose attention, KV cache, copies, communication, runtime overhead, memory hierarchy, scheduling policy, compiler lowering, and kernel implementation decisions as comparable design variables.

Mandrel starts from the hardest useful slice, then grows it into a full-stack design-space optimizer:

- **Attention/KV first**: dense `attention_prefill_i8` is the active executable baseline; paged KV legality and serving-shaped memory are next.
- **Design variables first**: layouts, tiles, runtime shapes, target assumptions, memory movement, and lowering policies should be explicit experiment knobs.
- **Compiler/runtime together**: model IR, schedule metadata, ABI/layout validation, MLIR generation, artifacts, runtime launch, and traces live in one Rust workspace.
- **Vortex/RISC-V first**: generated device code targets the Vortex toolchain and runs through `simx` today.
- **Correctness first**: generated kernels are compared against a Rust host reference.
- **Quantification first**: runtime shape, launch, transfer, cache, counters, workload, wall-time, correctness, derived metrics, and experiment summaries are persisted as JSONL evidence.

## Current executable spine

![Mandrel workload-driven codesign spine](docs/assets/mandrel-codesign-spine.svg)

Today this spine is implemented for dense attention prefill:

```text
AttentionOp::prefill_i8_demo
  -> dense online-softmax schedule
  -> VortexAttentionPrefillPlan
  -> ABI/layout metadata validation
  -> LLVM dialect MLIR
  -> Vortex LLVM object, ELF, and vxbin
  -> Vortex simx runtime launch
  -> host reference compare
  -> JSONL trace history, experiment result, and deltas
```

## What works today

| Area | Current state |
| --- | --- |
| Workload | Dense `attention_prefill_i8` baseline with runtime shape overrides. |
| Schedule | Dense KV layout, online max/sum softmax strategy, `4x16x64` tile metadata. |
| ABI/layout metadata | Buffer slots, scalar arg indices, dense row-major strides, quantization, runtime shape policy, and KV policy are validated at codegen/runtime gates. |
| Codegen | Rust plan emits LLVM dialect MLIR for Vortex. Current generated attention lowering is a dense scalar two-pass stable-softmax baseline; `key_tile` is metadata/ABI today, not yet a structural key-block loop. |
| Artifact pipeline | MLIR validates through `mlir-translate`, Vortex `clang`, `.o`, startup-aware `.elf`, and `.vxbin` packaging. |
| Runtime | `VortexBackend` wraps runtime/device/queue, artifact lookup, kernel cache, launch, transfers, and readback. |
| Correctness | Device output is compared against a Rust host reference. |
| Observability | `PERF`, launch dimensions, transfer bytes, cache hits, workload bytes/elements, logical MACs, wall time, correctness, derived ratios, trace JSONL, and companion experiment-result JSONL. |
| Experiment model | `mandrel-experiment` provides first-pass `ExperimentSpec`, `ExperimentResult`, target/memory specs, correctness records, and runtime-event records. |
| CLI | `xtask` is modularized and exposed through `clap` commands such as `cargo vortex-run-attention`. |
| Next | True runtime event emission, target/memory spec unification, Paged KV legality planning, and dense key-tiled online lowering. |

## How to read the current output

Mandrel's output is meant to explain the whole spine, not only say that a kernel ran. A recent local `cargo vortex-run-attention` run shows four kinds of evidence.

### 1. Flow gates

```text
attention.runtime: compiling attention launch plan
attention.runtime: validating attention ABI/layout metadata
attention.runtime: building deterministic attention input
attention.runtime: runtime shape compiled=64x64 default=8x16 actual=8x16 tile(query=4, key=16, head_dim=64)
attention.runtime: backend transfers host_to_device=384B device_to_host=128B total=512B
attention.runtime: execution shape workgroups=16 threads_per_workgroup=16 total_threads=256
attention.runtime: compare summary elements=128 mismatches=0 status=exact
attention runtime correctness PASSED
```

What this means:

| Line | Why it matters |
| --- | --- |
| `compiling attention launch plan` | The workload is represented as a Rust plan, not manually launched as an opaque binary. |
| `validating attention ABI/layout metadata` | Buffer slots, scalar arg indices, dense row-major layout, quantization, KV policy, and runtime shape are checked before codegen/runtime use them. |
| `compiled=64x64 ... actual=8x16` | The generated kernel has a compiled maximum shape while the smoke uses a smaller runtime prefix. This is the first step toward serving-shaped runtime variation. |
| `host_to_device=384B device_to_host=128B` | Data movement is visible. For a codesign lab, copy and memory traffic are first-class signals, not hidden overhead. |
| `workgroups=16 threads_per_workgroup=16` | The launch maps schedule decisions to hardware execution shape. |
| `mismatches=0 status=exact` | Correctness is a gate: metric changes are not trusted unless the host reference still matches. |

### 2. Runtime trace summary

```text
attention.runtime trace summary:
  kernel: attention_prefill_i8
  runtime: sequence=8 head_dim=16 query_tile=4 key_tile=16
  compiled: sequence=64 head_dim=64 head_dim_tile=64
  execution: workgroups=16 threads_per_workgroup=16 total_threads=256 module_cache_hit=false kernel_cache_hit=false
  workload: logical_macs=2048 q_elements=128 kv_elements=256 output_elements=128
  memory: q_bytes=128 kv_bytes=256 output_bytes=128 host_to_device_bytes=384 device_to_host_bytes=128 total_transfer_bytes=512
  counters: instructions=165144 cycles=414598 ipc=0.398
  derived: instrs/output=1290.188 cycles/output=3239.047 transfer_bytes/output=4.000 cycles/logical_mac=202.440 logical_macs/cycle=0.005 wall_time_ms=362
```

How to interpret the fields:

| Field group | Meaning | Codesign use |
| --- | --- | --- |
| `runtime` | The actual problem shape and tile knobs used for this run. | Lets experiments compare prefill/decode sizes, tile choices, and runtime prefix shapes. |
| `compiled` | The shape and tile assumptions baked into the generated artifact. | Separates generated-kernel capacity from runtime workload shape. |
| `execution` | Grid/block-derived workgroups, threads, and cache-hit state. | Connects schedule decisions to runtime launch behavior and module/kernel cache effects. |
| `workload` | Logical MACs and element counts. | Normalizes performance across shape changes. |
| `memory` | Logical buffer bytes and observed transfer bytes. | Makes copy/storage costs visible before paged KV and device-device movement are added. |
| `counters` | Vortex `PERF` instructions, cycles, and IPC. | Gives hardware/runtime evidence instead of relying on estimates. |
| `derived` | Per-output and per-MAC ratios. | Makes latest-vs-previous comparisons meaningful even when workload shape changes. |

### 3. Experiment result

```text
attention.experiment result:
  spec_id: attention_prefill_i8_vortex_smoke
  status: Succeeded
  events: 6
  total_transfer_bytes: 512
  correctness: passed=true compared_elements=128 mismatches=0
attention.experiment result jsonl: target/mandrel/vortex/attention_prefill_i8.experiment.jsonl
```

The first-pass experiment result converts the runtime trace into a stable experiment object. Today the event list is compact and summary-derived:

```text
copy -> kernel_launch -> sync -> copy -> perf_counter -> correctness_check
```

This is intentionally small. The next step is to emit true allocation, copy, launch, sync, and cache events from runtime boundaries rather than deriving them from summary fields.

### 4. History and deltas

```text
cargo vortex-trace-attention

records: 6 (showing latest 5)
delta latest vs previous:
  instrs: +0 (+0.00%) latest=165144
  cycles: +0 (+0.00%) latest=414598
  total_transfer_bytes: +0 (+0.00%) latest=512
  wall_time_ms: -1 (-0.28%) latest=358
  ipc: +0.000 (+0.00%) latest=0.398
  instrs/output: +0.000 (+0.00%) latest=1290.188
  cycles/output: +0.000 (+0.00%) latest=3239.047
  transfer_bytes/output: +0.000 (+0.00%) latest=4.000
  cycles/logical_mac: +0.000 (+0.00%) latest=202.440
  logical_macs/cycle: +0.000 (+0.00%) latest=0.005
```

This is the start of the lab loop: change a schedule, ABI, memory model, runtime policy, or kernel lowering, then compare the new trace against history. Exact wall time varies by host and build state; cycle/instruction values come from Vortex `PERF` output for the current smoke.

## Quick start

Inspect the local Vortex setup:

```sh
cargo vortex-status
```

Build/install the source toolchain when needed:

```sh
cargo vortex-toolchain-source
cargo vortex-install
```

Inspect, generate, run, and review the current attention path:

```sh
cargo vortex-plan-attention
cargo vortex-generate-attention
cargo vortex-run-attention
cargo vortex-trace-attention
```

Useful runtime knobs:

```sh
MANDREL_ATTENTION_RUNTIME_SEQUENCE=64 \
MANDREL_ATTENTION_RUNTIME_HEAD_DIM=64 \
cargo vortex-run-attention

MANDREL_ATTENTION_RUNTIME_SCALAR_LAUNCH=1 cargo vortex-run-attention
```

## Workspace map

```text
crates/
  core/             shared shape, dtype, and layout descriptors
  model-ir/         attention and LLM operator IR
  schedule/         attention tiling, layouts, and schedule candidates
  compiler/         model-ir + schedule -> Vortex kernel plans
  kernel-ir/        kernel symbols, signatures, and launch descriptors
  profiler/         estimates, runtime trace parsing, and counters
  experiment/       experiment specs/results, target/memory specs, correctness, runtime events
  device/           device capabilities, memory spaces, buffers, and command buffers
  vortex-backend/   Vortex codegen, ABI validation, artifacts, runtime wrapper, and C ABI
  runtime/          higher-level runtime-facing surfaces and fixtures
  kernels/          kernel catalog and CPU/reference-side helpers
  ggml-adapter/     conservative ggml-style attention probe boundary
  xtask/            clap CLI, toolchain, status, generation, runtime, trace commands
docs/
  mission.md        project mission and codesign framing
  roadmap.md        active milestones and short-term priorities
  design.md         design notes
  mlir.md           MLIR notes
  llm-serving-kv-attention-survey.md
                    survey of LLM serving, KV, attention systems, and Mandrel mapping
external/           local Vortex/LLVM checkouts and builds
```

## Codesign axes

Mandrel is being shaped around full-stack design variables and efficiency objectives:

| Layer | Design-space question | Efficiency objective |
| --- | --- | --- |
| Chip/target | Which RISC-V/Vortex execution resources matter for attention and KV paths? | Higher utilization and lower cycles/token. |
| Memory/storage | How should dense/paged KV, local memory, cache behavior, and page tables be modeled? | Lower bytes/token, fragmentation, and memory stalls. |
| Data movement | Which copies, layout transforms, compression, and future overlaps dominate serving paths? | Lower transfer bytes and hidden/overlapped movement. |
| Runtime/driver | How do allocation, launch, queue, sync, module cache, and transfer overhead affect decode/prefill? | Lower TTFT/TPOT and runtime overhead. |
| Compiler | How should semantic/layout metadata lower into target-specific loops and memory movement? | Better codegen choices per target. |
| Scheduling/layout | Which tiling, batching, P/D phase, and work-assignment policies fit the target? | Better goodput under latency and memory constraints. |
| Kernels/operators | Which attention, softmax, reduction, copy, and KV primitives are worth specializing? | Higher throughput per watt/area/cycle where measurable. |
| Observability/optimization | Can every design change be explained and compared through correctness, trace, and derived metrics? | A repeatable path from measurement to improved design. |

## Roadmap snapshot

The current near-term track is:

1. **Executable attention spine**: done for dense `attention_prefill_i8` on Vortex `simx`.
2. **Runtime trace loop**: done for current smoke, with JSONL history, correctness, and derived metrics.
3. **Experiment skeleton**: in progress; current attention trace can derive and write a companion `ExperimentResult` JSONL.
4. **Runtime event model**: in progress; current event list is summary-derived, true runtime-boundary events are next.
5. **ABI/layout metadata gates**: in progress; codegen/runtime validate the current dense attention ABI and reject unsupported Paged KV metadata.
6. **Target/memory specs and Paged KV legality**: next; model target facts, page tables, GQA/ragged-tail legality, and unsupported layouts before lowering.
7. **Dense key-tiled online lowering**: next; make `key_tile` structural in MLIR, then add online max/sum/accumulator state.

See [`docs/roadmap.md`](docs/roadmap.md) for details.

## Community direction

Mandrel is designed to sit below, not replace, serving/runtime projects:

- **RISC-V / open hardware**: provide workload-driven feedback from real LLM kernels.
- **SGLang-class serving**: use prefill/decode, paged KV, and batching shapes as north-star workloads.
- **llama.cpp / ggml-style runtimes**: provide a future conservative C/C++ boundary for one-op backend probes and local-inference experiments.

The intended contribution is a transparent full-stack optimization loop: generated kernels, artifacts, correctness, traces, and reports that explain how open AI hardware behaves under LLM-serving pressure and which design changes improve efficiency.

## Validation

Recent local validation includes:

```sh
cargo fmt --check
cargo check -p mandrel-kernel-ir -p mandrel-schedule -p mandrel-profiler -p mandrel-experiment -p mandrel-compiler -p mandrel-vortex-backend -p mandrel-ggml-adapter -p mandrel-kernels -p mandrel-runtime -p xtask
cargo test -p mandrel-kernel-ir -p mandrel-schedule -p mandrel-profiler -p mandrel-experiment -p mandrel-compiler -p mandrel-vortex-backend -p mandrel-ggml-adapter -p mandrel-kernels -p mandrel-runtime -p xtask
cargo vortex-run-attention
cargo vortex-trace-attention
```

The Vortex runtime smoke requires a local Vortex toolchain/runtime under `external/` or equivalent environment configuration.

## Status

Mandrel is early and intentionally narrow in implementation, but broad in purpose. The project currently prioritizes one hard vertical slice over many shallow demos. The goal is to keep the attention path executable while gradually turning hardware target specs, memory systems, copies, communication, runtime events, scheduling/layout policies, and experiment reports into first-class objects for full-stack efficiency optimization.
