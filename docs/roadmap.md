# Roadmap

This roadmap tracks the path from a single executable Vortex matmul smoke to a workload-driven compiler/operator/architecture co-design platform for LLM inference.

The guiding question is:

> Given an LLM workload, how can we generate better code for a RISC-V GPGPU target and use measured feedback to improve the operator schedule, compiler lowering, runtime policy, and architecture assumptions?

## 1. Current baseline

The current repository already has the first real executable loop:

```text
ggml-like MUL_MAT i8*i8 -> i32
  -> model-ir
  -> schedule
  -> compiler
  -> kernel-ir
  -> generated Vortex C++ or textual Vortex LLVM IR
  -> kernel.vxbin
  -> Rust VortexBackend
  -> Vortex simx
  -> host reference correctness check
```

Current baseline capabilities:

| Area | State |
| --- | --- |
| Direct matmul | Available and validated. |
| Direct Vortex C++ source generation | Default path. |
| Direct Vortex LLVM IR source generation | Opt-in path, validated. |
| Tiled local-memory matmul | Experimental `4x4x32` LLVM IR path, correctness smoke passes, not default. |
| VortexBackend | Runtime/device/queue ownership, module/kernel cache, buffer lifetime, launch validation. |
| Toolchain | Source-built Vortex LLVM + rv64 `compiler-rt` path works on ARM/aarch64. |
| Trace | First runtime summary exists; structured counter ingestion is next. |
| ggml/llama.cpp | Rust-side adapter and minimal C ABI exist; real backend registration is still pending. |

## 2. Milestones

| Milestone | Status | Goal | Exit criteria |
| --- | --- | --- | --- |
| M0: Rust workspace and IR skeleton | Done | Establish strict Rust-first workspace and no-std-friendly core crates. | `fmt`, `clippy`, tests, Miri/no-std policy in place. |
| M1: Vortex executable baseline | Done | Run a custom Vortex matmul kernel from Rust. | `cargo vortex-run-matmul` passes and checks `__vx_kentry_*` / `VXSYMTAB`. |
| M2: Generated kernel paths | Done | Generate Vortex C++ and textual LLVM IR for direct matmul. | Direct C++ and LLVM IR smokes pass. |
| M3: Runtime abstraction | Done | Wrap Vortex runtime in `VortexBackend`. | Backend owns runtime/device/queue, caches modules/kernels, manages buffers. |
| M4: Tiled local-memory experiment | In progress | Validate a schedule-driven local-memory matmul. | Correctness smoke passes; next exit requires trace comparison and policy decision. |
| M5: Structured profiling | Next | Convert runtime and Vortex counters into backend-neutral traces. | Trace records include launch, transfer, cycles, instructions, and memory summaries where available. |
| M6: Minimal ggml backend | Next | Offload one conservative `MUL_MAT i8*i8 -> i32` path from `ggml` / `llama.cpp`. | Unsupported requests fall back cleanly; supported path executes through VortexBackend. |
| M7: LLM kernel family | Later | Add softmax, reductions, quant/layout kernels, attention prefill/decode pieces. | At least one non-matmul operator is executable and traced. |
| M8: Architecture co-design loop | Later | Use workload traces to guide Vortex configuration or ISA/memory experiments. | A reproducible study links schedule/runtime traces to an architecture hypothesis. |

## 3. Short-term plan

### Phase 1: Keep the executable baseline stable

Do not sacrifice the working direct path while experimenting.

Tasks:

- Keep `MatmulI8I32` direct 4x4 as the default available kernel.
- Keep generated Vortex C++ as the default build path.
- Keep direct LLVM IR as an opt-in path through `MANDREL_VORTEX_CODEGEN=llvm-ir`.
- Ensure every Vortex smoke checks both correctness and artifact metadata.
- Preserve launch validation before entering `simx`.
- Add more shape tests, including non-multiple boundaries.

Exit criteria:

```sh
MANDREL_VORTEX_TOOLCHAIN_MODE=skip \
MANDREL_VORTEX_TOOLDIR=external/vortex-source-tools \
cargo vortex-run-matmul
```

continues to pass after every larger compiler/backend change.

### Phase 2: Turn tiled matmul into a real schedule experiment

The experimental tiled path now passes correctness on the current `simx` configuration, but it is slower than the direct baseline. The next work is to make that result useful rather than hiding it.

Tasks:

- Record direct vs tiled metrics under the same shape and Vortex configuration.
- Add structured trace fields for cycles, instructions, IPC, transfer bytes, workgroups, threads, and local memory.
- Add schedule compatibility checks that explain why `16x16x32` is invalid on the current 16-thread workgroup limit.
- Keep `MatmulI8I32Tiled` experimental until a policy decision is justified by trace data.
- Add boundary and randomized correctness tests for tiled schedules.

Exit criteria:

- The roadmap can state why tiled is slower on current `simx`.
- The compiler can report unsupported or suboptimal reasons instead of silently selecting a schedule.

### Phase 3: Generalize kernel artifacts and source descriptors

The project should stop treating matmul as a special case in the backend.

Tasks:

- Add a generic `KernelSymbol -> artifact/source descriptor` registry.
- Represent generated source type: Vortex C++, textual LLVM IR, future intrinsic/assembly path.
- Record required toolchain features per artifact.
- Record launch constraints per artifact.
- Allow `xtask` to build and smoke a selected kernel symbol rather than only `matmul_i8_i32`.

Exit criteria:

- Adding a new kernel does not require duplicating the matmul-specific artifact pipeline.
- `vortex-backend` can report exactly which artifact was loaded and why.

### Phase 4: Implement the first ggml / llama.cpp integration slice

The first integration should be deliberately small and conservative.

Tasks:

- Stabilize the C ABI descriptor for 2D contiguous `i8 * i8 -> i32` matmul.
- Align the Rust `ggml-adapter`, C ABI descriptor, and future C++ shim around the same shape/type/layout contract.
- Keep the public C header under `crates/vortex-backend/include/mandrel_vortex.h` and implement any llama.cpp-specific registration shim in a downstream integration branch or consumer repository.
- Implement an unsupported fallback path in that shim.
- Add a tiny graph smoke test that exercises probe, plan, execute, and fallback behavior.

Exit criteria:

- One conservative `MUL_MAT` call can be routed from a ggml-like boundary into VortexBackend.
- Unsupported cases return a clear reason and fall back.

## 4. Mid-term plan

### 4.1 Expand beyond matmul

The kernel set should grow in the order that maximizes learning for LLM inference.

Recommended order:

1. Row-wise reductions.
2. Elementwise unary/binary kernels.
3. RMSNorm or layernorm-like reduction plus scale.
4. Softmax building blocks.
5. Layout and packing transforms.
6. Quantized matmul variants.
7. Attention prefill pieces: `QK^T`, softmax, `PV`.
8. Decode-oriented KV cache read/update kernels.

Each new kernel should include:

- model IR payload;
- schedule candidates;
- kernel catalog entry;
- generated source path;
- host reference;
- Vortex smoke where possible;
- static metrics;
- measured trace summary.

### 4.2 Attention and KV cache design

Attention should not be implemented as a naive sequence of disconnected kernels unless that is explicitly used as a baseline.

Research tasks:

- Model prefill and decode separately.
- Represent KV cache descriptors in `model-ir`.
- Add layout and paging assumptions to schedule candidates.
- Compare naive `QK^T -> softmax -> PV` with IO-aware fused or staged variants.
- Track global memory traffic as a primary metric.
- Study whether local memory helps or reduces occupancy too much on current Vortex configurations.

### 4.3 Quantization and layout

Quantization must enter the operator contract, not appear as an afterthought in backend code.

Tasks:

- Add explicit quantized dtype and packing descriptors.
- Model dequantization cost and placement.
- Represent layout transforms and packing/unpacking traffic.
- Compare int8 and future int4 paths.
- Identify ISA or intrinsic opportunities for packed dot products.

### 4.4 Runtime buffer planning

The current backend can manage per-run buffers. The next step is workload-level lifetime planning.

Tasks:

- Represent buffer read/write sets in kernel plans.
- Add lifetime intervals for a small graph.
- Reuse buffers where safe.
- Track host/device transfer volume separately from device global memory traffic.
- Prepare for ggml-owned buffers and external memory handles.

## 5. Long-term co-design direction

The long-term goal is to use measured workload behavior to evaluate architecture changes.

Possible architecture axes:

| Axis | Questions |
| --- | --- |
| Threads and warps | Which schedule shapes are valid and efficient under different warp/thread configurations? |
| Local memory | How much local memory is useful for matmul and attention before occupancy drops too much? |
| Cache hierarchy | Which LLM kernels are bandwidth-bound, and where does reuse actually occur? |
| NoC and DRAM | Does KV cache traffic dominate decode? What traffic patterns stress interconnects? |
| ISA extensions | Which packed dot, vector, or tensor-like operations are justified by real traces? |
| Runtime launch overhead | Does fusion or batching matter more than individual kernel speed for small decode steps? |
| Host/device boundary | How much overhead comes from transfer and synchronization rather than compute? |

RTL/Verilator/FPGA should be introduced only after `simx` traces identify a concrete architecture question.

## 6. Research references and design influence

The project is influenced by the following systems and papers, not as dependencies, but as design guidance.

| Direction | Reference | Influence on this project |
| --- | --- | --- |
| End-to-end tensor compiler | TVM | Keep graph/operator/schedule/backend layers separate and measurable. |
| Automatic schedule search | Ansor | Treat schedule as a search space, not a single hard-coded tile. |
| Multi-level IR | MLIR | Use progressive lowering and avoid mixing abstraction levels. |
| Parametric accelerator stack | VTA | Connect compiler schedules with architecture parameters. |
| SoC accelerator evaluation | Gemmini | Measure more than kernel IPC; include runtime and memory hierarchy effects. |
| Dataflow analysis | MAESTRO | Track reuse, traffic, occupancy, and memory behavior explicitly. |
| Sparse/compact DNN hardware | Eyeriss v2 | Treat data movement and memory hierarchy as first-class. |
| IO-aware attention | FlashAttention and FlashAttention-2 | Design attention around memory traffic, not only FLOPs. |
| KV cache runtime | vLLM / PagedAttention | Treat decode memory management as a runtime/compiler problem. |
| Quantization | SmoothQuant / AWQ | Co-design quantized layout, dequant placement, and backend kernels. |
| GPU layout algebra | CuTe | Represent layouts, thread maps, and copy atoms compositionally. |
| Compiler executable boundary | XLA | Separate operator IR from executable plans and backend-specific launch details. |

## 7. Validation strategy

### 7.1 Rust quality gates

For core Rust changes:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo no-std-check
```

For focused backend work, run the narrow package set first:

```sh
cargo test \
  -p mandrel-schedule \
  -p mandrel-model-ir \
  -p mandrel-device \
  -p mandrel-profiler \
  -p mandrel-compiler \
  -p mandrel-vortex-backend \
  -p xtask
```

### 7.2 Vortex validation

For executable backend changes:

```sh
MANDREL_VORTEX_TOOLCHAIN_MODE=skip \
MANDREL_VORTEX_TOOLDIR=external/vortex-source-tools \
cargo vortex-run-matmul

MANDREL_VORTEX_TOOLCHAIN_MODE=skip \
MANDREL_VORTEX_TOOLDIR=external/vortex-source-tools \
MANDREL_VORTEX_CODEGEN=llvm-ir \
cargo vortex-run-matmul

MANDREL_VORTEX_TOOLCHAIN_MODE=skip \
MANDREL_VORTEX_TOOLDIR=external/vortex-source-tools \
cargo vortex-run-matmul-tiled
```

Validation should record:

- host architecture;
- Vortex LLVM version;
- Vortex configuration;
- kernel symbol;
- grid/block/shared memory;
- input shape and dtype;
- correctness result;
- `PERF` counters;
- runtime trace summary.

## 8. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Vortex LLVM version drift | Generated LLVM IR may stop matching backend expectations. | Pin `MANDREL_VORTEX_LLVM_REF`, record LLVM version, keep direct C++ fallback. |
| System LLVM accidentally used | Missing `__vx_kentry_*` and `VXSYMTAB`, broken runtime launch. | Fail fast and document patched LLVM requirement. |
| Dirty `external/vortex` checkout | Reproducibility loss. | Use fork/ref or managed patch staging only. |
| Tiled schedule slower than direct | Incorrect default policy could regress performance. | Keep tiled experimental until trace data justifies selection. |
| simx fidelity limits | Wrong architecture conclusions. | Use simx for fast iteration; move to RTL only for concrete questions. |
| ggml layout complexity | Incorrect offload or wrong results. | Start with contiguous row-major and explicit fallback. |
| Overgrown C ABI | Hard-to-maintain integration boundary. | Expose probe/plan/execute/free only; keep Rust internals private. |
| Matmul tunnel vision | Project stops before LLM-relevant kernels. | Add reductions, softmax, quant/layout, attention, and KV cache workstreams. |

## 9. Suggested next concrete tasks

Recommended immediate next tasks:

1. Add a structured comparison report for direct vs tiled matmul under the same shape and Vortex config.
2. Parse Vortex `PERF` output into `profiler::trace` instead of leaving it as console text.
3. Add non-multiple boundary tests for direct and tiled matmul.
4. Add a generic artifact registry keyed by `KernelSymbol`.
5. Add an unsupported-reason type for schedule and ggml offload decisions.
6. Add the first row-wise reduction kernel to avoid staying matmul-only.
7. Draft the minimal `llama.cpp` backend shim around the existing C ABI.

## 10. Rename plan if Mandrel is approved

If the proposed name is accepted, perform the rename deliberately:

1. Rename root package metadata.
2. Decide whether crate names remain `mandrel-*` for continuity or become `mandrel-*`.
3. Rename C ABI prefixes only if downstream compatibility is not yet important.
4. Rename generated artifact directories only after updating `xtask` and docs.
5. Keep a short compatibility note in the README during transition.

Do not mix the rename with major backend/compiler changes. A clean rename should be easy to review.
