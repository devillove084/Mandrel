# Mission

Mandrel exists to quantify, analyze, and maximize LLM-serving efficiency across the full hardware/software stack: chip architecture, memory hierarchy, KV layout, data movement, runtime/driver interfaces, scheduling policy, compiler lowering, and operator kernels.

It is not just an attention-kernel demo. The current Vortex attention path is the first executable measurement spine of a broader workload-driven full-stack design-space optimization system.

## Vision

Open AI accelerators should be evaluated and improved with modern serving workloads, not only microbenchmarks. Mandrel aims to make RISC-V/Vortex-like hardware a measurable, programmable, and optimizable target for LLM-serving systems.

The long-term vision is:

> A reproducible optimization loop, in the spirit of the quantitative approach to computer architecture, where LLM-serving workloads expose design-space variables, quantify bottlenecks, and drive the joint improvement of open accelerator chips, driver/runtime interfaces, compiler stacks, kernels, memory systems, copies, scheduling, communication, and algorithm choices.

## Mission statement

Mandrel is a workload-driven full-stack design-space quantification, analysis, and optimization system for open AI accelerators. It starts from LLM attention and KV cache because they stress the real boundaries between compute, memory, data movement, scheduling, and runtime overhead. The project uses an executable Vortex/RISC-V GPGPU path to turn design ideas into correctness results, traces, metrics, bottleneck analysis, and reports.

## Current executable spine

The first spine is intentionally narrow but real:

```text
Attention workload
  -> Rust model IR and schedule metadata
  -> Vortex kernel plan and ABI/layout validation
  -> LLVM dialect MLIR
  -> Vortex LLVM object, ELF, and vxbin artifacts
  -> Vortex simx runtime launch
  -> host-reference correctness
  -> PERF, transfer, workload, and history traces
```

This gives Mandrel a stable place to test cross-layer ideas without waiting for a complete custom chip, driver, or production serving runtime.

## Codesign layers

| Layer | What Mandrel wants to test | Current foothold |
| --- | --- | --- |
| Workload | How LLM serving shapes stress hardware and runtime | Dense `attention_prefill_i8`; paged KV next |
| Operators | Attention, softmax/reduction, KV read/write, copy, layout transform | Attention prefill and planned softmax/reduction |
| Compiler | How semantic/layout/schedule metadata lowers to target kernels | Rust plan -> LLVM dialect MLIR |
| Kernel ABI | How buffer slots, scalar args, layouts, and runtime shape policies stay consistent | ABI/layout validation gates |
| Runtime/driver | How launch, allocation, copy, sync, cache, and events affect serving kernels | Vortex runtime wrapper and launch traces |
| Memory/storage | How KV cache, paging, local memory, and transfer paths shape performance | Dense KV metadata; paged KV legality next |
| Data movement | How host-device/device-device copies, layout movement, and future overlap should be modeled | Transfer byte traces today |
| Hardware target | Which RISC-V/Vortex features matter for attention and KV workloads | Vortex simx and source toolchain |
| Optimization loop | Whether a design change can be measured, explained, compared, and improved | JSONL trace history, experiment results, and derived metrics |

## Why LLM serving

LLM serving is a pressure test for AI systems design:

- Prefill stresses parallel attention, memory bandwidth, reductions, and local-memory staging.
- Decode stresses small-batch latency, KV-cache reads, page lookup, copy overhead, and launch/runtime overhead.
- Paged KV exposes storage layout, indirection, gather/scatter, and cache behavior.
- Serving runtimes such as SGLang and llama.cpp/ggml provide concrete workload shapes and integration targets.

Mandrel uses these workloads to ask and answer hardware/software/algorithm optimization questions that microbenchmarks alone cannot answer.

## Relationship to communities

### RISC-V and open hardware

Mandrel should produce quantitative workload-driven feedback for open AI hardware:

- Which memory hierarchy choices matter for attention and KV cache?
- Which copy/runtime features are needed for decode-sized serving paths?
- Which reduction, barrier, vector, or packed-dot features would reduce real kernel cost?
- How do changes show up in instructions, cycles, transfer bytes, per-output metrics, and end-to-end efficiency objectives?

### Algorithm and model design

Mandrel should also feed measurements back into algorithm design:

- Which attention or KV-cache variants are efficient on open hardware, not only accurate on paper?
- Which compression, sparsity, paging, or layout assumptions create hardware-friendly access patterns?
- Which model-side changes reduce bytes/token, launches/token, synchronization, or page-table overhead without unacceptable quality loss?
- Which algorithmic choices should be avoided because they require hardware/runtime features that open targets do not provide yet?

### SGLang

SGLang is a useful north star for serving-shaped workloads: prefill/decode separation, paged KV, batching, and runtime scheduling. Mandrel should first model and replay SGLang-class attention/KV shapes before attempting a production backend.

### llama.cpp / ggml

llama.cpp/ggml is a practical integration direction for portable local inference. Mandrel should eventually provide conservative C/C++ boundaries and one-op backend probes, but only after the internal ABI/layout and trace story is stable.

## Non-goals for now

Mandrel should stay narrow while it becomes deep:

- Not a full LLM framework.
- Not a production SGLang or llama.cpp backend yet.
- Not a CUDA/Triton replacement.
- Not a generic compiler for every operator.
- Not a custom chip RTL project before workload and experiment infrastructure are credible.
- Not a benchmark-only project without correctness, traceability, and reproducibility.

## Operating principles

1. **Quantitative architecture method.** Treat AI workloads the way computer architecture treats benchmarks: characterize them, measure them, model bottlenecks, compare alternatives, and optimize based on evidence.
2. **Optimization-objective driven.** Use attention, KV cache, copy, and communication paths from LLM serving to define efficiency objectives and design variables.
3. **Executable spine first.** Every major abstraction should eventually attach to a runnable correctness/trace path.
4. **Correctness before performance.** Host-reference comparison remains a promotion gate.
5. **Quantification before optimization.** Design changes need metrics: cycles, instructions, transfer bytes, workload shape, cache behavior, correctness, and derived ratios.
6. **Analysis before claims.** A design is better only if the experiment explains which bottleneck moved and why.
7. **Hardware and algorithm co-evolution.** Measurements should guide both open-hardware features and algorithm/model choices.
8. **Metadata becomes gates.** Layout, ABI, runtime shape, and KV policies should be validated by compiler/runtime code, not only documented.
9. **Narrow public API, rich internal model.** Keep external integration conservative while internal design-space metadata evolves.
10. **One hard vertical slice beats many shallow demos.** The attention path is the first spine; broader layers should grow around it.

## Near-term path

1. Finish the attention ABI/layout metadata gates.
2. Add paged KV legality and page-layout modeling without prematurely lowering it.
3. Make tiled online attention lowering consume `key_tile` structurally.
4. Promote data movement, copy, and runtime event metadata into first-class experiment records.
5. Introduce target/hardware specs derived from Vortex configuration.
6. Turn trace JSONL into higher-level experiment reports.

The project succeeds when it can quantify how a cross-layer design change affects a real LLM-serving path on open hardware, explain the bottleneck movement, and guide the next efficiency-improving design decision.