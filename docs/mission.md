# Mission

Mandrel exists to test how open AI hardware should be built for LLM serving across chip architecture, memory hierarchy, data movement, runtime/driver interfaces, compiler lowering, and operator kernels.

It is not just an attention-kernel demo. The current Vortex attention path is the first executable spine of a broader workload-driven full-stack codesign lab.

## Vision

Open AI accelerators should be evaluated with modern serving workloads, not only microbenchmarks. Mandrel aims to make RISC-V/Vortex-like hardware a measurable, programmable, and optimizable target for LLM serving kernels.

The long-term vision is:

> A reproducible lab where LLM serving workloads drive the joint design of open accelerator chips, driver/runtime interfaces, compiler stacks, kernels, memory systems, copies, and communication.

## Mission statement

Mandrel is a workload-driven full-stack codesign lab for open AI accelerators. It starts from LLM attention and KV-cache because they stress the real boundaries between compute, memory, data movement, scheduling, and runtime overhead. The project uses an executable Vortex/RISC-V GPGPU path to turn design ideas into correctness results, traces, metrics, and reports.

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
| Observability | Whether a design change can be measured and explained | JSONL trace history and derived metrics |

## Why LLM serving

LLM serving is a pressure test for AI systems design:

- Prefill stresses parallel attention, memory bandwidth, reductions, and local-memory staging.
- Decode stresses small-batch latency, KV-cache reads, page lookup, copy overhead, and launch/runtime overhead.
- Paged KV exposes storage layout, indirection, gather/scatter, and cache behavior.
- Serving runtimes such as SGLang and llama.cpp/ggml provide concrete workload shapes and integration targets.

Mandrel uses these workloads to ask hardware/software codesign questions that microbenchmarks alone cannot answer.

## Relationship to communities

### RISC-V and open hardware

Mandrel should produce workload-driven feedback for open AI hardware:

- Which memory hierarchy choices matter for attention and KV cache?
- Which copy/runtime features are needed for decode-sized serving paths?
- Which reduction, barrier, vector, or packed-dot features would reduce real kernel cost?
- How do changes show up in instructions, cycles, transfer bytes, and per-output metrics?

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

1. **Workload-driven.** Use attention, KV cache, copy, and communication paths from LLM serving to drive design questions.
2. **Executable spine first.** Every major abstraction should eventually attach to a runnable correctness/trace path.
3. **Correctness before performance.** Host-reference comparison remains a promotion gate.
4. **Observability before optimization.** Design changes need metrics: cycles, instructions, transfer bytes, workload shape, cache behavior, and derived ratios.
5. **Metadata becomes gates.** Layout, ABI, runtime shape, and KV policies should be validated by compiler/runtime code, not only documented.
6. **Narrow public API, rich internal model.** Keep external integration conservative while internal design-space metadata evolves.
7. **One hard vertical slice beats many shallow demos.** The attention path is the first spine; broader layers should grow around it.

## Near-term path

1. Finish the attention ABI/layout metadata gates.
2. Add paged KV legality and page-layout modeling without prematurely lowering it.
3. Make tiled online attention lowering consume `key_tile` structurally.
4. Promote data movement, copy, and runtime event metadata into first-class experiment records.
5. Introduce target/hardware specs derived from Vortex configuration.
6. Turn trace JSONL into higher-level experiment reports.

The project succeeds when it can explain how a cross-layer idea changes a real LLM-serving kernel path on open hardware.