# Experiments

Mandrel experiments are designed and initiated by researchers. The framework records the contract and evidence; it does not automatically choose a historical baseline or recommend the next optimization.

A tracked experiment should state:

- workload semantics and shape;
- software design and lowering;
- hardware design/configuration;
- source and toolchain identities;
- execution backend and evidence class;
- correctness policy;
- controlled variables and hypotheses;
- expected artifacts and report fields.

Current `cargo vortex-run-attention` output is generated under `target/mandrel/vortex/` as:

- `attention_prefill_i8.experiment.json`: complete machine-readable result;
- `attention_prefill_i8.experiment.csv`: one-row summary for aggregation;
- MLIR, LLVM IR, object, ELF, and `.vxbin` artifacts.

Generated results are intentionally not committed by default. Publishable studies should add an explicit manifest and a curated result bundle rather than committing an implicit harness history.
