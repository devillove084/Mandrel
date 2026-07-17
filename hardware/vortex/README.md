# Vortex hardware inputs

This directory is the tracked input surface for Mandrel's hardware branch.

```text
source.lock.toml
  + configs/*.toml
  + patches/*
    -> resolved Vortex configuration
    -> generated VX_config.vh / VX_config.h
    -> SimX, RTL simulation, FPGA, or synthesis realization
    -> realized hardware manifest and experiment evidence
```

`external/vortex` and `external/llvm-vortex` are materialized checkouts. They are not the source of experiment identity by themselves. A report must bind the source revisions, resolved configuration, build identity, compiler target, executable artifact, and evidence class.

`configs/current-default.toml` mirrors the current executable SimX baseline. It does not enable TCU or DXA. Future configurations should vary one reviewed hardware factor at a time unless an experiment explicitly defines a factorial design.
