# Vortex patches

Mandrel-specific RTL or simulator changes belong here as reviewable patch series, not as undocumented edits under `external/vortex`.

Each future patch series must identify:

- the Vortex revision from `../source.lock.toml`;
- the hardware design variable it implements;
- matching SimX and RTL behavior;
- compiler/ISA exposure, if any;
- correctness tests and synthesis impact;
- the experiment manifests that use it.

The initial TCU/DXA studies use upstream mechanisms first. A Mandrel-specific RTL primitive is added only after workload evidence isolates a narrow operation worth implementing.
