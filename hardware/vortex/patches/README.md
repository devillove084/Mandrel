# Vortex patches

Mandrel-specific RTL or simulator changes belong here as reviewable patch series, not as undocumented edits under `external/vortex`.

Each future patch series must identify:

- the Vortex revision from `../source.lock.toml`;
- the hardware design variable it implements;
- SystemVerilog behavior exercised by Verilator RTLSim, plus any FPGA or synthesis implications;
- compiler/ISA exposure, if any;
- correctness tests and synthesis impact;
- the experiment manifests that use it.

Current series against Vortex revision `c992f3f35fa17b83dcc83648a5ac4014b0ea0ac6`:

| Patch | Purpose |
|---|---|
| `0001-rtlsim-allow-non-fatal-verilator-warnings.patch` | Keeps reviewed Verilator warnings visible without making the pinned RTLSim build fail solely because warnings exist. |
| `0002-rtl-expose-mandrel-config-identity-csr.patch` | Includes generated `VX_mandrel.vh`, reserves read-only CSR `0xFC5`, exposes the 64-bit resolved-config association tag in RTL, and adds the kernel-side CSR helper declaration. |

The second patch is exercised end to end by a generated MLIR/LLVM infrastructure probe before attention profiling. The host compares the observed CSR value with the materialized tag and rejects an association-tag mismatch. The probe executes through a separate backend/device instance so it does not alter the attention `PERF` counters.

The full resolved-config SHA-256 remains in the host manifest; the CSR carries only its first 64 bits as an association tag. This runtime check does not replace complete source, patch, build, and toolchain provenance.

The initial TCU/DXA studies use upstream mechanisms first. A Mandrel-specific RTL primitive is added only after workload evidence isolates a narrow operation worth implementing.
