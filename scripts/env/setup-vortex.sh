#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/vortex-env.sh"

mandrel_require_command git
mandrel_require_command make
mandrel_require_command uv
mandrel_require_file "$MANDREL_PYTHON_VENV_DIR/bin/python" "uv Python; run scripts/env/setup.sh python"
mandrel_require_file "$VERILATOR_PATH/bin/verilator" "project-local Verilator; run scripts/env/setup.sh verilator"
mandrel_require_file "$VORTEX_TOOL_DIR/llvm-vortex/bin/clang" "LLVM-Vortex clang; run scripts/env/setup.sh llvm-vortex"
mandrel_require_file "$VORTEX_TOOL_DIR/llvm-vortex/bin/mlir-translate" "LLVM-Vortex mlir-translate; run scripts/env/setup.sh llvm-vortex"
mandrel_require_file "$VORTEX_TOOL_DIR/libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a" "Vortex compiler-rt builtins"

repository=$(mandrel_lock_value vortex repository)
revision=$(mandrel_lock_value vortex revision)
checkout="$MANDREL_ROOT/$(mandrel_lock_value vortex checkout)"
jobs=$(mandrel_jobs)

mandrel_ensure_checkout "$repository" "$revision" "$checkout"
mandrel_run git -C "$checkout" submodule update --init --recursive
mandrel_apply_vortex_patches "$checkout"

mkdir -p "$VORTEX_BUILD_DIR" "$VORTEX_TOOL_DIR"
mandrel_note "configuring pinned Vortex source"
(
    cd "$VORTEX_BUILD_DIR"
    mandrel_run "$VORTEX_HOME/configure" \
        "--xlen=$MANDREL_VORTEX_XLEN" \
        "--tooldir=$VORTEX_TOOL_DIR" \
        "--prefix=$VORTEX_PATH"
)

export PATH="$MANDREL_ROOT/.venv/bin:$VERILATOR_PATH/bin:/usr/bin:/bin:$VORTEX_TOOL_DIR/llvm-vortex/bin:$VORTEX_TOOL_DIR/riscv64-gnu-toolchain/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

toolchain_stamp="$VORTEX_BUILD_DIR/sw/runtime/mandrel-rtlsim-toolchain.stamp"
toolchain_stamp_tmp="$toolchain_stamp.tmp"
{
    printf 'verilator_path=%s\n' "$VERILATOR_PATH"
    "$VERILATOR_PATH/bin/verilator" --version
    "$MANDREL_ROOT/.venv/bin/python" --version
    /usr/bin/g++ --version | sed -n '1p'
} >"$toolchain_stamp_tmp"
if ! cmp -s "$toolchain_stamp_tmp" "$toolchain_stamp"; then
    mandrel_note "RTLSim toolchain identity changed; rebuilding RTLSim objects"
    mandrel_run make -C "$VORTEX_BUILD_DIR/sw/runtime/rtlsim" \
        "DESTDIR=$VORTEX_BUILD_DIR/sw/runtime" clean
    mv "$toolchain_stamp_tmp" "$toolchain_stamp"
else
    rm -f "$toolchain_stamp_tmp"
fi

mandrel_note "building Vortex software runtime and Verilator RTLSim"
mandrel_run make -C "$VORTEX_HOME/third_party" softfloat ramulator
mandrel_run make -C "$VORTEX_BUILD_DIR/sw/kernel"
mandrel_run make -C "$VORTEX_BUILD_DIR/sw/runtime/stub"
mandrel_run make -C "$VORTEX_BUILD_DIR/sw/runtime/rtlsim" \
    "DESTDIR=$VORTEX_BUILD_DIR/sw/runtime" PERF=1 "THREADS=$jobs"

mandrel_require_file "$VORTEX_BUILD_DIR/sw/runtime/libvortex.so" "Vortex runtime stub"
mandrel_require_file "$VORTEX_BUILD_DIR/sw/runtime/libvortex-rtlsim.so" "Vortex RTLSim driver"
mandrel_require_file "$VORTEX_BUILD_DIR/sw/runtime/librtlsim.so" "Vortex RTLSim core"
mandrel_note "Vortex RTLSim runtime ready"
