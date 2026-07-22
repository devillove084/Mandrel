#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/vortex-env.sh"

failed=0
check_file() {
    local path=$1
    local label=$2
    if [[ -f $path ]]; then
        printf 'ok   %-28s %s\n' "$label" "${path#"$MANDREL_ROOT/"}"
    else
        printf 'MISS %-28s %s\n' "$label" "${path#"$MANDREL_ROOT/"}"
        failed=1
    fi
}

check_revision() {
    local checkout=$1
    local expected=$2
    local label=$3
    if [[ ! -d $checkout/.git ]]; then
        printf 'MISS %-28s %s\n' "$label" "${checkout#"$MANDREL_ROOT/"}"
        failed=1
        return
    fi
    local observed
    observed=$(git -C "$checkout" rev-parse HEAD)
    if [[ $observed == "$expected" ]]; then
        printf 'ok   %-28s %s\n' "$label" "$observed"
    else
        printf 'BAD  %-28s expected=%s observed=%s\n' "$label" "$expected" "$observed"
        failed=1
    fi
}

python_version=$(mandrel_lock_value python version)
check_file "$MANDREL_ROOT/.venv/bin/python" "uv Python"
if [[ -x $MANDREL_ROOT/.venv/bin/python ]]; then
    observed_python=$($MANDREL_ROOT/.venv/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    [[ $observed_python == "$python_version" ]] || { printf 'BAD  %-28s expected=%s observed=%s\n' "Python version" "$python_version" "$observed_python"; failed=1; }
    python_base=$($MANDREL_ROOT/.venv/bin/python -c 'import sys; print(sys.base_prefix)')
    [[ $python_base != *conda* && $python_base != *miniconda* && $python_base != *anaconda* ]] || { printf 'BAD  %-28s %s\n' "Python provider" "$python_base"; failed=1; }
fi

check_revision "$MANDREL_ROOT/$(mandrel_lock_value verilator checkout)" "$(mandrel_lock_value verilator revision)" "Verilator revision"
check_file "$VERILATOR_PATH/bin/verilator" "Verilator binary"
check_revision "$VORTEX_HOME" "$(mandrel_lock_value vortex revision)" "Vortex revision"
check_revision "$MANDREL_ROOT/$(mandrel_lock_value llvm_vortex checkout)" "$(mandrel_lock_value llvm_vortex revision)" "LLVM-Vortex revision"
check_file "$VORTEX_TOOL_DIR/llvm-vortex/bin/clang" "LLVM-Vortex clang"
check_file "$VORTEX_TOOL_DIR/llvm-vortex/bin/mlir-translate" "MLIR translator"
check_file "$VORTEX_TOOL_DIR/libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a" "compiler-rt builtins"
check_file "$VORTEX_TOOL_DIR/libc64/lib/libc.a" "RISC-V libc"
check_file "$VORTEX_TOOL_DIR/libc64/lib/libm.a" "RISC-V libm"
if [[ -e $VORTEX_TOOL_DIR/libc64/lib ]]; then
    libc_dir=$(realpath "$VORTEX_TOOL_DIR/libc64/lib")
    if [[ $libc_dir == */rv64ifd/lp64d ]]; then
        printf 'ok   %-28s %s\n' "RISC-V libc multilib" "$libc_dir"
    else
        printf 'BAD  %-28s expected=rv64ifd/lp64d observed=%s\n' "RISC-V libc multilib" "$libc_dir"
        failed=1
    fi
fi
check_file "$VORTEX_BUILD_DIR/sw/runtime/libvortex.so" "Vortex runtime"
check_file "$VORTEX_BUILD_DIR/sw/runtime/libvortex-rtlsim.so" "RTLSim driver"
check_file "$VORTEX_BUILD_DIR/sw/runtime/librtlsim.so" "RTLSim core"
check_file "$MANDREL_VORTEX_CONFIG_MANIFEST" "Vortex config manifest"
check_file "$MANDREL_VORTEX_CONFIG_SHA256_FILE" "Vortex config SHA-256"
check_file "$MANDREL_VORTEX_CONFIG_TAG_FILE" "Vortex config tag"
check_file "$VORTEX_BUILD_DIR/hw/VX_mandrel.vh" "Vortex config Verilog"

if [[ -x $MANDREL_ROOT/.venv/bin/python \
    && -f $MANDREL_VORTEX_CONFIG_MANIFEST \
    && -f $MANDREL_VORTEX_CONFIG_SHA256_FILE \
    && -f $MANDREL_VORTEX_CONFIG_TAG_FILE \
    && -f $VORTEX_BUILD_DIR/hw/VX_mandrel.vh ]]; then
    if "$MANDREL_ROOT/.venv/bin/python" \
        "$MANDREL_ROOT/scripts/env/materialize-vortex-config.py" \
        --root "$MANDREL_ROOT" \
        --source-config "$VORTEX_HOME/VX_config.toml" \
        --generator "$VORTEX_HOME/ci/gen_config.py" \
        --build-dir "$VORTEX_BUILD_DIR" \
        --realization-profile "$MANDREL_VORTEX_REALIZATION_PROFILE" \
        --generator-cflags "$MANDREL_VORTEX_RTLSIM_CONFIGS -DVX_CFG_XLEN=$MANDREL_VORTEX_XLEN" \
        --check; then
        printf 'ok   %-28s %s\n' "Vortex config integrity" "tag=$(<"$MANDREL_VORTEX_CONFIG_TAG_FILE")"
    else
        printf 'BAD  %-28s %s\n' "Vortex config integrity" "stale or inconsistent generated files"
        failed=1
    fi
fi

if (( failed != 0 )); then
    mandrel_die "environment check failed; run scripts/env/setup.sh all"
fi
mandrel_note "Mandrel HDL/LLVM/RTLSim environment is ready"
