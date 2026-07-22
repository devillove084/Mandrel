#!/usr/bin/env bash

# Source this file; it performs no downloads or builds.
_mandrel_env_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

export MANDREL_ROOT="$_mandrel_env_root"
export MANDREL_VORTEX_DIR="${MANDREL_VORTEX_DIR:-$_mandrel_env_root/external/vortex}"
export MANDREL_VORTEX_BUILD_DIR="${MANDREL_VORTEX_BUILD_DIR:-$_mandrel_env_root/external/vortex-build}"
export MANDREL_VORTEX_TOOLDIR="${MANDREL_VORTEX_TOOLDIR:-$_mandrel_env_root/external/vortex-source-tools}"
export MANDREL_VERILATOR_DIR="${MANDREL_VERILATOR_DIR:-$_mandrel_env_root/external/verilator-install}"
export MANDREL_PYTHON_VENV_DIR="${MANDREL_PYTHON_VENV_DIR:-$_mandrel_env_root/.venv}"
export MANDREL_VORTEX_XLEN="${MANDREL_VORTEX_XLEN:-64}"

export VORTEX_HOME="$MANDREL_VORTEX_DIR"
export VORTEX_BUILD_DIR="$MANDREL_VORTEX_BUILD_DIR"
export VORTEX_TOOL_DIR="$MANDREL_VORTEX_TOOLDIR"
export VORTEX_PATH="$VORTEX_BUILD_DIR/install"
export VERILATOR_PATH="$MANDREL_VERILATOR_DIR"
export VORTEX_DRIVER=rtlsim

_mandrel_prepend_unique() {
    local name=$1
    local value=$2
    local current=${!name-}
    [[ -d $value ]] || return 0
    case ":$current:" in
        *":$value:"*) return 0 ;;
    esac
    if [[ -n $current ]]; then
        printf -v "$name" '%s:%s' "$value" "$current"
    else
        printf -v "$name" '%s' "$value"
    fi
    export "$name"
}

_mandrel_prepend_unique PATH "$MANDREL_PYTHON_VENV_DIR/bin"
_mandrel_prepend_unique PATH "$VERILATOR_PATH/bin"
_mandrel_prepend_unique PATH "$VORTEX_TOOL_DIR/llvm-vortex/bin"
_mandrel_prepend_unique PATH "$VORTEX_TOOL_DIR/riscv64-gnu-toolchain/bin"
_mandrel_prepend_unique PATH "$VORTEX_BUILD_DIR/sim/rtlsim"
_mandrel_prepend_unique PATH "$VORTEX_PATH/bin"

_mandrel_prepend_unique LD_LIBRARY_PATH "$VORTEX_BUILD_DIR/sw/runtime"
_mandrel_prepend_unique LD_LIBRARY_PATH "$VORTEX_PATH/runtime/lib"
_mandrel_prepend_unique LD_LIBRARY_PATH "$VORTEX_PATH/lib"
_mandrel_prepend_unique LD_LIBRARY_PATH "$VORTEX_TOOL_DIR/llvm-vortex/lib"
_mandrel_prepend_unique PKG_CONFIG_PATH "$VORTEX_PATH/lib/pkgconfig"

unset -f _mandrel_prepend_unique
unset _mandrel_env_root
