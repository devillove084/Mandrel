#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Usage: scripts/env/setup.sh <command>

Commands:
  python        Create the frozen uv-managed Python environment
  verilator     Build the pinned project-local Verilator
  llvm-vortex   Build/verify the pinned LLVM-Vortex toolchain
  vortex        Fetch, patch, and build the Vortex RTLSim runtime
  check         Read-only verification of all required artifacts
  all           Run python, verilator, llvm-vortex, vortex, and check
EOF
}

command=${1:-}
case "$command" in
    python)
        exec uv sync --frozen --project "$SCRIPT_DIR/../.."
        ;;
    verilator)
        exec "$SCRIPT_DIR/setup-verilator.sh"
        ;;
    llvm-vortex)
        exec "$SCRIPT_DIR/setup-llvm-vortex.sh"
        ;;
    vortex)
        exec "$SCRIPT_DIR/setup-vortex.sh"
        ;;
    check)
        exec "$SCRIPT_DIR/check.sh"
        ;;
    all)
        "$SCRIPT_DIR/setup.sh" python
        "$SCRIPT_DIR/setup.sh" verilator
        "$SCRIPT_DIR/setup.sh" llvm-vortex
        "$SCRIPT_DIR/setup.sh" vortex
        exec "$SCRIPT_DIR/setup.sh" check
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
