#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

mandrel_require_command uv
mandrel_require_command git
mandrel_require_command autoconf
mandrel_require_command flex
mandrel_require_command bison

mandrel_require_command perl
mandrel_require_command make
mandrel_require_file /usr/include/FlexLexer.h "Flex C++ development header (libfl-dev)"

mandrel_note "syncing uv-managed Python environment"
mandrel_run uv sync --frozen --project "$MANDREL_ROOT"

repository=$(mandrel_lock_value verilator repository)
revision=$(mandrel_lock_value verilator revision)
version=$(mandrel_lock_value verilator version)
checkout="$MANDREL_ROOT/$(mandrel_lock_value verilator checkout)"
prefix="$MANDREL_ROOT/$(mandrel_lock_value verilator prefix)"
binary="$prefix/bin/verilator"
jobs=$(mandrel_jobs)

mandrel_ensure_checkout "$repository" "$revision" "$checkout"

if [[ -x $binary ]] && "$binary" --version | grep -Fq "Verilator $version"; then
    mandrel_note "$($binary --version) already installed at ${prefix#"$MANDREL_ROOT/"}"
    exit 0
fi

mandrel_note "building Verilator $version with uv Python and system GCC"
(
    cd "$checkout"
    PATH="$MANDREL_ROOT/.venv/bin:/usr/bin:/bin" mandrel_run autoconf
    PATH="$MANDREL_ROOT/.venv/bin:/usr/bin:/bin" CC=/usr/bin/gcc CXX=/usr/bin/g++ \
        mandrel_run "$checkout/configure" "--prefix=$prefix"
    PATH="$MANDREL_ROOT/.venv/bin:/usr/bin:/bin" CC=/usr/bin/gcc CXX=/usr/bin/g++ \
        mandrel_run make "-j$jobs" verilator_exe
    PATH="$MANDREL_ROOT/.venv/bin:/usr/bin:/bin" CC=/usr/bin/gcc CXX=/usr/bin/g++ \
        mandrel_run make installbin installredirect installdata install-msg
)

mandrel_require_file "$binary" "project-local Verilator"
"$binary" --version | grep -Fq "Verilator $version" || mandrel_die "installed Verilator version does not match $version"
mandrel_note "$($binary --version)"
