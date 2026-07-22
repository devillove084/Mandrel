#!/usr/bin/env bash

set -euo pipefail

MANDREL_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MANDREL_SOURCE_LOCK="$MANDREL_ROOT/hardware/vortex/source.lock.toml"

mandrel_die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

mandrel_note() {
    printf '==> %s\n' "$*"
}

mandrel_run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

mandrel_require_command() {
    command -v "$1" >/dev/null 2>&1 || mandrel_die "missing required command '$1'"
}

mandrel_lock_value() {
    local section=$1
    local key=$2
    awk -v wanted_section="$section" -v wanted_key="$key" '
        /^\[[^]]+\]$/ {
            current = substr($0, 2, length($0) - 2)
            next
        }
        current == wanted_section && $0 ~ "^[[:space:]]*" wanted_key "[[:space:]]*=" {
            value = $0
            sub(/^[^=]*=[[:space:]]*/, "", value)
            sub(/[[:space:]]*#.*/, "", value)
            gsub(/^"|"$/, "", value)
            print value
            exit
        }
    ' "$MANDREL_SOURCE_LOCK"
}

mandrel_jobs() {
    local jobs=${MANDREL_BUILD_JOBS:-}
    if [[ -z $jobs ]]; then
        jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
        if (( jobs > 8 )); then
            jobs=8
        fi
    fi
    [[ $jobs =~ ^[1-9][0-9]*$ ]] || mandrel_die "MANDREL_BUILD_JOBS must be a positive integer"
    printf '%s\n' "$jobs"
}

mandrel_ensure_checkout() {
    local repository=$1
    local revision=$2
    local checkout=$3

    if [[ ! -d $checkout/.git ]]; then
        [[ ! -e $checkout ]] || mandrel_die "$checkout exists but is not a Git checkout"
        mkdir -p "$(dirname -- "$checkout")"
        mandrel_run git clone --filter=blob:none "$repository" "$checkout"
    fi

    local observed
    observed=$(git -C "$checkout" rev-parse HEAD)
    if [[ $observed != "$revision" ]]; then
        local dirty
        dirty=$(git -C "$checkout" status --short --untracked-files=no)
        [[ -z $dirty ]] || mandrel_die "refusing to checkout $revision over tracked changes in $checkout"
        mandrel_run git -C "$checkout" fetch --depth=1 origin "$revision"
        mandrel_run git -C "$checkout" checkout --detach "$revision"
    fi

    observed=$(git -C "$checkout" rev-parse HEAD)
    [[ $observed == "$revision" ]] || mandrel_die "$checkout revision mismatch: expected $revision, observed $observed"
}

mandrel_apply_vortex_patches() {
    local checkout=$1
    local patch_dir="$MANDREL_ROOT/hardware/vortex/patches"
    local patch
    local allowed=''

    shopt -s nullglob
    for patch in "$patch_dir"/*.patch; do
        if git -C "$checkout" apply --reverse --check "$patch" >/dev/null 2>&1; then
            mandrel_note "patch already applied: ${patch#"$MANDREL_ROOT/"}"
        elif git -C "$checkout" apply --check "$patch" >/dev/null 2>&1; then
            mandrel_run git -C "$checkout" apply "$patch"
        else
            mandrel_die "patch neither applies nor reverses cleanly: ${patch#"$MANDREL_ROOT/"}"
        fi
        while IFS= read -r path; do
            allowed+="${path#b/}"$'\n'
        done < <(awk '/^\+\+\+ b\// { print $2 }' "$patch")
    done
    shopt -u nullglob

    local dirty_path
    while IFS= read -r dirty_path; do
        [[ -z $dirty_path ]] && continue
        grep -Fxq "$dirty_path" <<<"$allowed" || mandrel_die "unreviewed tracked change in Vortex checkout: $dirty_path"
    done < <(git -C "$checkout" diff --name-only)
}

mandrel_require_file() {
    [[ -f $1 ]] || mandrel_die "missing $2: $1"
}
