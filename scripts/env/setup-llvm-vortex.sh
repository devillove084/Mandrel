#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"
source "$(dirname -- "${BASH_SOURCE[0]}")/vortex-env.sh"

for program in uv git cmake ninja make riscv64-unknown-elf-gcc riscv64-unknown-elf-gcc-ar riscv64-unknown-elf-objcopy riscv64-unknown-elf-objdump; do
    mandrel_require_command "$program"
done
mandrel_run uv sync --frozen --project "$MANDREL_ROOT"

llvm_repository=$(mandrel_lock_value llvm_vortex repository)
llvm_revision=$(mandrel_lock_value llvm_vortex revision)
llvm_source="$MANDREL_ROOT/$(mandrel_lock_value llvm_vortex checkout)"
llvm_build=${MANDREL_VORTEX_LLVM_BUILD_DIR:-$MANDREL_ROOT/external/llvm-vortex-build}
compiler_rt_build=${MANDREL_VORTEX_COMPILER_RT_BUILD_DIR:-$MANDREL_ROOT/external/llvm-vortex-compiler-rt-build64}
llvm_prefix="$VORTEX_TOOL_DIR/llvm-vortex"
riscv_toolchain="$VORTEX_TOOL_DIR/riscv64-gnu-toolchain"
riscv_sysroot="$riscv_toolchain/riscv64-unknown-elf"
compiler_rt_prefix="$VORTEX_TOOL_DIR/libcrt64"
jobs=${MANDREL_VORTEX_TOOLCHAIN_JOBS:-$(mandrel_jobs)}

vortex_repository=$(mandrel_lock_value vortex repository)
vortex_revision=$(mandrel_lock_value vortex revision)
mandrel_ensure_checkout "$vortex_repository" "$vortex_revision" "$VORTEX_HOME"
mandrel_run git -C "$VORTEX_HOME" submodule update --init --recursive
mandrel_apply_vortex_patches "$VORTEX_HOME"
mandrel_ensure_checkout "$llvm_repository" "$llvm_revision" "$llvm_source"
mandrel_run git -C "$llvm_source" submodule update --init --recursive



find_riscv_include() {
    local candidate
    if [[ -n ${MANDREL_RISCV_C_INCLUDE_DIR:-} ]]; then
        candidate=$MANDREL_RISCV_C_INCLUDE_DIR
        [[ -f $candidate/newlib.h ]] || mandrel_die "newlib.h not found under MANDREL_RISCV_C_INCLUDE_DIR=$candidate"
        printf '%s\n' "$candidate"
        return
    fi
    for candidate in \
        /usr/lib/picolibc/riscv64-unknown-elf/include \
        /usr/riscv64-unknown-elf/include \
        /usr/lib/riscv64-unknown-elf/include; do
        if [[ -f $candidate/newlib.h ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    mandrel_die "missing RISC-V C headers; install picolibc-riscv64-unknown-elf or set MANDREL_RISCV_C_INCLUDE_DIR"
}

find_riscv_lib() {
    local include_dir=$1
    local candidate
    if [[ -n ${MANDREL_RISCV_C_LIB_DIR:-} ]]; then
        candidate=$MANDREL_RISCV_C_LIB_DIR
        [[ -f $candidate/libc.a && -f $candidate/libm.a ]] || mandrel_die "libc.a/libm.a not found under MANDREL_RISCV_C_LIB_DIR=$candidate"
        printf '%s\n' "$candidate"
        return
    fi
    for candidate in \
        "$(dirname -- "$include_dir")/lib/release/rv64ifd/lp64d" \
        "$(dirname -- "$include_dir")/lib/rv64ifd/lp64d" \
        /usr/lib/picolibc/riscv64-unknown-elf/lib/release/rv64ifd/lp64d \
        /usr/lib/picolibc/riscv64-unknown-elf/lib/rv64ifd/lp64d \
        /usr/riscv64-unknown-elf/lib \
        /usr/lib/riscv64-unknown-elf/lib; do
        if [[ -f $candidate/libc.a && -f $candidate/libm.a ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    mandrel_die "missing non-RVC rv64ifd/lp64d libc.a/libm.a; install picolibc-riscv64-unknown-elf or set MANDREL_RISCV_C_LIB_DIR"
}

replace_link() {
    local link=$1
    local target=$2
    mkdir -p "$(dirname -- "$link")"
    if [[ -L $link || -f $link ]]; then
        rm -f "$link"
    elif [[ -d $link ]]; then
        rmdir "$link" 2>/dev/null || mandrel_die "refusing to replace non-empty directory $link"
    elif [[ -e $link ]]; then
        mandrel_die "refusing to replace unsupported path $link"
    fi
    ln -s "$target" "$link"
}

write_gcc_wrapper() {
    local wrapper=$1
    local real=$2
    local include_dir=$3
    mkdir -p "$(dirname -- "$wrapper")"
    [[ ! -d $wrapper ]] || mandrel_die "refusing to replace directory $wrapper"
    rm -f "$wrapper"
    printf '#!/usr/bin/env bash\nset -euo pipefail\nexec %q -isystem %q "$@"\n' "$real" "$include_dir" >"$wrapper"
    chmod +x "$wrapper"
}

include_dir=$(find_riscv_include)
lib_dir=$(find_riscv_lib "$include_dir")
mkdir -p "$riscv_toolchain/bin"
replace_link "$riscv_sysroot/include" "$include_dir"
replace_link "$riscv_sysroot/lib" "$lib_dir"
replace_link "$VORTEX_TOOL_DIR/libc64/include" "$include_dir"
replace_link "$VORTEX_TOOL_DIR/libc64/lib" "$lib_dir"
write_gcc_wrapper "$riscv_toolchain/bin/riscv64-unknown-elf-gcc" "$(command -v riscv64-unknown-elf-gcc)" "$include_dir"

for suffix in gcc-ar objdump objcopy ar as cpp gcc-nm gcc-ranlib ld ld.bfd nm ranlib readelf size strings strip; do
    name="riscv64-unknown-elf-$suffix"
    if command -v "$name" >/dev/null 2>&1; then
        replace_link "$riscv_toolchain/bin/$name" "$(command -v "$name")"
    fi
done

mkdir -p "$VORTEX_BUILD_DIR" "$VORTEX_TOOL_DIR"
(
    cd "$VORTEX_BUILD_DIR"
    mandrel_run "$VORTEX_HOME/configure" --xlen=64 "--tooldir=$VORTEX_TOOL_DIR" "--prefix=$VORTEX_PATH"
)

required=(
    "$llvm_prefix/bin/clang"
    "$llvm_prefix/bin/mlir-opt"
    "$llvm_prefix/bin/mlir-translate"
    "$compiler_rt_prefix/lib/baremetal/libclang_rt.builtins-riscv64.a"
    "$VORTEX_BUILD_DIR/sw/kernel/libvortex2.a"
)
ready=1
for path in "${required[@]}"; do
    [[ -f $path ]] || ready=0
done
if (( ready == 1 )) && [[ ${MANDREL_FORCE_REBUILD:-0} != 1 ]]; then
    mandrel_note "pinned LLVM-Vortex, compiler-rt, and Vortex kernel runtime are already installed"
    exit 0
fi

host_target=RISCV
case $(uname -m) in
    x86_64) host_target='RISCV;X86' ;;
    aarch64) host_target='RISCV;AArch64' ;;
    arm*) host_target='RISCV;ARM' ;;
esac
llvm_projects=${MANDREL_VORTEX_LLVM_PROJECTS:-clang\;lld\;mlir}
llvm_targets=${MANDREL_VORTEX_LLVM_TARGETS:-$host_target}

mandrel_note "configuring LLVM-Vortex at pinned revision $llvm_revision"
mandrel_run cmake -G Ninja \
    -S "$llvm_source/llvm" -B "$llvm_build" \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_INSTALL_PREFIX=$llvm_prefix" \
    "-DLLVM_ENABLE_PROJECTS=$llvm_projects" \
    "-DLLVM_TARGETS_TO_BUILD=$llvm_targets" \
    -DBUILD_SHARED_LIBS=ON \
    -DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DMLIR_INCLUDE_TESTS=OFF
mandrel_run cmake --build "$llvm_build" --parallel "$jobs"
mandrel_run cmake --install "$llvm_build"

export PATH="$llvm_prefix/bin:$MANDREL_ROOT/.venv/bin:/usr/bin:/bin:$riscv_toolchain/bin"
export LD_LIBRARY_PATH="$llvm_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mandrel_run make -C "$VORTEX_BUILD_DIR/sw/kernel"
probe_dir="$compiler_rt_build/mandrel-probe"
mkdir -p "$probe_dir"
printf 'void mandrel_vortex_feature_probe(void) {}\n' >"$probe_dir/probe.c"
mandrel_run "$llvm_prefix/bin/clang" \
    --target=riscv64-unknown-elf "--sysroot=$riscv_sysroot" "--gcc-toolchain=$riscv_toolchain" \
    -march=rv64imafd -mabi=lp64d \
    -Xclang -target-feature -Xclang +xvortex \
    -Xclang -target-feature -Xclang +zicond \
    -mcmodel=medany -c "$probe_dir/probe.c" -o "$probe_dir/probe.o"

lld="$llvm_prefix/bin/ld.lld"
[[ -x $lld ]] || lld="$llvm_prefix/bin/lld"
mandrel_require_file "$lld" "LLVM-Vortex linker"
c_flags="--gcc-toolchain=$riscv_toolchain -march=rv64imafd -mabi=lp64d -Xclang -target-feature -Xclang +xvortex -Xclang -target-feature -Xclang +zicond -mcmodel=medany -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections"
asm_flags="--target=riscv64-unknown-elf $c_flags"
linker_flags="-fuse-ld=lld -nostartfiles -Wl,-Bstatic,--gc-sections,-T,$VORTEX_HOME/sw/kernel/scripts/link64.ld,--defsym=STARTUP_ADDR=0x180000000 $VORTEX_BUILD_DIR/sw/kernel/libvortex.a"

mandrel_run cmake -G Ninja \
    -S "$llvm_source/compiler-rt" -B "$compiler_rt_build" \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_INSTALL_PREFIX=$compiler_rt_prefix" \
    "-DCMAKE_AR=$llvm_prefix/bin/llvm-ar" \
    "-DCMAKE_LINKER=$lld" \
    "-DCMAKE_NM=$llvm_prefix/bin/llvm-nm" \
    "-DCMAKE_RANLIB=$llvm_prefix/bin/llvm-ranlib" \
    "-DCMAKE_C_COMPILER=$llvm_prefix/bin/clang" \
    -DCMAKE_C_COMPILER_TARGET=riscv64-unknown-elf \
    "-DCMAKE_C_FLAGS=$c_flags" \
    "-DCMAKE_ASM_COMPILER=$llvm_prefix/bin/clang" \
    -DCMAKE_ASM_COMPILER_TARGET=riscv64-unknown-elf \
    "-DCMAKE_ASM_FLAGS=$asm_flags" \
    "-DCMAKE_EXE_LINKER_FLAGS=$linker_flags" \
    "-DCMAKE_SYSROOT=$riscv_sysroot" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCOMPILER_RT_OS_DIR=baremetal \
    -DCOMPILER_RT_DEFAULT_TARGET_TRIPLE=riscv64-unknown-elf \
    -DCOMPILER_RT_BUILD_BUILTINS=ON \
    -DCOMPILER_RT_BUILD_CRT=OFF \
    -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCOMPILER_RT_BUILD_ORC=OFF \
    -DCOMPILER_RT_BUILD_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=OFF \
    -DCOMPILER_RT_BUILD_STANDALONE_LIBATOMIC=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BUILD_XRAY_NO_PREINIT=OFF \
    -DCOMPILER_RT_BAREMETAL_BUILD=ON \
    -DCOMPILER_RT_INCLUDE_TESTS=OFF
mandrel_run cmake --build "$compiler_rt_build" --parallel "$jobs"
mandrel_run cmake --install "$compiler_rt_build"

for path in "${required[@]}"; do
    mandrel_require_file "$path" "LLVM-Vortex toolchain artifact"
done
mandrel_note "LLVM-Vortex source toolchain ready"
