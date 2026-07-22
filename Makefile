SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ENV_SETUP := $(ROOT)/scripts/env/setup.sh
ENV_FILE := $(ROOT)/scripts/env/vortex-env.sh

.PHONY: \
	help all install setup setup-python setup-verilator setup-llvm setup-vortex \
	env env-check shell plan generate run profile e2e \
	fmt check clippy test no-std validate ci verify

.NOTPARALLEL: all install setup e2e validate ci verify

help:
	@printf '\033[1mMandrel HDL / LLVM / Verilator RTLSim workflow\033[0m\n\n'
	@printf '\033[1;36mEnvironment\033[0m\n'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'install' 'Materialize the complete pinned environment (alias: setup)'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'setup-python' 'Create the frozen uv-managed Python environment'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'setup-verilator' 'Build the pinned project-local Verilator'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'setup-llvm' 'Build/verify LLVM-Vortex and compiler-rt'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'setup-vortex' 'Fetch, patch, and build the Vortex RTLSim runtime'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'env-check' 'Read-only verification of materialized tools and runtime'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'env' 'Print the command that activates the project environment'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'shell' 'Open an interactive Bash with the project environment active'
	@printf '\n\033[1;36mAttention / RTLSim\033[0m\n'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'plan' 'Print the typed attention launch plan'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'generate' 'Generate and validate MLIR/LLVM/object/ELF/vxbin artifacts'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'run' 'Run exact correctness and RTL profiling through Verilator RTLSim'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'profile' 'Alias for run; writes JSON and CSV experiment reports'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'e2e' 'Install the environment, then execute the complete RTL gate'
	@printf '\n\033[1;36mValidation\033[0m\n'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'fmt' 'Check Rust formatting'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'check' 'Check the complete Rust workspace'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'clippy' 'Run Clippy with warnings denied'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'test' 'Run all workspace tests'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'no-std' 'Check no_std crates for the RISC-V target'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'validate' 'Run fmt, check, Clippy, tests, and no_std checks'
	@printf '  \033[32mmake %-18s\033[0m %s\n' 'verify' 'Run environment check, validation, and the RTL integration gate'

all: e2e

install: setup

setup:
	@printf '\n\033[1;36m==> Materializing the pinned Mandrel environment\033[0m\n'
	@"$(ENV_SETUP)" all

setup-python:
	@printf '\n\033[1;36m==> Syncing the uv Python environment\033[0m\n'
	@"$(ENV_SETUP)" python

setup-verilator:
	@printf '\n\033[1;36m==> Building pinned Verilator\033[0m\n'
	@"$(ENV_SETUP)" verilator

setup-llvm:
	@printf '\n\033[1;36m==> Building/verifying LLVM-Vortex\033[0m\n'
	@"$(ENV_SETUP)" llvm-vortex

setup-vortex:
	@printf '\n\033[1;36m==> Building Vortex Verilator RTLSim\033[0m\n'
	@"$(ENV_SETUP)" vortex

env-check:
	@printf '\n\033[1;36m==> Checking the Mandrel environment\033[0m\n'
	@"$(ENV_SETUP)" check

env:
	@printf 'Run in the current shell:\n\n  source %s\n' "$(ENV_FILE)"

shell: env-check
	@printf '\n\033[1;36m==> Opening an environment-enabled Bash; exit to return\033[0m\n'
	@source "$(ENV_FILE)" && exec /bin/bash -i

plan:
	@printf '\n\033[1;36m==> Compiling the attention launch plan\033[0m\n'
	@cd "$(ROOT)" && source "$(ENV_FILE)" && cargo vortex-plan-attention

generate: env-check
	@printf '\n\033[1;36m==> Generating Vortex attention artifacts\033[0m\n'
	@cd "$(ROOT)" && source "$(ENV_FILE)" && cargo vortex-generate-attention

run: env-check
	@printf '\n\033[1;36m==> Running attention through Verilator RTLSim\033[0m\n'
	@cd "$(ROOT)" && source "$(ENV_FILE)" && cargo vortex-run-attention

profile: run

e2e: setup
	@printf '\n\033[1;36m==> Executing the complete Mandrel RTL integration gate\033[0m\n'
	@cd "$(ROOT)" && source "$(ENV_FILE)" && cargo vortex-run-attention

fmt:
	@printf '\n\033[1;36m==> Checking Rust formatting\033[0m\n'
	@cd "$(ROOT)" && cargo fmt --all -- --check

check:
	@printf '\n\033[1;36m==> Checking the Rust workspace\033[0m\n'
	@cd "$(ROOT)" && cargo check --workspace --all-targets --all-features --locked

clippy:
	@printf '\n\033[1;36m==> Running Clippy\033[0m\n'
	@cd "$(ROOT)" && cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

test:
	@printf '\n\033[1;36m==> Running workspace tests\033[0m\n'
	@cd "$(ROOT)" && cargo test --workspace --all-targets --all-features --locked

no-std:
	@printf '\n\033[1;36m==> Checking no_std RISC-V crates\033[0m\n'
	@cd "$(ROOT)" && cargo no-std-check

validate: fmt check clippy test no-std

ci: validate

verify: env-check validate run
