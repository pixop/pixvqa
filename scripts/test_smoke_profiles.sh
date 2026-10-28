#!/usr/bin/env bash
# Smoke-test each installed profile: CPU-only, software decoding, and CUDA/NVDEC.
# Each run scores the executable copied into an isolated prefix, with
# LD_LIBRARY_PATH unset, which is what install_pixvqa.sh does.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

bash -n "$0"
[[ -f scripts/qa/cases_cli.py ]] || fail 'missing cases_cli.py'
grep -q 'SUPPORTED_INSTALLATIONS' scripts/qa/cases_cli.py || fail 'smoke profile table missing'
grep -q 'PSNR: gpu' scripts/qa/cases_cli.py || fail 'cuda smoke does not require a GPU PSNR score'

run_installed() {
  local src="$1" label="$2" prefix bin
  [[ -x "$src" ]] || fail "$label binary is absent: $src"
  prefix="$(mktemp -d)"
  PREFIX="$prefix" PIXVQA_BIN="$src" bash scripts/install.sh >/dev/null
  bin="$(readlink -f "$prefix/bin/pixvqa")"
  [[ "$bin" == "$(cd "$prefix" && pwd -P)"/* ]] || fail "$label installed outside $prefix"
  echo "== $label $bin"
  env -u LD_LIBRARY_PATH PIXVQA_QA_CPU="$bin" python3 scripts/run_qa.py --profile smoke
  rm -rf "$prefix"
  echo "OK   $label smoke 16/16 on the installed executable"
}

run_installed build/cpu/pixvqa "cpu-only"
run_installed build/swdec/pixvqa "software-decoding"
run_installed build/full/pixvqa "cuda-nvdec"
echo "All three installation profiles passed smoke QA"
