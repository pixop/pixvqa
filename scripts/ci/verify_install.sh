#!/usr/bin/env bash
# Checks produced by the guided installer. Does not compile PixVQA itself.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
bin="$root/bin/pixvqa"
[[ -x "$bin" ]] || { printf 'FAIL installed executable is missing: %s\n' "$bin" >&2; exit 1; }
real_bin="$(readlink -f "$bin")"
real_root="$(readlink -f "$root")"
case "$real_bin" in
    "$real_root"/*) ;;
    *) printf 'FAIL installed executable is outside the checkout: %s\n' "$real_bin" >&2; exit 1 ;;
esac
mkdir -p "$root/reports/qa"
version="$root/reports/qa/install-version.txt"
libs="$root/reports/qa/install-ldd.txt"
"$bin" --version
env -u LD_LIBRARY_PATH "$bin" --version | tee "$version"
env -u LD_LIBRARY_PATH ldd "$bin" | tee "$libs"
if grep -q 'not found' "$libs"; then
    printf 'FAIL a required shared library did not resolve\n' >&2
    exit 1
fi
grep -qx 'build: cpu' "$version"
grep -qx 'decode: software' "$version"
env -u LD_LIBRARY_PATH python3 scripts/run_qa.py --profile smoke
