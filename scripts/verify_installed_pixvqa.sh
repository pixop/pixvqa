#!/usr/bin/env bash
# Confirm the installed PixVQA executable in a clean dynamic-linker environment.
# Usage: scripts/verify_installed_pixvqa.sh PREFIX BUILD_KIND DECODE
# BUILD_KIND is "cpu" or "cuda". DECODE is "none", "software", or "software nvdec".
# LD_LIBRARY_PATH is removed so a private RPATH has to resolve every dependency.
set -euo pipefail

prefix="${1:?Set the install prefix}"
expect_build="${2:?Set the expected build kind}"
expect_decode="${3:?Set the expected decode line}"
bin="$prefix/bin/pixvqa"

[[ -x "$bin" ]] || { echo "STOP: $bin is not executable" >&2; exit 1; }
if [[ -e "$prefix/bin/pixvqa-with-libs" ]]; then
  echo "STOP: $prefix/bin/pixvqa-with-libs is present. The installed program is bin/pixvqa." >&2
  exit 1
fi

report="$(env -u LD_LIBRARY_PATH -u LIBRARY_PATH ldd "$bin" 2>&1)" || true
printf '%s\n' "$report"
if grep -Fq 'not found' <<< "$report"; then
  echo "STOP: $bin has an unresolved runtime library without LD_LIBRARY_PATH." >&2
  exit 1
fi

version="$(env -i PATH="/usr/bin:/bin" HOME="${HOME:-/tmp}" "$bin" --version)"
printf '%s\n' "$version"
grep -qx "build: ${expect_build}" <<< "$version" || {
  echo "STOP: expected build: ${expect_build}" >&2
  exit 1
}
grep -qx "decode: ${expect_decode}" <<< "$version" || {
  echo "STOP: expected decode: ${expect_decode}" >&2
  exit 1
}
echo "PASS: $bin reports build: ${expect_build}, decode: ${expect_decode}, and its libraries resolve"
