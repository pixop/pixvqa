#!/usr/bin/env bash
# Build tools/vmaf_ref_oracle against pinned libvmaf 3.2.1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/libvmaf_flags.sh
source "$ROOT/scripts/libvmaf_flags.sh"
pixvqa_detect_libvmaf "$ROOT"
if [[ "$PIXVQA_HAS_VMAF" != "1" ]]; then
  echo "libvmaf not found; run scripts/build_libvmaf.sh" >&2
  exit 1
fi
CC="${CC:-gcc}"
CXX="${CXX:-g++}"
OUT="$ROOT/build/cpu/vmaf_ref_oracle"
mkdir -p "$(dirname "$OUT")"
echo "CC tools/vmaf_ref_oracle.c -> $OUT"
# Compile as C, link with g++ so libvmaf's C++ (SVM) symbols resolve.
$CC -O2 -std=c11 -Wall -Wextra $PIXVQA_VMAF_CFLAGS -c \
  "$ROOT/tools/vmaf_ref_oracle.c" -o "$OUT.o"
$CXX -O2 -o "$OUT" "$OUT.o" $PIXVQA_VMAF_LIBS
rm -f "$OUT.o"
echo "OK $OUT"
