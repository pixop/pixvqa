#!/usr/bin/env bash
# Discover the pinned PixVQA libvmaf without falling back to /usr/local.
pixvqa_detect_libvmaf() {
  local root="$1" prefix="${LIBVMAF_PREFIX:-$1/deps/libvmaf-3.2.1}"
  local libdir="" candidate
  PIXVQA_HAS_VMAF=0
  PIXVQA_VMAF_CFLAGS=""
  PIXVQA_VMAF_LIBS=""
  [[ -f "$prefix/include/libvmaf/libvmaf.h" ]] || return 0
  for candidate in "$prefix/lib" "$prefix/lib64" "$prefix/lib/x86_64-linux-gnu"; do
    if [[ -f "$candidate/libvmaf.a" ]]; then libdir="$candidate"; break; fi
  done
  if [[ -z "$libdir" ]]; then
    echo "WARNING: pinned libvmaf headers found but libvmaf.a missing under $prefix" >&2
    return 0
  fi
  PIXVQA_VMAF_CFLAGS="-I$prefix/include"
  PIXVQA_VMAF_LIBS="-L$libdir -lvmaf -pthread -lm"
  PIXVQA_HAS_VMAF=1
}
