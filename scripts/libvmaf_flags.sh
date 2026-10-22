#!/usr/bin/env bash
# Shared libvmaf discovery for PixVQA build scripts.
# Sets: PIXVQA_VMAF_CFLAGS PIXVQA_VMAF_LIBS PIXVQA_HAS_VMAF (0|1)
# Optional: LIBVMAF_PREFIX (default: $ROOT/deps/libvmaf-3.2.1)

pixvqa_detect_libvmaf() {
  local root="$1"
  local prefix="${LIBVMAF_PREFIX:-$root/deps/libvmaf-3.2.1}"
  PIXVQA_HAS_VMAF=0
  PIXVQA_VMAF_CFLAGS=""
  PIXVQA_VMAF_LIBS=""

  if [[ ! -f "$prefix/include/libvmaf/libvmaf.h" ]]; then
    return 0
  fi

  local pc_dir=""
  if [[ -f "$prefix/lib64/pkgconfig/libvmaf.pc" ]]; then
    pc_dir="$prefix/lib64/pkgconfig"
  elif [[ -f "$prefix/lib/pkgconfig/libvmaf.pc" ]]; then
    pc_dir="$prefix/lib/pkgconfig"
  fi

  if [[ -n "$pc_dir" ]] && command -v pkg-config >/dev/null 2>&1; then
    local old_pc="${PKG_CONFIG_PATH:-}"
    export PKG_CONFIG_PATH="$pc_dir${old_pc:+:$old_pc}"
    if pkg-config --exists libvmaf; then
      PIXVQA_VMAF_CFLAGS="$(pkg-config --cflags libvmaf)"
      PIXVQA_VMAF_LIBS="$(pkg-config --libs --static libvmaf)"
      PIXVQA_HAS_VMAF=1
      export PKG_CONFIG_PATH="$old_pc"
      return 0
    fi
    export PKG_CONFIG_PATH="$old_pc"
  fi

  local libdir=""
  if [[ -f "$prefix/lib64/libvmaf.a" ]]; then
    libdir="$prefix/lib64"
  elif [[ -f "$prefix/lib/libvmaf.a" ]]; then
    libdir="$prefix/lib"
  fi
  if [[ -n "$libdir" ]]; then
    PIXVQA_VMAF_CFLAGS="-I$prefix/include"
    PIXVQA_VMAF_LIBS="-L$libdir -lvmaf -lpthread -lm -lstdc++"
    PIXVQA_HAS_VMAF=1
  fi
}
