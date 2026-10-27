#!/usr/bin/env bash
# Shared libvmaf discovery for PixVQA build scripts.
# Sets: PIXVQA_VMAF_CFLAGS PIXVQA_VMAF_LIBS PIXVQA_HAS_VMAF (0|1)
# Optional: LIBVMAF_PREFIX (default: $ROOT/deps/libvmaf-74f930d)
# The HDR pin is required. deps/libvmaf-3.2.1 is not a fallback: that
# library drops conversion_target and must not be linked for HDR models.

pixvqa_detect_libvmaf() {
  local root="$1"
  local prefix="${LIBVMAF_PREFIX:-$root/deps/libvmaf-74f930d}"
  PIXVQA_HAS_VMAF=0
  PIXVQA_VMAF_CFLAGS=""
  PIXVQA_VMAF_LIBS=""

  if [[ ! -f "$prefix/include/libvmaf/libvmaf.h" ]]; then
    return 0
  fi
  if ! grep -q vmaf_picture_convert "$prefix/include/libvmaf/picture.h" 2>/dev/null; then
    echo "libvmaf_flags: $prefix has no vmaf_picture_convert; refusing to link it" >&2
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
    local zimg_pc=""
    if [[ -f "$root/deps/zimg-3.0.6/lib64/pkgconfig/zimg.pc" ]]; then
      zimg_pc="$root/deps/zimg-3.0.6/lib64/pkgconfig"
    elif [[ -f "$root/deps/zimg-3.0.6/lib/pkgconfig/zimg.pc" ]]; then
      zimg_pc="$root/deps/zimg-3.0.6/lib/pkgconfig"
    fi
    export PKG_CONFIG_PATH="$pc_dir${zimg_pc:+:$zimg_pc}${old_pc:+:$old_pc}"
    if pkg-config --exists libvmaf; then
      PIXVQA_VMAF_CFLAGS="$(pkg-config --cflags libvmaf)"
      PIXVQA_VMAF_LIBS="$(pkg-config --libs --static libvmaf)"
      if [[ "$PIXVQA_VMAF_LIBS" != *"-lzimg"* ]]; then
        echo "libvmaf_flags: $prefix does not link zimg; refusing HDR-incapable libvmaf" >&2
        export PKG_CONFIG_PATH="$old_pc"
        return 0
      fi
      PIXVQA_HAS_VMAF=1
      pixvqa_add_libvmaf_stamp "$prefix"
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
    local zimg_lib=""
    if [[ -f "$root/deps/zimg-3.0.6/lib64/libzimg.a" ]]; then
      zimg_lib="-L$root/deps/zimg-3.0.6/lib64"
    elif [[ -f "$root/deps/zimg-3.0.6/lib/libzimg.a" ]]; then
      zimg_lib="-L$root/deps/zimg-3.0.6/lib"
    fi
    if [[ -z "$zimg_lib" ]]; then
      echo "libvmaf_flags: static libvmaf without a zimg archive; refusing" >&2
      return 0
    fi
    PIXVQA_VMAF_CFLAGS="-I$prefix/include"
    PIXVQA_VMAF_LIBS="-L$libdir -lvmaf $zimg_lib -lzimg -lpthread -lm -lstdc++"
    PIXVQA_HAS_VMAF=1
    pixvqa_add_libvmaf_stamp "$prefix"
  fi
}

pixvqa_add_libvmaf_stamp() {
  local prefix="$1"
  local stamp="$prefix/pixvqa-libvmaf-stamp.txt"
  [[ -f "$stamp" ]] || return 0
  local commit zimg
  commit="$(awk -F= '$1=="libvmaf_commit"{print $2}' "$stamp")"
  zimg="$(awk -F= '$1=="zimg_version"{print $2}' "$stamp")"
  if [[ -n "$commit" ]]; then
    PIXVQA_VMAF_CFLAGS+=" -DPIXVQA_LIBVMAF_COMMIT=${commit}"
  fi
  if [[ -n "$zimg" ]]; then
    PIXVQA_VMAF_CFLAGS+=" -DPIXVQA_ZIMG_VERSION=${zimg}"
  fi
}
