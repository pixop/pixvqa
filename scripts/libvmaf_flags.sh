#!/usr/bin/env bash
# Shared libvmaf discovery for PixVQA build scripts.
# Sets: PIXVQA_VMAF_CFLAGS PIXVQA_VMAF_LIBS PIXVQA_HAS_VMAF (0|1)
# Optional: LIBVMAF_PREFIX (default: $ROOT/deps/libvmaf-74f930d)
# The HDR pin is required. deps/libvmaf-3.2.1 is not a fallback: that
# library drops conversion_target and must not be linked for HDR models.
# Search lib, lib64, and lib/x86_64-linux-gnu under the prefix only.
# Do not accept a pkg-config result that resolves outside that prefix.

pixvqa_detect_libvmaf() {
  local root="$1"
  local prefix="${LIBVMAF_PREFIX:-$root/deps/libvmaf-74f930d}"
  local libdir="" candidate pc_dir=""
  PIXVQA_HAS_VMAF=0
  PIXVQA_VMAF_CFLAGS=""
  PIXVQA_VMAF_LIBS=""

  [[ -f "$prefix/include/libvmaf/libvmaf.h" ]] || return 0
  if ! grep -q vmaf_picture_convert "$prefix/include/libvmaf/picture.h" 2>/dev/null; then
    echo "libvmaf_flags: $prefix has no vmaf_picture_convert; refusing to link it" >&2
    return 0
  fi

  for candidate in "$prefix/lib" "$prefix/lib64" "$prefix/lib/x86_64-linux-gnu"; do
    if [[ -f "$candidate/libvmaf.a" ]]; then
      libdir="$candidate"
      break
    fi
  done
  if [[ -z "$libdir" ]]; then
    echo "WARNING: pinned libvmaf headers found but libvmaf.a missing under $prefix" >&2
    return 0
  fi

  for candidate in \
      "$prefix/lib64/pkgconfig" \
      "$prefix/lib/pkgconfig" \
      "$prefix/lib/x86_64-linux-gnu/pkgconfig"; do
    if [[ -f "$candidate/libvmaf.pc" ]]; then
      pc_dir="$candidate"
      break
    fi
  done

  if [[ -n "$pc_dir" ]] && command -v pkg-config >/dev/null 2>&1; then
    local old_pc="${PKG_CONFIG_PATH:-}"
    local zimg_pc=""
    zimg_pc="$(pixvqa_zimg_pc_dir "$root")"
    export PKG_CONFIG_PATH="$pc_dir${zimg_pc:+:$zimg_pc}${old_pc:+:$old_pc}"
    local resolved=""
    resolved="$(pkg-config --variable=pcfiledir libvmaf 2>/dev/null || true)"
    if [[ "$resolved" == "$pc_dir" ]] && pkg-config --exists libvmaf; then
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
    echo "libvmaf_flags: pkg-config resolved '$resolved', expected '$pc_dir'; refusing" >&2
    export PKG_CONFIG_PATH="$old_pc"
  fi

  local zimg_lib=""
  zimg_lib="$(pixvqa_zimg_lib_flag "$root")"
  if [[ -z "$zimg_lib" ]]; then
    echo "libvmaf_flags: static libvmaf without a zimg archive; refusing" >&2
    return 0
  fi
  PIXVQA_VMAF_CFLAGS="-I$prefix/include"
  PIXVQA_VMAF_LIBS="-L$libdir -lvmaf $zimg_lib -lzimg -lpthread -lm -lstdc++"
  PIXVQA_HAS_VMAF=1
  pixvqa_add_libvmaf_stamp "$prefix"
}

pixvqa_zimg_pc_dir() {
  local root="$1" candidate
  for candidate in \
      "$root/deps/zimg-3.0.6/lib64/pkgconfig" \
      "$root/deps/zimg-3.0.6/lib/pkgconfig" \
      "$root/deps/zimg-3.0.6/lib/x86_64-linux-gnu/pkgconfig"; do
    if [[ -f "$candidate/zimg.pc" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
}

pixvqa_zimg_lib_flag() {
  local root="$1" candidate
  for candidate in \
      "$root/deps/zimg-3.0.6/lib64" \
      "$root/deps/zimg-3.0.6/lib" \
      "$root/deps/zimg-3.0.6/lib/x86_64-linux-gnu"; do
    if [[ -f "$candidate/libzimg.a" ]]; then
      printf '%s\n' "-L$candidate"
      return 0
    fi
  done
}

pixvqa_refuse_libvmaf() {
  echo "libvmaf_flags: $1" >&2
  PIXVQA_HAS_VMAF=0
  PIXVQA_VMAF_CFLAGS=""
  PIXVQA_VMAF_LIBS=""
}

pixvqa_add_libvmaf_stamp() {
  local prefix="$1"
  local stamp="$prefix/pixvqa-libvmaf-stamp.txt"
  if [[ ! -f "$stamp" ]]; then
    pixvqa_refuse_libvmaf "$prefix has no pixvqa-libvmaf-stamp.txt; refusing an unverified prefix"
    return 0
  fi
  local commit zimg
  commit="$(awk -F= '$1=="libvmaf_commit"{print $2}' "$stamp")"
  zimg="$(awk -F= '$1=="zimg_version"{print $2}' "$stamp")"
  if [[ "$commit" != "74f930d17de8c58e7c2ce55e2b897b0030ebd8ef" ]]; then
    pixvqa_refuse_libvmaf "stamp commit '${commit:-missing}' is not the HDR pin; refusing"
    return 0
  fi
  if [[ "$zimg" != "3.0.6" ]]; then
    pixvqa_refuse_libvmaf "stamp zimg version '${zimg:-missing}' is not 3.0.6; refusing"
    return 0
  fi
  PIXVQA_VMAF_CFLAGS+=" -DPIXVQA_LIBVMAF_COMMIT=${commit} -DPIXVQA_ZIMG_VERSION=${zimg}"
}
