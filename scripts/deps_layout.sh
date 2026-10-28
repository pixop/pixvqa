#!/usr/bin/env bash
# Resolve PixVQA-managed third-party directories.
# PIXVQA_DEPS_ROOT unset: the historical home paths. libvmaf and zimg stay
# in the checkout (deps/libvmaf-74f930d and deps/zimg-3.0.6).
# PIXVQA_DEPS_ROOT set: source checkouts, toolchains, build directories, and
# install prefixes for those dependencies all live under that directory.
# This does not choose CUDA_HOME, the NVIDIA driver, or apt packages.

pixvqa_canon_dir() {
  local path="$1" parent base
  if [[ -d "$path" ]]; then
    (cd -- "$path" && pwd -P)
    return 0
  fi
  parent="$(dirname -- "$path")"
  base="$(basename -- "$path")"
  if [[ -d "$parent" ]]; then
    printf '%s/%s\n' "$(cd -- "$parent" && pwd -P)" "$base"
    return 0
  fi
  printf '%s\n' "$path"
}

pixvqa_apply_deps_layout() {
  local src_root="$1"
  local home_root legacy_dav1d legacy_ffmpeg root requested
  [[ -n "$src_root" ]] || return 1
  if [[ -n "${PIXVQA_DEPS_ROOT+x}" && -z "$PIXVQA_DEPS_ROOT" ]]; then
    echo "deps_layout: PIXVQA_DEPS_ROOT is set but empty" >&2
    return 1
  fi
  home_root="$(cd -- "$HOME" && pwd -P)"
  legacy_dav1d="$home_root/dav1d-1.5.1"
  legacy_ffmpeg="$home_root/ffmpeg-n8.1.3"
  if [[ -z "${PIXVQA_DEPS_ROOT:-}" ]]; then
    PIXVQA_DEPS_ISOLATED=0
    DAV1D_PREFIX="$legacy_dav1d"
    FFMPEG_PREFIX="$legacy_ffmpeg"
    DEPS_SRC="$home_root/pixvqa-deps/src"
    GCC13="$home_root/pixvqa-deps/gcc13"
    MAMBA_ROOT_PREFIX="$home_root/pixvqa-deps/micromamba-root"
    PIXVQA_TOOLS_DIR="$home_root/pixvqa-deps/tools"
    NVCODEC_PREFIX="$home_root/pixvqa-deps/nv-codec-headers"
    LIBVMAF_PREFIX="$src_root/deps/libvmaf-74f930d"
    unset LIBVMAF_SRC LIBVMAF_BUILD ZIMG_PREFIX ZIMG_SRC
    unset ZIMG_AUTOTOOLS_PREFIX ZIMG_AUTOTOOLS_DEBS
    return 0
  fi
  requested="$(pixvqa_canon_dir "$PIXVQA_DEPS_ROOT")"
  if [[ "$requested" == "$home_root" || "$requested" == "$legacy_dav1d" || "$requested" == "$legacy_ffmpeg" ]]; then
    echo "deps_layout: PIXVQA_DEPS_ROOT must not be the home directory or an existing dav1d/FFmpeg prefix" >&2
    return 1
  fi
  mkdir -p -- "$requested"
  root="$(cd -- "$requested" && pwd -P)"
  PIXVQA_DEPS_ROOT="$root"
  PIXVQA_DEPS_ISOLATED=1
  DAV1D_PREFIX="$root/dav1d-1.5.1"
  FFMPEG_PREFIX="$root/ffmpeg-n8.1.3"
  DEPS_SRC="$root/src"
  GCC13="$root/gcc13"
  MAMBA_ROOT_PREFIX="$root/micromamba-root"
  PIXVQA_TOOLS_DIR="$root/tools"
  NVCODEC_PREFIX="$root/nv-codec-headers"
  LIBVMAF_PREFIX="$root/libvmaf-74f930d"
  LIBVMAF_SRC="$root/src/vmaf-74f930d"
  LIBVMAF_BUILD="$root/build/libvmaf-74f930d"
  ZIMG_PREFIX="$root/zimg-3.0.6"
  ZIMG_SRC="$root/src/zimg"
  ZIMG_AUTOTOOLS_PREFIX="$root/src/host-autotools"
  ZIMG_AUTOTOOLS_DEBS="$root/src/host-debs"
  if [[ "$DAV1D_PREFIX" == "$legacy_dav1d" || "$FFMPEG_PREFIX" == "$legacy_ffmpeg" ]]; then
    echo "deps_layout: PIXVQA_DEPS_ROOT would reuse $legacy_dav1d or $legacy_ffmpeg" >&2
    return 1
  fi
}
