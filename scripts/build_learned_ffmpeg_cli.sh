#!/usr/bin/env bash
# Out-of-tree build of pinned FFmpeg n8.1.3 ffmpeg+ffprobe for learned-metric decoding.
# Separate from the native libavcodec install (--disable-programs). Requires only the
# pinned source checkout; does not reconfigure the in-tree library build.
set -Eeuo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1091
source "$root/scripts/deps_layout.sh"
pixvqa_apply_deps_layout "$root"
# shellcheck disable=SC1091
source "$root/scripts/learned_venv_helpers.sh"

FFMPEG_TAG=n8.1.3
FFMPEG_COMMIT=1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7
if (( PIXVQA_DEPS_ISOLATED )); then
  default_cli_root="$PIXVQA_DEPS_ROOT/learned-ffmpeg-cli"
else
  default_cli_root="$root/deps/learned-ffmpeg-cli"
fi
cli_root="$(learned_ffmpeg_cli_root "$root" "$default_cli_root")"
# Dedicated checkout: the native libavcodec tree may carry in-source config.h.
src="$cli_root/ffmpeg-src"
build_dir="$cli_root/build"
install_prefix="$cli_root/prefix"

usage() {
  cat <<EOF
Usage: $0 [--prefix DIR]

Build ffmpeg and ffprobe from pinned FFmpeg $FFMPEG_TAG ($FFMPEG_COMMIT).
Binaries install under --prefix/bin (default: $install_prefix).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) install_prefix="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

build_dir="$(dirname -- "$install_prefix")/build"
workspace_fstype="$(findmnt -n -o FSTYPE -T "$root" 2>/dev/null || true)"
case "$workspace_fstype" in 9p|drvfs|fuse*)
  build_dir="${TMPDIR:-/tmp}/pixvqa-learned-ffmpeg-cli-build"
  printf 'learned FFmpeg CLI: using native build dir %s (workspace on %s)\n' \
    "$build_dir" "${workspace_fstype:-unknown}" >&2
  ;;
esac
mkdir -p "$build_dir" "$install_prefix/bin" "$install_prefix/lib" "$build_dir/ffbuild"

git -C "$src" rev-parse HEAD >/dev/null 2>&1 || {
  printf 'FAIL learned FFmpeg CLI: source checkout missing at %s\n' "$src" >&2
  printf 'Clone pinned FFmpeg %s into %s first (see ensure_learned_ffmpeg_cli.sh).\n' \
    "$FFMPEG_TAG" "$src" >&2
  exit 1
}
got="$(git -C "$src" rev-parse HEAD 2>/dev/null || true)"
[[ "$got" == "$FFMPEG_COMMIT" ]] || {
  printf 'FAIL learned FFmpeg CLI: %s HEAD is %s, expected %s\n' "$src" "${got:-unknown}" "$FFMPEG_COMMIT" >&2
  exit 1
}

extra_configure=()
extra_ldflags=()
rpath_parts=()
if [[ -n "${DAV1D_PREFIX:-}" ]]; then
  DAV1D_PC="$(find "$DAV1D_PREFIX" -type f -name dav1d.pc -print -quit 2>/dev/null || true)"
  if [[ -n "$DAV1D_PC" ]]; then
    export PKG_CONFIG_PATH="$(dirname "$DAV1D_PC")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    DAV1D_LIB="$(pkg-config --variable=libdir dav1d 2>/dev/null || true)"
    if [[ -n "$DAV1D_LIB" && -f "$DAV1D_LIB/libdav1d.so" ]]; then
      extra_configure+=(--enable-libdav1d --extra-cflags="-I$DAV1D_PREFIX/include")
      extra_ldflags+=("-L$DAV1D_LIB")
      rpath_parts+=("$DAV1D_LIB")
      printf 'learned FFmpeg CLI: linking libdav1d from %s\n' "$DAV1D_LIB"
    fi
  fi
fi

(
  cd "$build_dir"
  rpath="$install_prefix/lib"
  for part in "${rpath_parts[@]}"; do
    rpath="$rpath:$part"
  done
  "$src/configure" \
    --prefix="$install_prefix" \
    --enable-shared --disable-static \
    --disable-gpl --disable-version3 \
    --disable-doc \
    --enable-ffmpeg --enable-ffprobe --disable-ffplay \
    --disable-avdevice \
    --enable-avfilter --enable-swscale --enable-swresample \
    --disable-encoders --enable-encoder=mpeg4 --enable-encoder=rawvideo \
    --disable-muxers --enable-muxer=mp4 --enable-muxer=rawvideo \
    --disable-autodetect \
    --enable-protocol=file \
    --enable-demuxer=mov,mp4,mpegts,matroska,avi \
    --enable-decoder=h264,hevc,mpeg4 \
    --enable-parser=h264,hevc,mpeg4video \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
    --enable-filter=scale,format,testsrc2 \
    "${extra_configure[@]}" \
    --extra-ldflags="${extra_ldflags[*]} -Wl,-rpath,$rpath"
  for key in FFMPEG FFPROBE SWSCALE HEVC_DECODER H264_DECODER SCALE_FILTER FORMAT_FILTER; do
    grep -q "^CONFIG_${key}=yes$" ffbuild/config.mak || {
      printf 'FAIL configure did not enable CONFIG_%s\n' "$key" >&2
      exit 1
    }
  done
  build_jobs="$(nproc)"
  build_fstype="$(findmnt -n -o FSTYPE -T "$build_dir" 2>/dev/null || true)"
  case "$build_fstype" in 9p|drvfs|fuse*)
    build_jobs=1
    printf 'learned FFmpeg CLI: serial make on %s mount to avoid parallel link races\n' \
      "${build_fstype:-unknown}" >&2
    ;;
  esac
  make -j"$build_jobs" ffmpeg ffprobe
  # Installs only what this prefix build enabled (programs, DSOs, headers); nothing over
  # the native PixVQA libavcodec prefix under FFMPEG_PREFIX. STRIP=: avoids WSL drvfs/9p
  # strip failures on cross-mounted workspaces.
  make install STRIP=:
)

for bin in ffmpeg ffprobe; do
  target="$install_prefix/bin/$bin"
  [[ -x "$target" ]] || {
    printf 'FAIL %s missing after install\n' "$target" >&2
    exit 1
  }
done
shopt -s nullglob
libs=("$install_prefix/lib/"libav{codec,format,util,filter,swscale,swresample}.so*)
shopt -u nullglob
((${#libs[@]} >= 4)) || {
  printf 'FAIL learned FFmpeg CLI prefix lacks expected shared libraries under %s/lib\n' \
    "$install_prefix" >&2
  exit 1
}

# Record the pin for ensure_learned_ffmpeg_cli.sh idempotent reuse.
printf '%s\n' "$FFMPEG_COMMIT" > "$cli_root/FFMPEG_COMMIT"
printf '%s\n' "$install_prefix" > "$cli_root/INSTALL_PREFIX"

cli_libdir="$install_prefix/lib"
if env LD_LIBRARY_PATH="$cli_libdir" ldd "$install_prefix/bin/ffmpeg" 2>/dev/null | grep -q 'not found'; then
  env LD_LIBRARY_PATH="$cli_libdir" ldd "$install_prefix/bin/ffmpeg" >&2 || true
  printf 'FAIL learned FFmpeg CLI has unresolved libraries\n' >&2
  exit 1
fi

ff_version_cmd=(env "LD_LIBRARY_PATH=$cli_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
"${ff_version_cmd[@]}" "$install_prefix/bin/ffmpeg" -hide_banner -version | grep -qF "$FFMPEG_TAG" || {
  printf 'FAIL ffmpeg -version does not report %s\n' "$FFMPEG_TAG" >&2
  exit 1
}
"${ff_version_cmd[@]}" "$install_prefix/bin/ffprobe" -hide_banner -version | grep -qF "$FFMPEG_TAG" || {
  printf 'FAIL ffprobe -version does not report %s\n' "$FFMPEG_TAG" >&2
  exit 1
}

decoders="$(
  "${ff_version_cmd[@]}" "$install_prefix/bin/ffmpeg" -hide_banner -decoders 2>/dev/null || true
)"
grep -Eq '[[:space:]]hevc[[:space:]]' <<< "$decoders" || {
  printf 'FAIL ffmpeg -decoders lacks native HEVC decoder\n' >&2
  exit 1
}
pix_fmts="$(
  "${ff_version_cmd[@]}" "$install_prefix/bin/ffmpeg" -hide_banner -pix_fmts 2>/dev/null || true
)"
grep -qw bgr24 <<< "$pix_fmts" || {
  printf 'FAIL ffmpeg -pix_fmts lacks bgr24 output\n' >&2
  exit 1
}

# HEVC Main 10 -> BGR24 decode smoke when a host encoder is available for fixture generation.
fixture_ffmpeg="$(command -v ffmpeg 2>/dev/null || true)"
if [[ -n "$fixture_ffmpeg" ]]; then
  smoke_dir="$(mktemp -d)"
  trap 'rm -rf "$smoke_dir"' EXIT
  hevc_mp4="$smoke_dir/hevc_main10.mp4"
  bgr_raw="$smoke_dir/out.bgr24"
  if "$fixture_ffmpeg" -hide_banner -loglevel error -y \
      -f lavfi -i "color=c=black:s=64x64:d=1" \
      -c:v libx265 -pix_fmt yuv420p10le -x265-params lossless=1 \
      -frames:v 1 "$hevc_mp4" 2>/dev/null; then
    if ! "${ff_version_cmd[@]}" "$install_prefix/bin/ffmpeg" -hide_banner -loglevel error \
        -i "$hevc_mp4" -map 0:v:0 -an -pix_fmt bgr24 -f rawvideo "$bgr_raw"; then
      printf 'FAIL HEVC Main10 -> BGR24 decode smoke failed\n' >&2
      exit 1
    fi
    [[ "$(stat -c%s "$bgr_raw" 2>/dev/null || wc -c < "$bgr_raw")" -eq $((64 * 64 * 3)) ]] || {
      printf 'FAIL HEVC Main10 -> BGR24 output size mismatch\n' >&2
      exit 1
    }
    printf 'learned FFmpeg CLI: HEVC Main10 -> BGR24 decode smoke passed\n'
  else
    printf 'learned FFmpeg CLI: skipped HEVC Main10 decode smoke (host libx265 encoder unavailable)\n'
  fi
  trap - EXIT
  rm -rf "$smoke_dir"
fi

printf 'learned FFmpeg CLI installed: %s/bin/ffmpeg %s/bin/ffprobe\n' \
  "$install_prefix" "$install_prefix"
