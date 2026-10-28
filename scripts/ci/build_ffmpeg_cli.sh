#!/usr/bin/env bash
# Build the ffmpeg program from the installer's already configured n8.1.3 tree.
# It is a decode oracle only. Encoding stays on the distribution ffmpeg.
# The installed shared libraries are not replaced.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
# shellcheck disable=SC1091
source "$root/scripts/deps_layout.sh"
pixvqa_apply_deps_layout "$root"
src="$DEPS_SRC/ffmpeg"
[[ -d "$src" ]] || { printf 'FAIL FFmpeg source is missing: %s\n' "$src" >&2; exit 1; }
DAV1D_PC="$(find "$DAV1D_PREFIX" -type f -name dav1d.pc -print -quit)"
[[ -n "$DAV1D_PC" ]] || { printf 'FAIL dav1d.pc missing under %s\n' "$DAV1D_PREFIX" >&2; exit 1; }
export PKG_CONFIG_PATH="$(dirname "$DAV1D_PC")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
DAV1D_LIB="$(pkg-config --variable=libdir dav1d)"
[[ -n "$DAV1D_LIB" && -f "$DAV1D_LIB/libdav1d.so" ]] || {
    printf 'FAIL libdav1d.so missing (libdir=%s)\n' "${DAV1D_LIB:-unset}" >&2
    exit 1
}
printf 'decode oracle uses dav1d %s\n' "$DAV1D_LIB"
out="$root/deps/ci-ffmpeg"
mkdir -p "$out"
(
    cd "$src"
    ./configure \
        --prefix="$FFMPEG_PREFIX" \
        --enable-shared --disable-static \
        --disable-gpl --disable-version3 \
        --disable-doc \
        --disable-ffplay --disable-ffprobe \
        --disable-avdevice --disable-avfilter \
        --disable-swscale --disable-swresample \
        --disable-encoders --disable-muxers \
        --disable-autodetect \
        --enable-libdav1d --enable-protocol=file \
        --extra-cflags="-I$DAV1D_PREFIX/include" \
        --extra-ldflags="-L$DAV1D_LIB -Wl,-rpath,$DAV1D_LIB -Wl,-rpath,$FFMPEG_PREFIX/lib"
    make -j"$(nproc)" ffmpeg
)
cp "$src/ffmpeg" "$out/ffmpeg"
chmod 755 "$out/ffmpeg"
if ldd "$out/ffmpeg" | grep -q 'not found'; then
    ldd "$out/ffmpeg" >&2
    printf 'FAIL decode oracle has unresolved libraries\n' >&2
    exit 1
fi
"$out/ffmpeg" -hide_banner -version
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf 'ffmpeg=%s\n' "$out/ffmpeg" >> "$GITHUB_OUTPUT"
fi
