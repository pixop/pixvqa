#!/usr/bin/env bash
# Build the ffmpeg program from the installer's n8.1.3 source.
# It is a decode oracle only. Encoding stays on the distribution ffmpeg.
# The ffmpeg program needs libavfilter, libswscale, and libswresample, which
# the installer leaves disabled, so this rebuild stays in the source tree.
# The rawvideo muxer stays enabled so the oracle can write comparison YUV.
# Nothing is installed over the libraries PixVQA already links.
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
rpath="$src/libavcodec:$src/libavformat:$src/libavutil:$src/libswscale:$src/libswresample:$src/libavfilter:$DAV1D_LIB"
(
    cd "$src"
    ./configure \
        --prefix="$FFMPEG_PREFIX" \
        --enable-shared --disable-static \
        --disable-gpl --disable-version3 \
        --disable-doc \
        --enable-ffmpeg --disable-ffplay --disable-ffprobe \
        --disable-avdevice \
        --disable-encoders --disable-muxers --enable-muxer=rawvideo \
        --disable-autodetect \
        --enable-libdav1d --enable-protocol=file \
        --extra-cflags="-I$DAV1D_PREFIX/include" \
        --extra-ldflags="-L$DAV1D_LIB -Wl,-rpath,$rpath"
    grep -q '^CONFIG_FFMPEG=yes$' ffbuild/config.mak || {
        printf 'FAIL configure did not enable the ffmpeg program\n' >&2
        exit 1
    }
    make -j"$(nproc)" ffmpeg
)
cp "$src/ffmpeg" "$out/ffmpeg"
chmod 755 "$out/ffmpeg"
if ldd "$out/ffmpeg" | grep -q 'not found'; then
    ldd "$out/ffmpeg" >&2
    printf 'FAIL decode oracle has unresolved libraries\n' >&2
    exit 1
fi
env -u LD_LIBRARY_PATH "$out/ffmpeg" -hide_banner -version
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf 'ffmpeg=%s\n' "$out/ffmpeg" >> "$GITHUB_OUTPUT"
fi
