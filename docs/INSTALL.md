# Installing PixVQA on Ubuntu (manual procedure)

This guide builds PixVQA **manually**, using the same pinned dependencies as the guided `install_pixvqa.sh` installer. It targets Linux x86-64 (Ubuntu 22.04/24.04, including WSL2). Native Windows and Git Bash are not supported. The GPU option requires an existing NVIDIA driver and CUDA **12.8** toolkit; this guide does not install either.

**Run only the commands inside Bash code blocks.** Run the blocks in order. Each installation block runs inside `( ... )`, a *subshell*: a failed command stops that block **without closing your interactive Ubuntu terminal**. Look for `PASS:` at the end of each block. **Do not continue after a missing PASS.** Do not run the guided installer during this manual test.

## 0. Choose the checkout and build type

Open an Ubuntu/Linux terminal and **change to the root of your cloned PixVQA repository**, wherever you cloned it. Run this block from that directory; no checkout name or absolute path is assumed.

```bash
# Discard PixVQA-specific settings from earlier installation attempts.
unset PIXVQA_SRC PIXVQA_BIN LIBVMAF_PREFIX FFMPEG_PREFIX
unset DAV1D_PREFIX DAV1D_LIB GCC13 GCC_ENV CUDART_LIBDIR
unset PIXVQA_LIBCUDA_DIR PIXVQA_FULL_LINK_CXX
unset ZIMG_PREFIX ZIMG_SRC

if [ -f build.sh ] && [ -f build_decode.sh ] && [ -f build_cuda.sh ] && \
   [ -f scripts/build_libvmaf.sh ] && [ -f scripts/install.sh ]; then
    export PIXVQA_SRC="$(pwd -P)"
    export PREFIX="$PIXVQA_SRC"
    export PIXVQA_ENABLE_NVIDIA=1  # Set to 0 for CPU/software decoding only.
    # Checkout-local manual root. It is separate from the guided installer's
    # $HOME paths and from PIXVQA_DEPS_ROOT. apt packages, the NVIDIA driver,
    # and CUDA_HOME stay outside this directory.
    export PIXVQA_MANUAL_ROOT="${PIXVQA_MANUAL_ROOT:-$PIXVQA_SRC/deps/manual}"
    export LIBVMAF_PREFIX="${LIBVMAF_PREFIX:-$PIXVQA_SRC/deps/libvmaf-74f930d}"
    export ZIMG_PREFIX="${ZIMG_PREFIX:-$PIXVQA_SRC/deps/zimg-3.0.6}"
    export ZIMG_SRC="${ZIMG_SRC:-$PIXVQA_SRC/deps/src/zimg}"
    printf 'PixVQA checkout: %s\nInstall prefix: %s\nManual dependency root: %s\nNVIDIA enabled: %s\n' \
        "$PIXVQA_SRC" "$PREFIX" "$PIXVQA_MANUAL_ROOT" "$PIXVQA_ENABLE_NVIDIA"
else
    echo 'STOP: Navigate to the root of your PixVQA clone before continuing.' >&2
fi
```

**Directory layout:** The default manual root is `$PIXVQA_SRC/deps/manual` (`PIXVQA_MANUAL_ROOT`). It contains dav1d, FFmpeg, CUDA codec headers, and the GCC 13.4 environment. Pinned libvmaf stays at `$LIBVMAF_PREFIX` (default `$PIXVQA_SRC/deps/libvmaf-74f930d`) and zimg at `$ZIMG_PREFIX` (default `$PIXVQA_SRC/deps/zimg-3.0.6`). To keep those two prefixes inside the manual root as well, change those three assignments in Section 0 to `$PIXVQA_MANUAL_ROOT/libvmaf-74f930d`, `$PIXVQA_MANUAL_ROOT/zimg-3.0.6`, and `$PIXVQA_MANUAL_ROOT/src/zimg`. Later blocks keep the values Section 0 exported. `build/` contains intermediate binaries and `bin/pixvqa` is the installed executable. None of these are fixed home-directory paths. `scripts/build_libvmaf_3.2.1.sh` can still build the previous tag into `deps/libvmaf-3.2.1`, and that prefix is not linked.

`PIXVQA_MANUAL_ROOT` does not isolate Ubuntu `apt` packages. The NVIDIA driver and CUDA toolkit remain wherever they are already installed; set `CUDA_HOME` to that toolkit. Use a different checkout, or a different `PIXVQA_MANUAL_ROOT`, from a guided install that sets `PIXVQA_DEPS_ROOT`.

`scripts/build_libvmaf.sh`, `scripts/build_libvmaf_3.2.1.sh`, and `scripts/build_zimg.sh` prepend `$GCC_ENV/bin` only when `GCC_ENV` is set. They do not search a user-specific compiler directory. `gcc`, `g++`, GNU `make`, `perl`, `autoconf`, `automake`, `libtoolize`, and `m4` must be on `PATH` for the zimg build. Ubuntu's `build-essential` package supplies `make` and `g++`. Install `perl`, `autoconf`, `automake`, `libtool`, and `m4` from the same Ubuntu release. `scripts/build_zimg.sh` does not extract Ubuntu 24.04 binary packages on Ubuntu 22.04. On Ubuntu 24.04 only, if those tools are absent, it may download the SHA-256-pinned Ubuntu 24.04 packages named in the script. The libvmaf and zimg commits stay pinned. `scripts/libvmaf_flags.sh` links a prefix only when `pixvqa-libvmaf-stamp.txt` records commit `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` and zimg 3.0.6. The final NVDEC link uses `/usr/bin/g++`.

If you reopen the terminal, navigate back to **your** checkout and repeat Section 0. Subsequent blocks rediscover dependency paths. The manual process does not delete or reuse the automatic install's private dependency directories.

> **Existing checkout warning:** For a clean test, use a clone without generated `deps/`, `build/` or `bin/` directories. The instructions never remove those directories.

## 1. Ubuntu prerequisites

```bash
(
    set -Eeuo pipefail
    sudo apt-get update
    sudo apt-get install -y build-essential python3 python3-numpy git meson ninja-build nasm pkg-config curl ca-certificates bzip2 tar perl autoconf automake libtool m4
    for cmd in git g++ python3 meson ninja nasm pkg-config make perl autoconf automake libtoolize m4; do
        command -v "$cmd" >/dev/null || { echo "STOP: Missing $cmd" >&2; exit 1; }
    done
    test -x /usr/bin/g++ || { echo 'STOP: /usr/bin/g++ is required for the final NVDEC link' >&2; exit 1; }
    echo 'PASS: Ubuntu build prerequisites'
)
```

## 2. Build dav1d 1.5.1

```bash
(
    set -Eeuo pipefail
    : "${PIXVQA_SRC:?Run Section 0 first}"
    root="${PIXVQA_MANUAL_ROOT:-$PIXVQA_SRC/deps/manual}"
    mkdir -p "$root/src"
    if [ ! -e "$root/src/dav1d" ]; then
        git clone --depth 1 --branch 1.5.1 https://code.videolan.org/videolan/dav1d.git "$root/src/dav1d"
    fi
    test -d "$root/src/dav1d/.git" || { echo 'STOP: dav1d source checkout invalid' >&2; exit 1; }
    test "$(git -C "$root/src/dav1d" rev-parse HEAD)" = 42b2b24fb8819f1ed3643aa9cf2a62f03868e3aa || { echo 'STOP: dav1d is not at pinned commit' >&2; exit 1; }
    if [ ! -f "$root/src/dav1d/build/build.ninja" ]; then
        meson setup "$root/src/dav1d/build" "$root/src/dav1d" --prefix="$root/dav1d-1.5.1" --default-library=shared --buildtype=release
    fi
    ninja -C "$root/src/dav1d/build"
    ninja -C "$root/src/dav1d/build" install
    pc="$(find "$root/dav1d-1.5.1" -type f -name dav1d.pc -print -quit)"
    test -n "$pc" || { echo 'STOP: dav1d.pc missing after install' >&2; exit 1; }
    export PKG_CONFIG_PATH="$(dirname "$pc")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    test "$(pkg-config --modversion dav1d)" = 1.5.1 || { echo 'STOP: incorrect dav1d version' >&2; exit 1; }
    libdir="$(pkg-config --variable=libdir dav1d)"
    test -f "$libdir/libdav1d.so" || { echo 'STOP: libdav1d.so missing' >&2; exit 1; }
    printf 'PASS: dav1d 1.5.1 installed (%s)\n' "$libdir"
)
```

## 3. Build FFmpeg n8.1.3 (choose NVIDIA now)

The manual decoder needs FFmpeg **libraries**, not its CLI. This build enables libdav1d, leaves the native VVC decoder available, disables encoders and GPL/version3 options, and optionally enables NVDEC/CUVID. Choosing NVIDIA later would require rebuilding FFmpeg.

**3A. NVIDIA codec headers (only needed for NVIDIA selection).** This branch uses a local `nv-codec-headers` installation and does not change system directories. Run it even if `ffnvcodec` is already globally installed, so the test doesn't silently depend on that installation.

```bash
(
    set -Eeuo pipefail
    : "${PIXVQA_SRC:?Run Section 0 first}"
    : "${PIXVQA_ENABLE_NVIDIA:?Run Section 0 first}"
    if [ "$PIXVQA_ENABLE_NVIDIA" = 1 ]; then
        root="${PIXVQA_MANUAL_ROOT:-$PIXVQA_SRC/deps/manual}"
        mkdir -p "$root/src"
        if [ ! -e "$root/src/nv-codec-headers" ]; then
            git clone --depth 1 --branch n13.0.19.0 https://github.com/FFmpeg/nv-codec-headers.git "$root/src/nv-codec-headers"
        fi
        test -d "$root/src/nv-codec-headers/.git" || { echo 'STOP: codec header source invalid' >&2; exit 1; }
        tagcommit="$(git -C "$root/src/nv-codec-headers" rev-parse 'refs/tags/n13.0.19.0^{commit}')"
        test "$(git -C "$root/src/nv-codec-headers" rev-parse HEAD)" = "$tagcommit" || { echo 'STOP: incorrect codec headers revision' >&2; exit 1; }
        make -C "$root/src/nv-codec-headers" PREFIX="$root/nv-codec-headers" install
        export PKG_CONFIG_PATH="$root/nv-codec-headers/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
        pkg-config --modversion ffnvcodec
        echo 'PASS: private NVIDIA codec headers'
    else
        echo 'PASS: NVIDIA codec headers not needed (CPU selection)'
    fi
)
```

**3B. Clone, configure, build, install and verify FFmpeg.** Do not run `make` if configure fails. This block checks the pinned commit and every required output. It also avoids Bash line-continuation errors by building the configuration arguments in an array.

```bash
(
    set -Eeuo pipefail
    : "${PIXVQA_SRC:?Run Section 0 first}"
    : "${PIXVQA_ENABLE_NVIDIA:?Run Section 0 first}"
    root="${PIXVQA_MANUAL_ROOT:-$PIXVQA_SRC/deps/manual}"
    src="$root/src/ffmpeg"
    prefix="$root/ffmpeg-n8.1.3"
    dav1d_prefix="$root/dav1d-1.5.1"
    dav1d_pc="$(find "$dav1d_prefix" -type f -name dav1d.pc -print -quit 2>/dev/null)"
    test -n "$dav1d_pc" || { echo 'STOP: dav1d.pc missing; complete Section 2' >&2; exit 1; }
    export PKG_CONFIG_PATH="$(dirname "$dav1d_pc")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    test "$(pkg-config --modversion dav1d)" = 1.5.1 || { echo 'STOP: dav1d package version mismatch' >&2; exit 1; }
    dav1d_lib="$(pkg-config --variable=libdir dav1d)"
    test -f "$dav1d_lib/libdav1d.so" || { echo 'STOP: installed dav1d library missing' >&2; exit 1; }
    mkdir -p "$root/src"
    if [ ! -e "$src" ]; then
        # git.ffmpeg.org may occasionally fail with HTTP/2; the official GitHub mirror is a fallback.
        if ! git -c http.version=HTTP/1.1 clone --depth 1 --branch n8.1.3 https://git.ffmpeg.org/ffmpeg.git "$src"; then
            if [ -d "$src" ]; then
                echo "STOP: Inspect incomplete FFmpeg checkout at $src before retrying." >&2
                exit 1
            fi
            git -c http.version=HTTP/1.1 clone --depth 1 --branch n8.1.3 https://github.com/FFmpeg/FFmpeg.git "$src"
        fi
    fi
    test -d "$src/.git" || { echo 'STOP: FFmpeg source checkout missing or invalid' >&2; exit 1; }
    test "$(git -C "$src" rev-parse HEAD)" = 1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7 || { echo 'STOP: incorrect FFmpeg commit' >&2; exit 1; }
    hw=()
    if [ "$PIXVQA_ENABLE_NVIDIA" = 1 ]; then
        export PKG_CONFIG_PATH="$root/nv-codec-headers/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
        pkg-config --exists ffnvcodec || { echo 'STOP: private ffnvcodec headers missing; run Section 3A' >&2; exit 1; }
        hw=(--enable-ffnvcodec --enable-nvdec --enable-cuvid)
    fi
    cd "$src"
    opts=(
        "--prefix=$prefix" --enable-shared --disable-static
        --disable-gpl --disable-version3 --disable-doc --disable-programs
        --disable-avdevice --disable-avfilter --disable-swscale --disable-swresample
        --disable-encoders --disable-muxers --disable-autodetect
        --enable-libdav1d --enable-protocol=file
        "${hw[@]}"
        "--extra-cflags=-I$dav1d_prefix/include"
        "--extra-ldflags=-L$dav1d_lib -Wl,-rpath,$dav1d_lib"
    )
    ./configure "${opts[@]}" || { echo 'STOP: FFmpeg configure failed; see config.log' >&2; exit 1; }
    test -f config.h || { echo 'STOP: FFmpeg config.h missing after configure' >&2; exit 1; }
    grep -Eq '^#define CONFIG_LIBDAV1D 1$' config.h || { echo 'STOP: FFmpeg libdav1d not enabled' >&2; exit 1; }
    if [ "$PIXVQA_ENABLE_NVIDIA" = 1 ]; then
        for feature in FFNVCODEC NVDEC CUVID; do
            grep -Eq "^#define CONFIG_${feature} 1$" config.h || { echo "STOP: FFmpeg $feature not enabled" >&2; exit 1; }
        done
    fi
    make -j"$(nproc)" || { echo 'STOP: FFmpeg compilation failed' >&2; exit 1; }
    make install || { echo 'STOP: FFmpeg make install failed' >&2; exit 1; }
    test -f "$prefix/lib/libavcodec.so" || { echo 'STOP: installed libavcodec.so missing' >&2; exit 1; }
    test -f "$prefix/include/libavcodec/avcodec.h" || { echo 'STOP: installed avcodec.h missing' >&2; exit 1; }
    LD_LIBRARY_PATH="$prefix/lib:$dav1d_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$prefix/lib/libavcodec.so" | grep libdav1d || { echo 'STOP: FFmpeg cannot resolve libdav1d' >&2; exit 1; }
    printf 'PASS: FFmpeg n8.1.3 installed (%s), NVIDIA=%s\n' "$prefix" "$PIXVQA_ENABLE_NVIDIA"
)
```

**If you do not see `PASS: FFmpeg ...`, stop here.** In particular, a missing `config.h` means FFmpeg never configured successfully; do not continue to libvmaf or PixVQA builds. The FFmpeg library build deliberately does not provide an `ffmpeg` CLI; full QA requires a separate executable.

## 4. Build libvmaf commit 74f930d and zimg 3.0.6

GNU `make` must be on `PATH`. `scripts/build_libvmaf.sh` builds zimg first, then checks out Netflix `vmaf` commit `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef`. This is not a tagged release. The previous tag is v3.2.1, commit `f85a853692a8c730d0270cd733c8bb30b5b93b7c`, still built only by `scripts/build_libvmaf_3.2.1.sh`. Meson reports project version 3.2.0. `vmaf --version` prints `74f930d`.

zimg is tag `release-3.0.6`, commit `f819b14e8f39d1282400b0d9543e8ef73c1b2bbd`, static, prefix `deps/zimg-3.0.6`. libvmaf is static, prefix `deps/libvmaf-74f930d`, with `-Denable_zimg=true` and CUDA off. The script stops unless `pkg-config` resolves the file installed under that prefix, `--libs --static` contains `-lzimg`, and `picture.h` declares `vmaf_picture_convert`.

```bash
(
    set -Eeuo pipefail
    : "${PIXVQA_SRC:?Run Section 0 first}"
    cd "$PIXVQA_SRC"
    test -f scripts/build_libvmaf.sh || { echo 'STOP: wrong PixVQA checkout' >&2; exit 1; }
    export GCC_ENV=/usr
    export LIBVMAF_PREFIX="${LIBVMAF_PREFIX:-$PIXVQA_SRC/deps/libvmaf-74f930d}"
    export ZIMG_PREFIX="${ZIMG_PREFIX:-$PIXVQA_SRC/deps/zimg-3.0.6}"
    export ZIMG_SRC="${ZIMG_SRC:-$PIXVQA_SRC/deps/src/zimg}"
    bash scripts/build_libvmaf.sh
    archive="$(find "$LIBVMAF_PREFIX" -type f -name libvmaf.a -print -quit 2>/dev/null)"
    test -n "$archive" && test -f "$archive" || { echo 'STOP: pinned libvmaf.a missing' >&2; exit 1; }
    grep -q vmaf_picture_convert "$LIBVMAF_PREFIX/include/libvmaf/picture.h" || { echo 'STOP: picture.h has no vmaf_picture_convert' >&2; exit 1; }
    source scripts/libvmaf_flags.sh
    pixvqa_detect_libvmaf "$PIXVQA_SRC"
    test "${PIXVQA_HAS_VMAF:-0}" = 1 || { echo 'STOP: PixVQA did not detect pinned libvmaf' >&2; exit 1; }
    case "$PIXVQA_VMAF_LIBS" in *-lzimg*) ;; *) echo 'STOP: libvmaf is not linked with zimg' >&2; exit 1 ;; esac
    case "$PIXVQA_VMAF_CFLAGS" in *74f930d17de8c58e7c2ce55e2b897b0030ebd8ef*) ;; *) echo 'STOP: libvmaf stamp is not 74f930d' >&2; exit 1 ;; esac
    printf 'PASS: pinned libvmaf enabled (%s)\n' "$archive"
)
```

The eight SDR model files and the eight HDR v1.1.8 model files in [model/](../model/README.md) are what `--model` loads. Without the 74f930d prefix the PixVQA binary still builds, and selecting `vmaf` is an error.

## 5. Build CPU and software-decoding PixVQA

```bash
(
    set -Eeuo pipefail
    : "${PIXVQA_SRC:?Run Section 0 first}"
    cd "$PIXVQA_SRC"
    root="${PIXVQA_MANUAL_ROOT:-$PIXVQA_SRC/deps/manual}"
    export GCC_ENV=/usr
    export LIBVMAF_PREFIX="${LIBVMAF_PREFIX:-$PIXVQA_SRC/deps/libvmaf-74f930d}"
    export ZIMG_PREFIX="${ZIMG_PREFIX:-$PIXVQA_SRC/deps/zimg-3.0.6}"
    export ZIMG_SRC="${ZIMG_SRC:-$PIXVQA_SRC/deps/src/zimg}"
    bash build.sh
    test -x build/cpu/pixvqa || { echo 'STOP: CPU executable missing' >&2; exit 1; }
    dav1d_pc="$(find "$root/dav1d-1.5.1" -type f -name dav1d.pc -print -quit 2>/dev/null)"
    test -n "$dav1d_pc" || { echo 'STOP: dav1d not installed' >&2; exit 1; }
    export PKG_CONFIG_PATH="$(dirname "$dav1d_pc")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export DAV1D_LIB="$(pkg-config --variable=libdir dav1d)"
    export FFMPEG_PREFIX="$root/ffmpeg-n8.1.3"
    test -f "$FFMPEG_PREFIX/lib/libavcodec.so" || { echo 'STOP: FFmpeg library missing; complete Section 3B' >&2; exit 1; }
    unset GCC13 CUDA_HOME PIXVQA_LIBCUDA_DIR
    export LD_LIBRARY_PATH="$FFMPEG_PREFIX/lib:$DAV1D_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    bash build_decode.sh
    test -x build/swdec/pixvqa || { echo 'STOP: software-decoder executable missing' >&2; exit 1; }
    echo 'PASS: PixVQA CPU and software-decoding binaries'
)
```

## 6. Optional CUDA/NVDEC build (only when `PIXVQA_ENABLE_NVIDIA=1`)

Requires NVIDIA driver/GPU, the **CUDA 12.8** toolkit, and conda-forge GCC **13.4** (used as the CUDA host compiler). The current `build_cuda.sh` targets `sm_89` and `sm_120`; other GPU architectures are not validated. On WSL2, do not install a separate Linux NVIDIA display driver over the Windows-provided WSL driver.

```bash
(
    set -Eeuo pipefail
    : "${PIXVQA_SRC:?Run Section 0 first}"
    : "${PIXVQA_ENABLE_NVIDIA:?Run Section 0 first}"
    if [ "$PIXVQA_ENABLE_NVIDIA" != 1 ]; then echo 'PASS: NVIDIA build skipped by selection'; exit 0; fi
    root="${PIXVQA_MANUAL_ROOT:-$PIXVQA_SRC/deps/manual}"
    export GCC13="$root/gcc13"
    export MAMBA_ROOT_PREFIX="$root/micromamba-root"
    compiler="$GCC13/bin/x86_64-conda-linux-gnu-g++"
    if [ ! -x "$compiler" ]; then
        if command -v conda >/dev/null 2>&1; then
            conda create -y -p "$GCC13" --override-channels -c conda-forge 'gxx_linux-64=13.4'
        elif command -v micromamba >/dev/null 2>&1; then
            micromamba create -y -p "$GCC13" --override-channels -c conda-forge 'gxx_linux-64=13.4'
        else
            mkdir -p "$root/tools"
            curl -fLsS 'https://micro.mamba.pm/api/micromamba/linux-64/latest' -o "$root/tools/micromamba.tar.bz2"
            tar -xjf "$root/tools/micromamba.tar.bz2" -C "$root/tools" bin/micromamba
            "$root/tools/bin/micromamba" create -y -p "$GCC13" --override-channels -c conda-forge 'gxx_linux-64=13.4'
        fi
    fi
    test -x "$compiler" || { echo 'STOP: GCC 13.4 executable missing' >&2; exit 1; }
    "$compiler" -dumpfullversion | grep -Eq '^13\.4(\.|$)' || { echo 'STOP: expected GCC 13.4' >&2; exit 1; }
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
    test -x "$CUDA_HOME/bin/nvcc" || { echo 'STOP: CUDA 12.8 nvcc missing (set CUDA_HOME)' >&2; exit 1; }
    "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.8' || { echo 'STOP: expected CUDA toolkit 12.8' >&2; exit 1; }
    command -v nvidia-smi >/dev/null && nvidia-smi || { echo 'STOP: NVIDIA GPU/driver unavailable' >&2; exit 1; }
    export PIXVQA_LIBCUDA_DIR="${PIXVQA_LIBCUDA_DIR:-}"
    if [ -z "$PIXVQA_LIBCUDA_DIR" ]; then
        if [ -f /usr/lib/wsl/lib/libcuda.so.1 ]; then PIXVQA_LIBCUDA_DIR=/usr/lib/wsl/lib
        elif [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then PIXVQA_LIBCUDA_DIR=/usr/lib/x86_64-linux-gnu
        fi
    fi
    test -f "$PIXVQA_LIBCUDA_DIR/libcuda.so.1" || { echo 'STOP: libcuda.so.1 missing; set PIXVQA_LIBCUDA_DIR' >&2; exit 1; }
    export PIXVQA_LIBCUDA_DIR
    cudart=''
    for candidate in "$CUDA_HOME/targets/x86_64-linux/lib" "$CUDA_HOME/lib64" "$CUDA_HOME/lib"; do
        if [ -f "$candidate/libcudart.so" ]; then cudart="$candidate"; break; fi
    done
    test -n "$cudart" || { echo 'STOP: CUDA libcudart.so missing' >&2; exit 1; }
    export CUDART_LIBDIR="$cudart"
    export LIBRARY_PATH="$cudart${LIBRARY_PATH:+:$LIBRARY_PATH}"
    dav1d_pc="$(find "$root/dav1d-1.5.1" -type f -name dav1d.pc -print -quit 2>/dev/null)"
    test -n "$dav1d_pc" || { echo 'STOP: dav1d missing' >&2; exit 1; }
    export PKG_CONFIG_PATH="$(dirname "$dav1d_pc")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export DAV1D_LIB="$(pkg-config --variable=libdir dav1d)"
    export FFMPEG_PREFIX="$root/ffmpeg-n8.1.3"
    for feature in FFNVCODEC NVDEC CUVID; do
        grep -Eq "^#define CONFIG_${feature} 1$" "$root/src/ffmpeg/config.h" || { echo "STOP: FFmpeg lacks $feature; redo Section 3B with NVIDIA enabled" >&2; exit 1; }
    done
    test -f "$FFMPEG_PREFIX/lib/libavcodec.so" || { echo 'STOP: FFmpeg libraries missing' >&2; exit 1; }
    export LD_LIBRARY_PATH="$cudart:$FFMPEG_PREFIX/lib:$DAV1D_LIB:$PIXVQA_LIBCUDA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export GCC_ENV=/usr
    export LIBVMAF_PREFIX="${LIBVMAF_PREFIX:-$PIXVQA_SRC/deps/libvmaf-74f930d}"
    export ZIMG_PREFIX="${ZIMG_PREFIX:-$PIXVQA_SRC/deps/zimg-3.0.6}"
    export ZIMG_SRC="${ZIMG_SRC:-$PIXVQA_SRC/deps/src/zimg}"
    export PIXVQA_FULL_LINK_CXX=/usr/bin/g++
    cd "$PIXVQA_SRC"
    bash build_cuda.sh
    bash build_decode.sh
    test -x build/full/pixvqa || { echo 'STOP: full PixVQA executable missing' >&2; exit 1; }
    build/full/pixvqa --version | grep -q 'decode: software nvdec' || { echo 'STOP: full binary does not report NVDEC' >&2; exit 1; }
    echo 'PASS: PixVQA CUDA and NVDEC executable'
)
```

### Build outputs: which executable is which?

Build scripts create executables under `build/`; they do **not** install multiple versions of PixVQA. The install step in Section 7 selects one build and copies it to `$PREFIX/bin/pixvqa`.

| Executable | Capabilities |
|---|---|
| `build/cpu/pixvqa` | CPU metrics, raw YUV input only |
| `build/cuda/pixvqa` | CPU and CUDA-supported metrics, raw YUV input only |
| `build/swdec/pixvqa` | CPU metrics with FFmpeg software decoding |
| `build/full/pixvqa` | CPU and CUDA-supported metrics with software decoding and NVDEC |
| `$PREFIX/bin/pixvqa` | Installed copy of the selected build |
| `$PREFIX/bin/pixvqa-with-libs` | Launcher generated **only by the guided installer**, for private runtime libraries |

The manual procedure below installs `build/full/pixvqa` when `PIXVQA_ENABLE_NVIDIA=1`, otherwise `build/swdec/pixvqa`. Not all build variants must exist. `PREFIX` is the installation destination, while `PIXVQA_BIN` identifies the compiled executable to copy.

## 7. Install into this checkout and check dynamic libraries

This matches the default **repository-root installation prefix** of the updated guided installer. The binary is installed at `$PIXVQA_SRC/bin/pixvqa`, without depending on the checkout directory name. The manual `scripts/install.sh` does **not** generate the guided installer's `pixvqa-with-libs` launcher, so verify the ELF runtime dependencies with an explicit library path.

```bash
(
    set -Eeuo pipefail
    : "${PIXVQA_SRC:?Run Section 0 first}"
    : "${PIXVQA_ENABLE_NVIDIA:?Run Section 0 first}"
    cd "$PIXVQA_SRC"
    root="${PIXVQA_MANUAL_ROOT:-$PIXVQA_SRC/deps/manual}"
    export PREFIX="$PIXVQA_SRC"
    if [ "$PIXVQA_ENABLE_NVIDIA" = 1 ]; then
        export PIXVQA_BIN=build/full/pixvqa
    else
        export PIXVQA_BIN=build/swdec/pixvqa
    fi
    test -x "$PIXVQA_BIN" || { echo "STOP: binary $PIXVQA_BIN missing; complete earlier sections" >&2; exit 1; }
    PREFIX="$PREFIX" PIXVQA_BIN="$PIXVQA_BIN" bash scripts/install.sh
    test -x "$PREFIX/bin/pixvqa" || { echo 'STOP: installed binary missing' >&2; exit 1; }
    cmp "$PREFIX/bin/pixvqa" "$PIXVQA_BIN" || { echo 'STOP: installed executable differs from selected build' >&2; exit 1; }
    dav1d_pc="$(find "$root/dav1d-1.5.1" -type f -name dav1d.pc -print -quit 2>/dev/null)"
    test -n "$dav1d_pc" || { echo 'STOP: dav1d not installed' >&2; exit 1; }
    export PKG_CONFIG_PATH="$(dirname "$dav1d_pc")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    dav1d_lib="$(pkg-config --variable=libdir dav1d)"
    libdirs="$root/ffmpeg-n8.1.3/lib:$dav1d_lib"
    if [ "$PIXVQA_ENABLE_NVIDIA" = 1 ]; then
        cuda="${CUDA_HOME:-/usr/local/cuda-12.8}"
        for candidate in "$cuda/targets/x86_64-linux/lib" "$cuda/lib64" "$cuda/lib"; do
            if [ -f "$candidate/libcudart.so" ]; then libdirs="$candidate:$libdirs"; break; fi
        done
        for candidate in /usr/lib/wsl/lib /usr/lib/x86_64-linux-gnu; do
            if [ -f "$candidate/libcuda.so.1" ]; then libdirs="$candidate:$libdirs"; break; fi
        done
    fi
    export LD_LIBRARY_PATH="$libdirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    report="$(ldd "$PREFIX/bin/pixvqa")"
    printf '%s\n' "$report"
    if grep -Fq 'not found' <<< "$report"; then echo 'STOP: unresolved shared library' >&2; exit 1; fi
    "$PREFIX/bin/pixvqa" --version
    echo 'PASS: PixVQA manually installed and dynamic libraries resolved'
)
```

**To run it in a later terminal**, restore the runtime library search paths (or add appropriate RPATH/RUNPATH during a separate packaging improvement). Do not assume that `bin/pixvqa` runs in a new shell without `LD_LIBRARY_PATH`. Unlike the guided installer, the manual procedure does not add an `LD_LIBRARY_PATH` launcher.

`scripts/install.sh` copies the executable, the model JSON files, the README, the third-party notices, and the PU21 notice. It does not copy FFmpeg, libdav1d, or CUDA. SDR models are in `$PREFIX/share/pixvqa/model`. HDR v1.1.8 models are in `$PREFIX/share/pixvqa/model/hdr_vmaf_v1.1.8` and `$PREFIX/share/pixvqa/model/hdr_vmaf_v1.1.8_hfr`. Pass one file to `--model`.

## 8. Quality assurance

From the root of **the checkout being installed**, with `PIXVQA_SRC` initialized in Section 0:

```bash
cd "$PIXVQA_SRC"
python3 scripts/run_qa.py --profile smoke
```

The smoke profile verifies the CPU build (16 cases in the previously validated environment). After a full NVIDIA build, optionally run:

```bash
PIXVQA_CHECK_GPU=1 python3 scripts/run_qa.py --profile hardware
```

For the wider QA suite:

```bash
python3 scripts/run_qa.py --profile full
```

**QA dependencies are separate from build dependencies.** The full suite can require externally supplied FFmpeg/VVDEC/VMAF CLI tools and fixtures. The FFmpeg build above deliberately installs no CLI. A `SKIP` is not a `PASS`; inspect the JSON reports under `reports/qa/`. In particular, do not distribute a downloaded third-party VVC fixture unless its redistribution rights are established.

## Pinned components and validation limits

| Dependency | Pinned version / requirement |
|---|---|
| dav1d | 1.5.1 shared; commit `42b2b24fb8819f1ed3643aa9cf2a62f03868e3aa` |
| FFmpeg | `n8.1.3`; commit `1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7` |
| libvmaf | commit `74f930d17de8c58e7c2ce55e2b897b0030ebd8ef` (not a release tag; previous tag v3.2.1), static, zimg 3.0.6, CUDA off |
| zimg | tag `release-3.0.6`, commit `f819b14e8f39d1282400b0d9543e8ef73c1b2bbd` |
| VMAF models | eight SDR v1.0.16 JSON files and eight HDR v1.1.8 JSON files in `model/` |
| CUDA / NVDEC | CUDA 12.8, preinstalled GPU driver, private `nv-codec-headers`, `sm_89` and `sm_120` |
| CUDA host compiler | conda-forge GCC 13.4; final FFmpeg/PixVQA combined linker `/usr/bin/g++` |

**Previously observed:** The guided installer has passed smoke QA and the built GPU executable reported `build: cuda` / `decode: software nvdec` on the Ubuntu 22.04 WSL2 RTX 4070 setup. A separate prior hardware QA run passed 3/3. These observations are **not** evidence that this rewritten manual procedure has been executed end to end. The NVIDIA header-bootstrap and CPU-only branches also require independent validation.

### Troubleshooting and safe recovery

- **A block prints `STOP` or no final `PASS`:** Do not continue. Fix that exact stage; the surrounding subshell means your Ubuntu window stays open.
- **FFmpeg `config.h` or `libavcodec.so` missing:** Section 3B did not succeed. Inspect `deps/manual/src/ffmpeg/ffbuild/config.log` or `config.log` where present; do not run `make install` before configure and compilation succeed.
- **Git clone fails (HTTP/2/RPC):** Retry the clone after inspecting its destination. The FFmpeg block uses HTTP/1.1 and has an official mirror fallback; it will not delete a partial checkout automatically.
- **libdav1d not found:** The blocks find `dav1d.pc` and put its directory on `PKG_CONFIG_PATH`; library dirs may be `lib/x86_64-linux-gnu` rather than `lib`.
- **`libvmaf: disabled`:** Verify `deps/libvmaf-74f930d` and `scripts/libvmaf_flags.sh`. The helper refuses a prefix without `vmaf_picture_convert` or `-lzimg`, including a leftover `deps/libvmaf-3.2.1`. A system-wide libvmaf is not a substitute.
- **CUDA `-lcudart` unresolved:** The GPU block searches `$CUDA_HOME/targets/x86_64-linux/lib`, `$CUDA_HOME/lib64` and `$CUDA_HOME/lib`, exporting both `LIBRARY_PATH` and `LD_LIBRARY_PATH`.
- **`hypot@GLIBC_2.35` in the combined link:** Preserve `PIXVQA_FULL_LINK_CXX=/usr/bin/g++`; the conda-forge host compiler is used for CUDA compilation, not the final Linux link.
- **Reopened terminal:** Repeat Section 0; each subsequent build block rediscovers its dependencies. Do not set `PIXVQA_SRC` from the FFmpeg directory.
- **Existing incomplete dependency source directory:** Inspect it rather than deleting or silently replacing it. This manual test is deliberately non-destructive.

Consult [`README.md`](../README.md) for usage, [`INSTALL_VALIDATION.md`](INSTALL_VALIDATION.md) for test criteria, and [third-party notices](../THIRD_PARTY_NOTICES.md) for licensing.
