#!/usr/bin/env bash
# PixVQA guided installer. Mirrors docs/INSTALL.md (23 October 2026).
# Experimental: verify each branch on the target system before release.
# Revision: 2026-10-23-r6-glibc-vmaf
set -Eeuo pipefail
trap 'printf "\nERROR: installation failed at line %s (command: %s).\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

info() { printf '\n==> %s\n' "$*"; }
die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }
ask() {
    local answer
    read -r -p "$1" answer </dev/tty || die 'Interactive terminal required.'
    case "${answer,,}" in
        y|yes) return 0 ;;
        n|no) return 1 ;;
        '') [[ "$2" == yes ]] ;;
        *) die "Invalid response '$answer'. Please rerun and enter y or n." ;;
    esac
}

[[ "$(uname -s)" == Linux ]] || die 'Run inside Linux/Ubuntu, not Git Bash.'
[[ "$(uname -m)" == x86_64 ]] || die 'Only Linux x86-64 is currently supported.'
[[ -f /etc/os-release ]] || die 'Unable to identify Linux distribution.'

# Discard PixVQA-specific settings inherited from earlier builds. This script
# runs in its own Bash process, so the caller's shell is not changed.
# Keep CUDA_HOME (toolkit) and PIXVQA_DEPS_ROOT (optional isolated third-party
# directory). PREFIX is resolved after the checkout is known: another PixVQA
# checkout is never reused, and any other PREFIX remains a custom location.
# Also retain general toolchain/library search paths used by other software.
unset PIXVQA_SRC PIXVQA_BIN LIBVMAF_PREFIX FFMPEG_PREFIX
unset DAV1D_PREFIX DAV1D_LIB GCC13 GCC_ENV CUDART_LIBDIR
unset PIXVQA_LIBCUDA_DIR PIXVQA_FULL_LINK_CXX
unset LIBVMAF_SRC LIBVMAF_BUILD ZIMG_PREFIX ZIMG_SRC
unset ZIMG_AUTOTOOLS_PREFIX ZIMG_AUTOTOOLS_DEBS

# The script lives at the root of the source checkout.
export PIXVQA_SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for file in build.sh build_decode.sh build_cuda.sh scripts/build_libvmaf.sh scripts/deps_layout.sh scripts/install.sh; do
    [[ -f "$PIXVQA_SRC/$file" ]] || die "Expected source file missing: $PIXVQA_SRC/$file"
done
[[ "$PIXVQA_SRC" != "$HOME" ]] || die 'Refusing to treat the home directory as the PixVQA repository.'
# shellcheck disable=SC1091
source "$PIXVQA_SRC/scripts/install_prefix.sh"
# shellcheck disable=SC1091
source "$PIXVQA_SRC/scripts/git_retry.sh"
# shellcheck disable=SC1091
source "$PIXVQA_SRC/scripts/install_choices.sh"
# Default to this checkout. A PREFIX inherited from another checkout is ignored.
export PREFIX="$(pixvqa_select_prefix "$PIXVQA_SRC")"
# shellcheck disable=SC1091
source "$PIXVQA_SRC/scripts/deps_layout.sh"
pixvqa_apply_deps_layout "$PIXVQA_SRC" || die 'Could not resolve the third-party dependency layout.'
export DAV1D_PREFIX FFMPEG_PREFIX GCC13 MAMBA_ROOT_PREFIX LIBVMAF_PREFIX
if (( PIXVQA_DEPS_ISOLATED )); then
    export LIBVMAF_SRC LIBVMAF_BUILD ZIMG_PREFIX ZIMG_SRC
    export ZIMG_AUTOTOOLS_PREFIX ZIMG_AUTOTOOLS_DEBS
fi

printf 'Installer revision: 2026-10-23-r6-glibc-vmaf\n'
printf 'PixVQA checkout: %s\n' "$PIXVQA_SRC"
printf 'Install prefix: %s\n' "$PREFIX"
if (( PIXVQA_DEPS_ISOLATED )); then
    printf 'Isolated third-party root: %s\n' "$PIXVQA_DEPS_ROOT"
else
    printf 'Third-party source workspace: %s\n' "$DEPS_SRC"
fi
printf 'dav1d prefix: %s\n' "$DAV1D_PREFIX"
printf 'FFmpeg prefix: %s\n' "$FFMPEG_PREFIX"
printf 'NOTE: apt packages, the NVIDIA driver, and the CUDA toolkit are not moved by PIXVQA_DEPS_ROOT.\n'
# PIXVQA_ENABLE_DAV1D and PIXVQA_ENABLE_NVIDIA accept 1/0 (or yes/no).
# When either variable is unset, the installer still asks on a terminal.
dav1d_choice="$(pixvqa_choice PIXVQA_ENABLE_DAV1D)"
case "$dav1d_choice" in
    yes) WITH_DAV1D=1 ;;
    no) WITH_DAV1D=0 ;;
    unset)
        if ask 'Install dav1d 1.5.1 for software AV1 decoding? [Y/n] ' yes; then WITH_DAV1D=1; else WITH_DAV1D=0; fi
        ;;
    *) die 'PIXVQA_ENABLE_DAV1D must be 1/yes or 0/no.' ;;
esac
nvidia_choice="$(pixvqa_choice PIXVQA_ENABLE_NVIDIA)"
case "$nvidia_choice" in
    yes) WITH_NVIDIA=1 ;;
    no) WITH_NVIDIA=0 ;;
    unset)
        if ask 'Build NVIDIA GPU support (CUDA metrics + NVDEC)? [y/N] ' no; then WITH_NVIDIA=1; else WITH_NVIDIA=0; fi
        ;;
    *) die 'PIXVQA_ENABLE_NVIDIA must be 1/yes or 0/no.' ;;
esac
printf 'dav1d: %s\n' "$( (( WITH_DAV1D )) && printf yes || printf no )"
printf 'NVIDIA: %s\n' "$( (( WITH_NVIDIA )) && printf yes || printf no )"

# build_decode.sh expects a concrete dav1d library path in the documented setup.
# Rather than claim untested no-dav1d decoder support, choose the raw-YUV CPU build.
if (( !WITH_DAV1D )); then
    if (( WITH_NVIDIA )); then
        die 'NVIDIA + no-dav1d is not validated by the supplied INSTALL.md. Select dav1d=Yes for the full build.'
    fi
    info 'No-dav1d choice: installing the raw-YUV CPU binary only (no compressed decoding).'
fi

export GCC_ENV=/usr
# Active HDR pin. scripts/build_libvmaf_3.2.1.sh remains the historical recipe.
# DAV1D_PREFIX, FFMPEG_PREFIX, LIBVMAF_PREFIX, GCC13, and MAMBA_ROOT_PREFIX
# come from scripts/deps_layout.sh.

# Private, user-owned shared-library directories are not in the default
# dynamic loader search path. Preserve the original path and add each prefix
# exactly once, including for ldd, QA and installed-binary verification.
require_pkgconfig_prefix() {
    local module="$1" expected="$2" got
    got="$(pkg-config --variable=prefix "$module" 2>/dev/null || true)"
    [[ -n "$got" ]] || die "pkg-config has no prefix for $module"
    [[ "$(cd -- "$got" && pwd -P)" == "$(cd -- "$expected" && pwd -P)" ]] ||
        die "$module pkg-config prefix $got is not $expected"
}
add_runtime_libdir() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$dir:"*) ;;
        *) export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}
show_missing_libraries() {
    local target="$1" report status=0
    # ldd may print useful diagnostics even when it returns nonzero.
    report="$(ldd "$target" 2>&1)" || status=$?
    printf 'Dynamic-linker report for %s:\n%s\n' "$target" "$report"
    if (( status != 0 )) || grep -Fq 'not found' <<< "$report"; then
        printf 'STOP: Runtime linker could not resolve dependencies for %s.\n' "$target" >&2
        printf 'Current LD_LIBRARY_PATH: %s\n' "${LD_LIBRARY_PATH:-<empty>}" >&2
        printf 'Unresolved entries (if present):\n' >&2
        grep -F 'not found' <<< "$report" >&2 || true
        return 1
    fi
}
# FFmpeg's libavcodec depends on other FFmpeg DSOs in the same private prefix.
add_runtime_libdir "$FFMPEG_PREFIX/lib"

info 'Checking/installing Ubuntu build prerequisites'
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    [[ "${ID:-}" == ubuntu ]] || die 'Automated APT setup supports Ubuntu only; see docs/INSTALL.md for manual installation.'
    case "${VERSION_ID:-}" in 22.04|24.04) ;; *) die "Ubuntu ${VERSION_ID:-unknown} is not among the documented versions." ;; esac
fi
require sudo
sudo apt-get update
# perl is required by autoconf. autoconf, automake, libtool, and m4 are the
# distribution packages for the zimg bootstrap. Do not fetch another Ubuntu
# release's binary packages.
sudo apt-get install -y build-essential python3 python3-numpy git meson ninja-build nasm pkg-config curl ca-certificates bzip2 tar perl autoconf automake libtool m4
for tool in g++ python3 git meson ninja nasm pkg-config make curl tar perl autoconf automake libtoolize m4; do require "$tool"; done
[[ -x /usr/bin/g++ ]] || die 'Expected /usr/bin/g++ for GCC_ENV=/usr and for the final NVDEC link.'
mkdir -p "$DEPS_SRC"

clone_pinned() {
    local url="$1" ref="$2" dest="$3" expected="$4"
    if [[ -e "$dest" ]] && { [[ ! -d "$dest/.git" ]] || ! git -C "$dest" rev-parse --verify HEAD >/dev/null 2>&1; }; then
        printf 'Removing incomplete checkout %s\n' "$dest" >&2
        rm -rf -- "$dest"
    fi
    if [[ ! -e "$dest" ]]; then
        pixvqa_git_clone "$ref" "$url" "$dest" || die "Could not clone $url"
    fi
    [[ -d "$dest/.git" ]] || die "Existing $dest is not a Git checkout. Inspect it manually."
    local got
    got="$(git -C "$dest" rev-parse HEAD)"
    if [[ -n "$expected" ]]; then
        [[ "$got" == "$expected" ]] || die "$dest HEAD $got differs from expected $expected. Do not overwrite existing sources."
    else
        [[ "$(git -C "$dest" rev-parse "refs/tags/$ref^{commit}")" == "$got" ]] || die "$dest is not at pinned tag $ref."
    fi
}

if (( WITH_DAV1D )); then
    info 'Building dav1d 1.5.1 (shared)'
    clone_pinned https://code.videolan.org/videolan/dav1d.git 1.5.1 "$DEPS_SRC/dav1d" 42b2b24fb8819f1ed3643aa9cf2a62f03868e3aa
    if [[ ! -f "$DEPS_SRC/dav1d/build/build.ninja" ]]; then
        meson setup "$DEPS_SRC/dav1d/build" "$DEPS_SRC/dav1d" --prefix="$DAV1D_PREFIX" --default-library=shared --buildtype=release
    fi
    ninja -C "$DEPS_SRC/dav1d/build"
    ninja -C "$DEPS_SRC/dav1d/build" install
    DAV1D_PC="$(find "$DAV1D_PREFIX" -type f -name dav1d.pc -print -quit)"
    [[ -n "$DAV1D_PC" ]] || die "dav1d.pc missing under $DAV1D_PREFIX"
    export PKG_CONFIG_PATH="$(dirname "$DAV1D_PC")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    [[ "$(pkg-config --modversion dav1d)" == 1.5.1 ]] || die 'Wrong dav1d version.'
    require_pkgconfig_prefix dav1d "$DAV1D_PREFIX"
    export DAV1D_LIB="$(pkg-config --variable=libdir dav1d)"
    [[ -f "$DAV1D_LIB/libdav1d.so" ]] || die "Missing $DAV1D_LIB/libdav1d.so"
    add_runtime_libdir "$DAV1D_LIB"
fi

if (( WITH_NVIDIA )); then
    info 'Checking NVIDIA driver and CUDA 12.8 (never installed automatically)'
    require nvidia-smi
    nvidia-smi || die 'NVIDIA driver/GPU not available in Linux/WSL2.'
    if [[ -z "${CUDA_HOME:-}" ]]; then
        if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then export CUDA_HOME=/usr/local/cuda-12.8
        elif command -v nvcc >/dev/null 2>&1; then
            export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
        fi
    fi
    [[ -x "${CUDA_HOME:-}/bin/nvcc" ]] || die 'CUDA 12.8 nvcc missing. Install the toolkit separately and set CUDA_HOME.'
    "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.8' || die 'CUDA nvcc must be version 12.8.'

    info 'Checking GCC 13.4 host compiler'
    compiler="$GCC13/bin/x86_64-conda-linux-gnu-g++"
    if [[ ! -x "$compiler" ]]; then
        if command -v conda >/dev/null 2>&1; then
            conda create -y -p "$GCC13" --override-channels -c conda-forge 'gxx_linux-64=13.4'
        elif command -v micromamba >/dev/null 2>&1; then
            micromamba create -y -p "$GCC13" --override-channels -c conda-forge 'gxx_linux-64=13.4'
        else
            mkdir -p "$PIXVQA_TOOLS_DIR"
            curl -fLsS https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$PIXVQA_TOOLS_DIR/micromamba.tar.bz2"
            tar -xjf "$PIXVQA_TOOLS_DIR/micromamba.tar.bz2" -C "$PIXVQA_TOOLS_DIR" bin/micromamba
            "$PIXVQA_TOOLS_DIR/bin/micromamba" create -y -p "$GCC13" --override-channels -c conda-forge 'gxx_linux-64=13.4'
        fi
    fi
    [[ -x "$compiler" ]] || die "Missing $compiler"
    "$compiler" -dumpfullversion | grep -Eq '^13\.4(\.|$)' || die 'GCC host compiler is not version 13.4.'

    info 'Checking or installing NVIDIA codec headers before FFmpeg configure'
    # Isolated mode accepts only the headers installed under NVCODEC_PREFIX.
    # The legacy mode still reuses an ffnvcodec package already visible to pkg-config.
    nvcodec_ready=0
    if (( PIXVQA_DEPS_ISOLATED )); then
        if [[ -f "$NVCODEC_PREFIX/lib/pkgconfig/ffnvcodec.pc" ]]; then nvcodec_ready=1; fi
    elif pkg-config --exists ffnvcodec; then
        nvcodec_ready=1
    fi
    if (( ! nvcodec_ready )); then
        clone_pinned https://github.com/FFmpeg/nv-codec-headers.git n13.0.19.0 "$DEPS_SRC/nv-codec-headers" ''
        make -C "$DEPS_SRC/nv-codec-headers" PREFIX="$NVCODEC_PREFIX" install
    fi
    export PKG_CONFIG_PATH="$NVCODEC_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    pkg-config --modversion ffnvcodec || die 'ffnvcodec headers unavailable.'
    if (( PIXVQA_DEPS_ISOLATED )); then
        require_pkgconfig_prefix ffnvcodec "$NVCODEC_PREFIX"
    fi
fi

if (( WITH_DAV1D )); then
    info 'Building FFmpeg n8.1.3 shared decoding libraries'
    clone_pinned https://git.ffmpeg.org/ffmpeg.git n8.1.3 "$DEPS_SRC/ffmpeg" 1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7
    FFMPEG_HW_ARGS=()
    if (( WITH_NVIDIA )); then FFMPEG_HW_ARGS=(--enable-ffnvcodec --enable-nvdec --enable-cuvid); fi
    # Resume after a previous successful FFmpeg make install. Reuse only if
    # the installed headers and libraries exist AND the recorded configuration
    # matches the requested CPU/NVIDIA feature selection.
    ffmpeg_ready=0
    if [[ -f "$FFMPEG_PREFIX/lib/libavcodec.so" &&
          -f "$FFMPEG_PREFIX/include/libavcodec/avcodec.h" &&
          -f "$DEPS_SRC/ffmpeg/config.h" ]] &&
       grep -Eq '^#define CONFIG_LIBDAV1D 1$' "$DEPS_SRC/ffmpeg/config.h"; then
        ffmpeg_ready=1
        for feature in FFNVCODEC NVDEC CUVID; do
            if (( WITH_NVIDIA )); then
                grep -Eq "^#define CONFIG_${feature} 1$" "$DEPS_SRC/ffmpeg/config.h" || ffmpeg_ready=0
            else
                if grep -Eq "^#define CONFIG_${feature} 1$" "$DEPS_SRC/ffmpeg/config.h"; then ffmpeg_ready=0; fi
            fi
        done
    fi
    if (( ffmpeg_ready )); then
        info 'Reusing previously installed FFmpeg libraries (matching config.h)'
    else
        info 'Configuring and compiling FFmpeg (existing outputs absent or incompatible)'
        (
            cd "$DEPS_SRC/ffmpeg"
            ./configure \
              --prefix="$FFMPEG_PREFIX" \
              --enable-shared --disable-static \
              --disable-gpl --disable-version3 \
              --disable-doc --disable-programs \
              --disable-avdevice --disable-avfilter \
              --disable-swscale --disable-swresample \
              --disable-encoders --disable-muxers \
              --disable-autodetect \
              --enable-libdav1d --enable-protocol=file \
              "${FFMPEG_HW_ARGS[@]}" \
              --extra-cflags="-I$DAV1D_PREFIX/include" \
              --extra-ldflags="-L$DAV1D_LIB -Wl,-rpath,$DAV1D_LIB"
            make -j"$(nproc)"
            make install
        )
    fi
    [[ -f "$FFMPEG_PREFIX/lib/libavcodec.so" ]] || die 'FFmpeg libavcodec.so missing.'
    [[ -f "$FFMPEG_PREFIX/include/libavcodec/avcodec.h" ]] || die 'FFmpeg avcodec.h missing.'
    if (( WITH_NVIDIA )); then
        for feature in FFNVCODEC NVDEC CUVID; do
            grep -Eq "^#define CONFIG_${feature} 1$" "$DEPS_SRC/ffmpeg/config.h" || die "FFmpeg did not enable $feature."
        done
    fi
    # Ensure both private prefixes are visible to the ELF loader before checking.
    add_runtime_libdir "$FFMPEG_PREFIX/lib"
    add_runtime_libdir "$DAV1D_LIB"
    show_missing_libraries "$FFMPEG_PREFIX/lib/libavcodec.so" || die 'FFmpeg dependency check failed; inspect linker diagnostics above.'
fi

info 'Building pinned libvmaf and PixVQA CPU binary'
cd "$PIXVQA_SRC"
bash scripts/build_libvmaf.sh
[[ -d "$LIBVMAF_PREFIX" ]] || die 'libvmaf dependency prefix missing.'
# Validate actual pinned archive; pkg-config without a targeted search path
# can accidentally resolve a different installation under /usr/local.
VMAF_ARCHIVE="$(find "$LIBVMAF_PREFIX" -type f -name libvmaf.a -print -quit)"
[[ -f "$VMAF_ARCHIVE" ]] || die "Pinned static libvmaf.a missing under $LIBVMAF_PREFIX"
[[ -f "$LIBVMAF_PREFIX/include/libvmaf/libvmaf.h" ]] || die 'Pinned libvmaf headers missing.'
VMAF_PC="$(find "$LIBVMAF_PREFIX" -type f -name libvmaf.pc -print -quit)"
[[ -n "$VMAF_PC" ]] || die 'Pinned libvmaf.pc missing.'
export PKG_CONFIG_PATH="$(dirname "$VMAF_PC")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
printf 'Pinned libvmaf archive: %s\n' "$VMAF_ARCHIVE"
# The build scripts source scripts/libvmaf_flags.sh, which must recognize
# architecture-specific libdirs and avoid accidentally selecting /usr/local.
source "$PIXVQA_SRC/scripts/libvmaf_flags.sh"
pixvqa_detect_libvmaf "$PIXVQA_SRC"
if [[ "$PIXVQA_HAS_VMAF" != 1 ]]; then
    printf 'Pinned libvmaf prefix: %s\n' "$LIBVMAF_PREFIX" >&2
    printf 'Helper loaded from: %s\n' "$PIXVQA_SRC/scripts/libvmaf_flags.sh" >&2
    printf 'Helper SHA256: ' >&2
    sha256sum "$PIXVQA_SRC/scripts/libvmaf_flags.sh" >&2
    die 'libvmaf archive exists but detection is disabled. Ensure the new helper is installed at scripts/libvmaf_flags.sh (not repository root).'
fi
printf 'Pinned libvmaf enabled: CFLAGS=%s LIBS=%s\n' "$PIXVQA_VMAF_CFLAGS" "$PIXVQA_VMAF_LIBS"
case "$PIXVQA_VMAF_LIBS" in
    *-lzimg*) ;;
    *) die 'Pinned libvmaf is not linked with zimg.' ;;
esac
case "$PIXVQA_VMAF_CFLAGS" in
    *74f930d17de8c58e7c2ce55e2b897b0030ebd8ef*) ;;
    *) die 'Pinned libvmaf stamp is not commit 74f930d.' ;;
esac
grep -q vmaf_picture_convert "$LIBVMAF_PREFIX/include/libvmaf/picture.h" || die 'Pinned picture.h has no vmaf_picture_convert.'

bash build.sh
[[ -x build/cpu/pixvqa ]] || die 'CPU binary missing.'

if (( WITH_DAV1D )); then
    info 'Building PixVQA software decoder'
    # Prevent the software-decoder step from initiating a premature CUDA build.
    CUDA_SAVED="${CUDA_HOME:-}"
    unset CUDA_HOME
    bash build_decode.sh
    [[ -x build/swdec/pixvqa ]] || die 'Software-decoding binary missing.'
    if (( WITH_NVIDIA )); then
        export CUDA_HOME="$CUDA_SAVED"
        info 'Building PixVQA CUDA and combined NVDEC binary'
        if [[ -z "${PIXVQA_LIBCUDA_DIR:-}" ]]; then
            if [[ -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then export PIXVQA_LIBCUDA_DIR=/usr/lib/wsl/lib
            elif [[ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]]; then export PIXVQA_LIBCUDA_DIR=/usr/lib/x86_64-linux-gnu
            fi
        fi
        [[ -f "${PIXVQA_LIBCUDA_DIR:-}/libcuda.so.1" ]] || die 'libcuda.so.1 not found; set PIXVQA_LIBCUDA_DIR.'
        add_runtime_libdir "$PIXVQA_LIBCUDA_DIR"
        # GCC/ld use LIBRARY_PATH for -l searches; LD_LIBRARY_PATH is for the
        # runtime loader and does not reliably resolve -lcudart at link time.
        # CUDA 12.8 commonly installs libcudart under targets/x86_64-linux/lib.
        CUDART_LIBDIR=""
        for candidate in \
            "$CUDA_HOME/targets/x86_64-linux/lib" \
            "$CUDA_HOME/lib64" \
            "$CUDA_HOME/lib" \
            "$CUDA_HOME/lib/x64"; do
            if [[ -f "$candidate/libcudart.so" ]]; then
                CUDART_LIBDIR="$candidate"
                break
            fi
        done
        if [[ -z "$CUDART_LIBDIR" ]]; then
            printf 'CUDA_HOME=%s\n' "$CUDA_HOME" >&2
            printf 'Candidate libcudart locations checked under CUDA_HOME; none contained libcudart.so.\n' >&2
            find "$CUDA_HOME" -maxdepth 6 \
                \( -name 'libcudart.so*' -o -name 'libcudart_static.a' \) \
                -print >&2 || true
            die 'CUDA 12.8 development runtime (libcudart.so) is missing. Check your toolkit installation.'
        fi
        export CUDART_LIBDIR
        export LIBRARY_PATH="$CUDART_LIBDIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
        add_runtime_libdir "$CUDART_LIBDIR"
        printf 'CUDA runtime link directory: %s\n' "$CUDART_LIBDIR"
        printf 'CUDA runtime library: %s\n' "$(readlink -f "$CUDART_LIBDIR/libcudart.so")"
        bash build_cuda.sh
        # Use Ubuntu glibc to resolve symbols required by the Ubuntu-built FFmpeg.
        export PIXVQA_FULL_LINK_CXX=/usr/bin/g++
        bash build_decode.sh
        [[ -x build/full/pixvqa ]] || die 'Full NVIDIA binary missing.'
        full_version="$(build/full/pixvqa --version)"
        grep -q 'decode: software nvdec' <<< "$full_version" || die 'Full binary does not report NVDEC.'
        PIXVQA_BIN=build/full/pixvqa
        PIXVQA_EXPECT_BUILD=cuda
        PIXVQA_EXPECT_DECODE="software nvdec"
    else
        PIXVQA_BIN=build/swdec/pixvqa
        PIXVQA_EXPECT_BUILD=cpu
        PIXVQA_EXPECT_DECODE=software
    fi
else
    PIXVQA_BIN=build/cpu/pixvqa
    PIXVQA_EXPECT_BUILD=cpu
    PIXVQA_EXPECT_DECODE=none
fi
export PIXVQA_BIN

info 'Installing PixVQA application'
[[ -x "$PIXVQA_BIN" ]] || die "Selected binary not executable: $PIXVQA_BIN"
PREFIX="$PREFIX" PIXVQA_BIN="$PIXVQA_BIN" bash scripts/install.sh
[[ -x "$PREFIX/bin/pixvqa" ]] || die 'Installed executable missing.'
bash scripts/verify_installed_pixvqa.sh "$PREFIX" "$PIXVQA_EXPECT_BUILD" "$PIXVQA_EXPECT_DECODE"

info 'Running smoke QA on the installed executable'
pixvqa_run_installed_smoke "$PIXVQA_SRC" "$PREFIX"
PIXVQA_SRC="$PIXVQA_SRC" bash scripts/finish_install.sh
info 'Installation succeeded (smoke QA passed). Full and hardware QA require separate validation.'
printf 'Installed executable: %s\n' "$PREFIX/bin/pixvqa"
printf 'Run it directly. Private libraries are recorded in the executable RPATH.\n'
