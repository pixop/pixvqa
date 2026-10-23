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
# Keep PREFIX (user-selected installation location) and CUDA_HOME (toolkit).
# Also retain general toolchain/library search paths used by other software.
unset PIXVQA_SRC PIXVQA_BIN LIBVMAF_PREFIX FFMPEG_PREFIX
unset DAV1D_PREFIX DAV1D_LIB GCC13 GCC_ENV CUDART_LIBDIR
unset PIXVQA_LIBCUDA_DIR PIXVQA_FULL_LINK_CXX

# The script lives at the root of the source checkout.
export PIXVQA_SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for file in build.sh build_decode.sh build_cuda.sh scripts/build_libvmaf.sh scripts/install.sh; do
    [[ -f "$PIXVQA_SRC/$file" ]] || die "Expected source file missing: $PIXVQA_SRC/$file"
done
[[ "$PIXVQA_SRC" != "$HOME" ]] || die 'Refusing to treat the home directory as the PixVQA repository.'
# Install into this checkout by default; callers may override PREFIX.
export PREFIX="${PREFIX:-$PIXVQA_SRC}"

printf 'Installer revision: 2026-10-23-r6-glibc-vmaf\n'
printf 'PixVQA checkout: %s\n' "$PIXVQA_SRC"
printf 'Third-party source workspace: %s\n' "$HOME/pixvqa-deps/src"
if ask 'Install dav1d 1.5.1 for software AV1 decoding? [Y/n] ' yes; then WITH_DAV1D=1; else WITH_DAV1D=0; fi
if ask 'Build NVIDIA GPU support (CUDA metrics + NVDEC)? [y/N] ' no; then WITH_NVIDIA=1; else WITH_NVIDIA=0; fi

# build_decode.sh expects a concrete dav1d library path in the documented setup.
# Rather than claim untested no-dav1d decoder support, choose the raw-YUV CPU build.
if (( !WITH_DAV1D )); then
    if (( WITH_NVIDIA )); then
        die 'NVIDIA + no-dav1d is not validated by the supplied INSTALL.md. Select dav1d=Yes for the full build.'
    fi
    info 'No-dav1d choice: installing the raw-YUV CPU binary only (no compressed decoding).'
fi

export GCC_ENV=/usr
export DAV1D_PREFIX="$HOME/dav1d-1.5.1"
export FFMPEG_PREFIX="$HOME/ffmpeg-n8.1.3"
export LIBVMAF_PREFIX="$PIXVQA_SRC/deps/libvmaf-3.2.1"
export GCC13="${GCC13:-$HOME/pixvqa-deps/gcc13}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/pixvqa-deps/micromamba-root}"
DEPS_SRC="$HOME/pixvqa-deps/src"

# Private, user-owned shared-library directories are not in the default
# dynamic loader search path. Preserve the original path and add each prefix
# exactly once, including for ldd, QA and installed-binary verification.
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
sudo apt-get install -y build-essential python3 python3-numpy git meson ninja-build nasm pkg-config curl ca-certificates bzip2 tar
for tool in g++ python3 git meson ninja nasm pkg-config make curl tar; do require "$tool"; done
[[ -x /usr/bin/g++ ]] || die 'Expected /usr/bin/g++ for GCC_ENV=/usr.'
mkdir -p "$DEPS_SRC"

clone_pinned() {
    local url="$1" ref="$2" dest="$3" expected="$4"
    if [[ ! -e "$dest" ]]; then
        git clone --depth 1 --branch "$ref" "$url" "$dest"
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
            mkdir -p "$HOME/pixvqa-deps/tools"
            curl -fLsS https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$HOME/pixvqa-deps/tools/micromamba.tar.bz2"
            tar -xjf "$HOME/pixvqa-deps/tools/micromamba.tar.bz2" -C "$HOME/pixvqa-deps/tools" bin/micromamba
            "$HOME/pixvqa-deps/tools/bin/micromamba" create -y -p "$GCC13" --override-channels -c conda-forge 'gxx_linux-64=13.4'
        fi
    fi
    [[ -x "$compiler" ]] || die "Missing $compiler"
    "$compiler" -dumpfullversion | grep -Eq '^13\.4(\.|$)' || die 'GCC host compiler is not version 13.4.'

    info 'Checking or installing NVIDIA codec headers before FFmpeg configure'
    if ! pkg-config --exists ffnvcodec; then
        clone_pinned https://github.com/FFmpeg/nv-codec-headers.git n13.0.19.0 "$DEPS_SRC/nv-codec-headers" ''
        make -C "$DEPS_SRC/nv-codec-headers" PREFIX="$HOME/pixvqa-deps/nv-codec-headers" install
        export PKG_CONFIG_PATH="$HOME/pixvqa-deps/nv-codec-headers/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    fi
    pkg-config --modversion ffnvcodec || die 'ffnvcodec headers unavailable.'
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
    else
        PIXVQA_BIN=build/swdec/pixvqa
    fi
else
    PIXVQA_BIN=build/cpu/pixvqa
fi
export PIXVQA_BIN

info 'Installing PixVQA application'
[[ -x "$PIXVQA_BIN" ]] || die "Selected binary not executable: $PIXVQA_BIN"
PREFIX="$PREFIX" PIXVQA_BIN="$PIXVQA_BIN" bash scripts/install.sh
[[ -x "$PREFIX/bin/pixvqa" ]] || die 'Installed executable missing.'
"$PREFIX/bin/pixvqa" --version
show_missing_libraries "$PREFIX/bin/pixvqa" || die 'Unresolved runtime libraries in installed PixVQA.'

# Persist a documented launcher for NEW terminal sessions, which do not inherit
# this installer's LD_LIBRARY_PATH. Do not alter user shell startup files.
info 'Writing a launcher with the required private library paths'
launcher="$PREFIX/bin/pixvqa-with-libs"
{
    printf '#!/usr/bin/env bash\nset -e\n'
    printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH:-}"
    printf 'exec %q "$@"\n' "$PREFIX/bin/pixvqa"
} > "$launcher"
chmod +x "$launcher"
"$launcher" --version

info 'Running CPU smoke QA'
python3 scripts/run_qa.py --profile smoke
info 'Installation succeeded (smoke QA passed). Full and hardware QA require separate validation.'
printf 'Installed executable: %s\n' "$PREFIX/bin/pixvqa"
printf 'Portable-to-new-shell launcher (same machine): %s\n' "$launcher"
printf 'NOTE: direct pixvqa execution in a new shell may require LD_LIBRARY_PATH; use the launcher.\n'
