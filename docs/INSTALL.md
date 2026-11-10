# Manual Installation (Ubuntu)

For most users, the [guided installer](GETTING_STARTED.md) is simpler and recommended. This page is for **building PixVQA from source with dependencies already installed**. Supported systems: Ubuntu 22.04/24.04 x86-64 (including WSL2). Native Windows builds are not validated.

## Requirements

- C++17 compiler, GNU Make, Git, Python 3, Meson/Ninja, pkg-config and build tools.
- **Native CPU metrics:** no external FFmpeg dependency for raw YUV. VMAF requires the pinned libvmaf/zimg installation.
- **Compressed inputs:** FFmpeg **n8.1.3** development libraries; optional dav1d **1.5.1** for AV1 software decoding.
- **NVIDIA build:** compatible NVIDIA GPU/driver, CUDA **12.8**, codec headers and GCC **13.4** host toolchain.

Install common Ubuntu build prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git python3 python3-numpy \
  meson ninja-build nasm pkg-config curl ca-certificates \
  perl autoconf automake libtool m4
```

The **guided installer** builds and pins dependencies automatically. For manual dependency compilation, use the maintained scripts `scripts/build_libvmaf.sh`, `scripts/build_zimg.sh`, and the dependency setup in `install_pixvqa.sh`. The manual Make targets **do not download dependencies**.

## Build

From the PixVQA repository root:

```bash
# CPU/raw YUV:
make -j"$(nproc)" cpu
./build/cpu/pixvqa --version
```

When FFmpeg n8.1.3 libraries are installed in a prefix containing `include/` and `lib/libavcodec.so`:

```bash
export FFMPEG_PREFIX=/path/to/ffmpeg-n8.1.3
export GCC_ENV=/usr
make -j"$(nproc)" swdec
./build/swdec/pixvqa --version
```

To build the NVIDIA variant, first configure `CUDA_HOME`, `GCC13`, `FFMPEG_PREFIX`, the codec headers and CUDA runtime as required by `build_cuda.sh` / `scripts/make_config.sh`:

```bash
make -j"$(nproc)" full
./build/full/pixvqa --version
```

If your dependencies are installed under the checkout's `deps/` folder, set `PIXVQA_DEPS_ROOT="$PWD/deps"` before running Make. The script-based alternatives are `bash build.sh`, `bash build_decode.sh`, and `bash build_cuda.sh` (in that order as applicable).

## Install the Selected Build

```bash
make install VARIANT=swdec PREFIX="$PWD"
./bin/pixvqa --version
```

Replace `swdec` with `cpu` or `full` as appropriate. Building alone does not update `bin/pixvqa`. Bundled model JSON files are installed with the program; VMAF chooses an appropriate model automatically when it can.

For Python-backed metrics, run `bash scripts/install_learned_metrics.sh` after a native installation. The script provisions a private Python 3.12 interpreter when missing, then creates the local learned-metric environment; pretrained third-party model assets still require manual configuration. See [Learned Metrics](LEARNED_METRICS.md). For usage, see [CLI Usage](CLI_USAGE.md).

## Rebuilding an Existing Installation

If the guided installer has completed successfully, the ignored
`build/local-config.mk` records its resolved dependency paths. The incremental
Make targets reuse that configuration across terminal sessions, while explicit
environment variables or `make FFMPEG_PREFIX=... full` override it.

```bash
make -j"$(nproc)" full
make install VARIANT=full PREFIX="$PWD"
```

Choose `cpu` or `swdec` (and the corresponding `VARIANT`) if that was the
installed build. `make install` updates `bin/pixvqa`; `make full` alone does not.
If you installed a version before configuration persistence was introduced,
provide the dependency paths as shown above, or use a matching dependency
layout. A normal rebuild does not require running the guided installer again.

## Common Problems

- `FFMPEG_PREFIX must point at FFmpeg n8.1.3`: provide the installed development-library prefix, not just an `ffmpeg` executable.
- `libvmaf: disabled`: run `scripts/build_libvmaf.sh` and ensure its pinned installation is discoverable; otherwise VMAF is unavailable.
- CUDA/host-compiler failures: check `CUDA_HOME`, `GCC13`, NVIDIA driver visibility, and `scripts/make_config.sh`.
- A CPU-only build cannot decode MP4; choose `swdec` or `full`.

See [Installation & Builds](GETTING_STARTED.md) for the recommended guided workflow.
