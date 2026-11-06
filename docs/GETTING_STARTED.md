# Installation & Builds

PixVQA supports Ubuntu 22.04/24.04 x86-64 (including WSL2). Native Windows builds are not validated.

## Guided installation (recommended)

```bash
git clone https://github.com/pixop/pixvqa.git
cd pixvqa
bash install_pixvqa.sh
./bin/pixvqa --version
```

The installer offers FFmpeg software decoding and optional CUDA/NVDEC support. NVIDIA requires a compatible driver and CUDA 12.8 already installed. To isolate downloaded dependencies inside the checkout, use `PIXVQA_DEPS_ROOT="$PWD/deps" bash install_pixvqa.sh`.

For noninteractive runs, set `PIXVQA_ENABLE_DAV1D=0|1`, `PIXVQA_ENABLE_NVIDIA=0|1`, and optionally `PIXVQA_ENABLE_LEARNED=1` (the guided prompt for learned metrics defaults to **no**). Python-backed metrics always require [separate environment and asset setup](LEARNED_METRICS.md).

## Build variants

If dependencies are **already installed**, use GNU Make (no downloads):

| Command | Input / acceleration | Output |
|---|---|---|
| `make cpu` | Raw YUV, CPU metrics | `build/cpu/pixvqa` |
| `make swdec` | FFmpeg compressed decoding, CPU metrics | `build/swdec/pixvqa` |
| `make full` | FFmpeg, CUDA metrics and NVDEC | `build/full/pixvqa` |

```bash
make -j"$(nproc)" swdec
make install VARIANT=swdec PREFIX="$PWD"
./bin/pixvqa --version
```

`make install` is required to update `bin/pixvqa`; building a variant alone does not. See [Manual Installation](INSTALL.md) for prerequisites and environment variables.

## First evaluation

```bash
./bin/pixvqa --metrics psnr,ssim,msssim \
  -r samples/ref_hevc.mp4 -t samples/dist_hevc.mp4
```

For raw YUV, color/HDR options, and metric groups, see [CLI Usage](CLI_USAGE.md). Check installed capabilities with `./bin/pixvqa --version`.
