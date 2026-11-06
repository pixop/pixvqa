<p align="center">
  <img src="docs/brand/pixop-vqa-logo-crop.png" alt="Pixop VQA" width="640">
</p>

# PixVQA — Video Quality Assessment by Pixop

[![CPU CI](https://github.com/pixop/pixvqa/actions/workflows/qa.yml/badge.svg?branch=main)](https://github.com/pixop/pixvqa/actions/workflows/qa.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PixVQA compares video quality using native full-reference metrics and optional Python-backed perceptual models. It supports raw YUV and, with an FFmpeg-enabled build, compressed video. Input formats and restrictions: [Metrics & Formats](docs/METRICS_AND_FORMATS.md).

## Features

- **Native metrics:** PSNR, XPSNR, SSIM, MS-SSIM, VMAF and PU21-PSNR/SSIM/MS-SSIM. VMAF uses a trained model but runs natively through libvmaf.
- **Optional Python-backed metrics:** ColorVideoVDP (`cvvdp`, full-reference) and CompressedVQA-HDR NR (`cvqa-hdr-nr`, no-reference).
- **SDR/HDR:** Explicit metrics or `--metrics sdr|pq|hlg` groups; optional CUDA metric acceleration and NVDEC; temporal [Quality Consistency Score (QCS)](docs/QCS.md).

## Quick start

Linux x86-64 (Ubuntu 22.04/24.04, including WSL2); native Windows builds are not validated.

```bash
git clone https://github.com/pixop/pixvqa.git
cd pixvqa
bash install_pixvqa.sh
./bin/pixvqa --version
```

Choose FFmpeg software decoding or the NVIDIA-enabled build to process compressed video; NVIDIA requires a compatible driver and CUDA 12.8. Optional Python-backed metrics need [separate setup](docs/LEARNED_METRICS.md).

```bash
./bin/pixvqa --metrics psnr,ssim,msssim \
  -r samples/ref_hevc.mp4 -t samples/dist_hevc.mp4
```

For raw input, build variants and manual setup, see [Installation & Builds](docs/GETTING_STARTED.md).

## Usage

```bash
# VMAF automatic model selection requires supported 1080p or 2160p input.
# For raw YUV, declare geometry, frame rate and transfer:
./bin/pixvqa --metrics vmaf --hdr-transfer sdr --fps 60 \
  -r reference.yuv -t distorted.yuv -s 1920x1080 -f 420p:10
```

Use `--device auto|cpu|gpu` to select supported backends (VMAF always uses CPU libvmaf). See [CLI Usage](docs/CLI_USAGE.md) for model-selection limitations, metric groups and options; [Learned metrics](docs/LEARNED_METRICS.md) for optional models; or `./bin/pixvqa --help`, `--help-advanced`, `--help-hdr`.

## Documentation

[Installation & Builds](docs/GETTING_STARTED.md) · [Manual Installation](docs/INSTALL.md) · [CLI Usage](docs/CLI_USAGE.md) · [Metrics & Formats](docs/METRICS_AND_FORMATS.md) · [Optional Python-backed Metrics](docs/LEARNED_METRICS.md) · [PU21](docs/PU21.md) · [QCS](docs/QCS.md) · [References](docs/REFERENCES.md) · [Licensing](docs/LICENSING.md)

## License

MIT © 2026 Pixop ApS. See [Licensing](docs/LICENSING.md) and [brand usage](docs/brand/README.md). The logo is not licensed for unrestricted trademark use.
