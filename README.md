<p align="center">
  <img src="docs/brand/pixop-vqa-logo-crop.png" alt="Pixop VQA" width="640">
</p>

# PixVQA — Video Quality Assessment by Pixop

[![CPU CI](https://github.com/pixop/pixvqa/actions/workflows/qa.yml/badge.svg?branch=main)](https://github.com/pixop/pixvqa/actions/workflows/qa.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PixVQA compares reference and distorted video using **PSNR, XPSNR, SSIM, MS-SSIM and VMAF**, plus HDR **PU21** metrics. Optional Python-backed models add full-reference **ColorVideoVDP** and no-reference **CompressedVQA-HDR NR**. [Quality Consistency Score (QCS)](docs/QCS.md) reports temporal quality reliability.

PixVQA accepts planar raw YUV; compressed inputs require an FFmpeg-enabled build. CPU, optional CUDA metric acceleration, and NVDEC are available depending on the build.

## Install

Supported: Ubuntu 22.04/24.04 x86-64 (including WSL2). Native Windows builds are not validated.

```bash
git clone https://github.com/pixop/pixvqa.git
cd pixvqa
bash install_pixvqa.sh
./bin/pixvqa --version
```

The guided installer offers software decoding and optional CUDA/NVDEC support (CUDA 12.8 and a compatible NVIDIA driver must already be installed). Python-backed metrics need [separate setup](docs/LEARNED_METRICS.md).

## Use

For compressed video (software-decoder or full build):

```bash
./bin/pixvqa --metrics psnr,ssim,msssim \
  -r samples/ref_hevc.mp4 -t samples/dist_hevc.mp4
```

For planar 10-bit 4:2:0 YUV:

```bash
./bin/pixvqa --metrics psnr,ssim --fps 50 \
  -r reference.yuv -t distorted.yuv -s 1920x1080 -f 420p:10
```

Full-reference comparisons stop at the shorter input and warn if frame counts differ. Use `--frames N` to cap evaluation explicitly.

## Documentation

[Installation & Builds](docs/GETTING_STARTED.md) · [Manual Installation](docs/INSTALL.md) · [CLI Usage](docs/CLI_USAGE.md) · [Metrics & Formats](docs/METRICS_AND_FORMATS.md) · [Learned Metrics](docs/LEARNED_METRICS.md) · [PU21](docs/PU21.md) · [QCS](docs/QCS.md) · [References](docs/REFERENCES.md) · [Licensing](docs/LICENSING.md)

Run `./bin/pixvqa --help`, `--help-advanced` or `--help-hdr` for the authoritative option reference.

## License

MIT © 2026 Pixop ApS. See [Licensing](docs/LICENSING.md) for third-party and trademark notices.
