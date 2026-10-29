<p align="center">
  <img src="docs/brand/pixop-vqa-logo-crop.png" alt="Pixop VQA" width="640">
</p>

# PixVQA — A Unified Video Quality Assessment Tool by Pixop

[![CPU CI](https://github.com/pixop/pixvqa/actions/workflows/qa.yml/badge.svg?branch=main)](https://github.com/pixop/pixvqa/actions/workflows/qa.yml) [![FFmpeg](https://img.shields.io/badge/FFmpeg-n8.1.3-blue)](https://ffmpeg.org/) [![libvmaf](https://img.shields.io/badge/libvmaf-74f930d-blue)](https://github.com/Netflix/vmaf) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PixVQA compares a reference sequence with a distorted sequence and prints text scores for the selected metrics. The default lists each frame and then a summary. There is no JSON mode. It reads raw planar YUV and, when built with FFmpeg, compressed video. Decoding uses the codec FFmpeg reports.

## Features

PixVQA supports PSNR, XPSNR, SSIM, MS-SSIM, VMAF, and PU21-PSNR/SSIM/MS-SSIM, with optional temporal Quality Consistency Score (QCS). It accepts raw YUV and, with an FFmpeg-enabled build, compressed video. CPU implementations are available for every metric; optional CUDA acceleration is available for selected metrics and NVDEC decoding of supported HEVC/AV1 streams.

See [supported metrics, SDR/HDR recommendations, pixel formats, and CPU/GPU capabilities](docs/METRICS_AND_FORMATS.md).

## Quick Start

Supported platform: Linux x86-64 (including Ubuntu 22.04/24.04 and WSL2). Native Windows builds have not been validated.

```bash
git clone https://github.com/pixop/pixvqa.git
cd pixvqa
bash install_pixvqa.sh
./bin/pixvqa --version
python3 scripts/run_qa.py --profile smoke
```

The guided installer sets up dependencies and lets you select CPU-only, FFmpeg software decoding, or optional CUDA/NVDEC support. CUDA/NVDEC requires a compatible NVIDIA driver and CUDA toolkit 12.8 installed.

To compare the bundled HEVC samples (requires software decoding or the full build):

```bash
./bin/pixvqa --metrics psnr,ssim,msssim \
  -r samples/ref_hevc.mp4 \
  -t samples/dist_hevc.mp4
```

CPU-only builds use raw YUV inputs. For dependency locations, `make cpu`, `make swdec`, `make full`, installation prefixes, and troubleshooting, read the [installation and build guide](docs/GETTING_STARTED.md).

## Documentation

**Getting started:** [Installation & builds](docs/GETTING_STARTED.md) · [Manual installation](docs/INSTALL.md) · [Make build guide](docs/BUILD_MAKE.md) · [Installation validation](docs/INSTALL_VALIDATION.md)

**Using PixVQA:** [CLI examples & options](docs/CLI_USAGE.md) · [Supported metrics & formats](docs/METRICS_AND_FORMATS.md)

**Technical guides:** [PU21 / HDR](docs/PU21.md) · [QCS](docs/QCS.md) · [CI](docs/CI.md)

**Additional information:** [References](docs/REFERENCES.md) · [Licensing](docs/LICENSING.md)

## License

PixVQA is licensed under the [MIT License](LICENSE) (SPDX `MIT`). Copyright (c) 2026 Pixop ApS. [docs/LICENSING.md](docs/LICENSING.md) explains third-party and model licenses. Dependency notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The logo at `docs/brand/pixop-vqa-logo-crop.png` is included with the repository. The MIT license does not grant unrestricted trademark or logo use. See [docs/brand/README.md](docs/brand/README.md).
