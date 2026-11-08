<p align="center">
  <img src="docs/brand/pixop-vqa-logo-crop.png" alt="Pixop VQA" width="640">
</p>

# PixVQA — Video Quality Assessment by Pixop

[![CPU CI](https://github.com/pixop/pixvqa/actions/workflows/qa.yml/badge.svg?branch=main)](https://github.com/pixop/pixvqa/actions/workflows/qa.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PixVQA compares reference and distorted video using **PSNR, XPSNR, SSIM, MS-SSIM and VMAF**, plus HDR **PU21** metrics. Optional Python-backed models add full-reference **ColorVideoVDP**; no-reference **CompressedVQA-HDR NR**, **SimpleVQA**, and **HDR-ChipQA** (`pixvqa --metrics simplevqa` or `hdr-chipqa`). The same optional **repository-local** `.venv` can also satisfy the standalone **SimpleVQA reference runner** (`python -m pixvqa_simplevqa`) — see [Learned Metrics](docs/LEARNED_METRICS.md). [Quality Consistency Score (QCS)](docs/QCS.md) reports temporal quality reliability.

PixVQA accepts planar raw YUV; compressed inputs require an FFmpeg-enabled build. CPU, optional CUDA metric acceleration, and NVDEC are available depending on the build.

## Metrics at a glance

**Full-reference (FR)** metrics need a reference and a distorted video; **no-reference (NR)** metrics score the distorted video alone. The tables below list **recommended** choices by transfer characteristic. Input requirements (raw YUV, compressed, tags) are stricter than model availability — see [Metrics & Formats](docs/METRICS_AND_FORMATS.md#recommended-metrics-for-sdr-and-hdr-video) and [Learned Metrics](docs/LEARNED_METRICS.md).

### Recommended metrics by video format

| Metric | SDR (BT.709) | PQ / BT.2020 (HDR10) | HLG (BT.2020) |
|---|:---:|:---:|:---:|
| PSNR | ✅ | ⚠️ | ⚠️ |
| XPSNR | ✅ | ⚠️ | ⚠️ |
| SSIM | ✅ | ⚠️ | ⚠️ |
| MS-SSIM | ✅ | ⚠️ | ⚠️ |
| VMAF (SDR v1.0.16 models) | ✅ | ❌ | ❌ |
| VMAF (HDR v1.1.8 models) | ❌ | ✅ | ❌ |
| PU21-PSNR | ❌ | ✅ | ✅ |
| PU21-SSIM | ❌ | ✅ | ✅ |
| PU21-MS-SSIM | ❌ | ✅ | ✅ |
| CVVDP (`sdr` / `pq` / `hlg` interpretation) | ✅ | ✅ | ✅ |
| CompressedVQA-HDR NR | ✅ | ✅ | ❌ |
| SimpleVQA (`simplevqa`) | ✅ | ❌ | ❌ |
| HDR-ChipQA (`hdr-chipqa`) | ✅ | ✅ | 🔬 |

**Legend:** ✅ Recommended · ⚠️ Supported, but not generally recommended for HDR perceptual assessment · 🔬 Experimental (runs in PixVQA; scores not paper-validated) · ❌ Unsupported. NR metrics **SimpleVQA** and **HDR-ChipQA** report one pooled score per video and do not support QCS. See [Metrics & Formats](docs/METRICS_AND_FORMATS.md#recommended-metrics-for-sdr-and-hdr-video) for YUV format and transfer-tag caveats.

### Metric execution backends

| Metric | CPU C/C++ | CUDA C++ | CPU Python (NumPy/SciPy) | CPU PyTorch | CUDA PyTorch |
|---|:---:|:---:|:---:|:---:|:---:|
| PSNR | ✅ | ✅ | — | — | — |
| XPSNR | ✅ | ✅ | — | — | — |
| SSIM | ✅ | ✅ | — | — | — |
| MS-SSIM | ✅ | ✅ | — | — | — |
| VMAF | ✅ | — | — | — | — |
| PU21-PSNR | ✅ | ✅ | — | — | — |
| PU21-SSIM | ✅ | ✅ | — | — | — |
| PU21-MS-SSIM | ✅ | ✅ | — | — | — |
| ColorVideoVDP (`cvvdp`) | — | — | — | ✅ | ✅ |
| CompressedVQA-HDR NR (`cvqa-hdr-nr`) | — | — | — | ✅ | ✅ |
| SimpleVQA (`simplevqa`) | — | — | — | ✅ | ✅ |
| HDR-ChipQA (`hdr-chipqa`) | — | — | ✅ | — | — |

✅ Available primary computation backend · — Not implemented for that backend. Python-based metrics may use other libraries for decoding or preprocessing; CUDA C++ refers to native CUDA metric kernels. QCS is a temporal descriptor derived from eligible frame scores, not a separate execution backend. See [Metrics & Formats](docs/METRICS_AND_FORMATS.md#metric-execution-backends) for details.

## Install

Supported: Ubuntu 22.04/24.04 x86-64 (including WSL2). Native Windows builds are not validated.

```bash
git clone https://github.com/pixop/pixvqa.git
cd pixvqa
bash install_pixvqa.sh
./bin/pixvqa --version
```

The guided installer offers software decoding and optional CUDA/NVDEC support (CUDA 12.8 and a compatible NVIDIA driver must already be installed). Optional learned Python metrics are **off by default** (prompt defaults to no; use `PIXVQA_ENABLE_LEARNED=1` to enable non-interactively). Python-backed metrics always need [separate environment and asset setup](docs/LEARNED_METRICS.md); the guided step only runs the helper script when you opt in.

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
