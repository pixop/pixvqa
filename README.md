<p align="center">
  <img src="docs/brand/pixop-vqa-logo-crop.png" alt="Pixop VQA" width="640">
</p>

# PixVQA — A Unified Video Quality Assessment Toolkit by Pixop

[![CPU CI](https://github.com/pixop/pixvqa/actions/workflows/qa.yml/badge.svg?branch=main)](https://github.com/pixop/pixvqa/actions/workflows/qa.yml) [![FFmpeg](https://img.shields.io/badge/FFmpeg-n8.1.3-blue)](https://ffmpeg.org/) [![libvmaf](https://img.shields.io/badge/libvmaf-74f930d-blue)](https://github.com/Netflix/vmaf/commit/74f930d17de8c58e7c2ce55e2b897b0030ebd8ef) [![zimg](https://img.shields.io/badge/zimg-3.0.6-blue)](https://github.com/sekrit-twc/zimg) [![CUDA optional](https://img.shields.io/badge/CUDA-12.8%20optional-blue)](https://developer.nvidia.com/cuda-toolkit) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PixVQA is a **performance-oriented video quality assessment toolkit** for evaluating compression and processing quality across SDR and HDR workflows. It combines established full-reference metrics—**PSNR, XPSNR, SSIM, MS-SSIM and VMAF**—with **PU21-based HDR metrics** and optional learned models, including **ColorVideoVDP, CompressedVQA-HDR NR, SimpleVQA and HDR-ChipQA**. Optimized native CPU processing, optional CUDA acceleration and NVIDIA hardware decoding support efficient evaluation of high-resolution video, with support for both raw YUV and compressed video.

Beyond average quality scores, PixVQA integrates [**Quality Consistency Score (QCS)**](docs/QCS.md), a temporal reliability descriptor that uses frame-level or sampled quality scores to quantify how consistently video quality meets defined thresholds. QCS complements conventional metrics by exposing quality drops that an average score can obscure; it is available for metrics that provide suitable per-frame or sampled scores.

## Metrics at a Glance

**Full-reference** metrics need a reference and a distorted video; **no-reference** metrics score the distorted video alone. The tables below show **supported** metrics by transfer characteristic. Input requirements (raw YUV, compressed, tags) are stricter than model availability — see [Metrics & Formats](docs/METRICS_AND_FORMATS.md#video-format-support-by-metric) and [Learned Metrics](docs/LEARNED_METRICS.md).

### Video Format Support by Metric

| Metric | Type | SDR – BT.709 | PQ / HDR10 – BT.2020 | HLG – BT.2020 |
|---|:---:|:---:|:---:|:---:|
| PSNR | Full-reference | ✅ | ✅ | ✅ |
| XPSNR | Full-reference | ✅ | ✅ | ✅ |
| SSIM | Full-reference | ✅ | ✅ | ✅ |
| MS-SSIM | Full-reference | ✅ | ✅ | ✅ |
| VMAF (SDR v1.0.16 models) | Full-reference | ✅ | ❌ | ❌ |
| VMAF (HDR v1.1.8 models) | Full-reference | ❌ | ✅ | ❌ |
| PU21-PSNR | Full-reference | ❌ | ✅ | ✅ |
| PU21-SSIM | Full-reference | ❌ | ✅ | ✅ |
| PU21-MS-SSIM | Full-reference | ❌ | ✅ | ✅ |
| ColorVideoVDP | Full-reference | ✅ | ✅ | ✅ |
| CompressedVQA-HDR | No-reference | ✅ | ✅ | ❌ |
| SimpleVQA | No-reference | ✅ | ❌ | ❌ |
| HDR-ChipQA | No-reference | ✅ | ✅ | 🔬 |

**Legend:** ✅ Supported · 🔬 Experimental support (validation incomplete) · ❌ Unsupported.

**PU21 metrics:** PSNR, SSIM and MS-SSIM can be computed after applying the PU21 perceptually uniform transfer function. PU21 transforms HDR luminance values to better approximate the nonlinear sensitivity of human vision. As a result, PU21 provides a perceptually motivated signal representation for conventional full-reference metrics.

### Metric Execution Backends

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
| ColorVideoVDP | — | — | — | ✅ | ✅ |
| CompressedVQA-HDR | — | — | — | ✅ | ✅ |
| SimpleVQA | — | — | — | ✅ | ✅ |
| HDR-ChipQA | — | — | ✅ | — | — |

✅ Available primary computation backend · — Not implemented for that backend. Python-based metrics may use other libraries for decoding or preprocessing; CUDA C++ refers to native CUDA metric kernels. QCS is a temporal descriptor derived from eligible frame scores, not a separate execution backend.

## Install

Supported: Ubuntu 22.04/24.04 x86-64 (including WSL2). Native Windows builds are not validated.

```bash
git clone https://github.com/pixop/pixvqa.git
cd pixvqa
bash install_pixvqa.sh
./bin/pixvqa --version
```

The guided installer lets you choose the features you need:

- **Standard installation:** Native CPU metrics and software video decoding.
- **GPU acceleration (optional):** CUDA-accelerated metrics and NVIDIA hardware decoding (NVDEC). Requires **CUDA Toolkit 12.8** and a compatible NVIDIA driver.
- **AI-based metrics (optional):** ColorVideoVDP, CVQA-HDR-NR, SimpleVQA and HDR-ChipQA. These are disabled by default; enable them when prompted, or set `PIXVQA_ENABLE_LEARNED=1` for unattended installation.

AI-based metrics use a separate **Python 3.12** environment with **PyTorch 2.13.0 (CUDA 13.0)**. HDR-ChipQA also needs a small **Python 3.10** environment for its pretrained regression model. You do **not** need a system-wide CUDA 13 toolkit for the Python metrics, and the installer will not overwrite an existing incompatible Python environment.

See [Learned Metrics](docs/LEARNED_METRICS.md) for detailed requirements and setup.

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

### CVQA-HDR-NR estimated-quality calibration (provisional)

PixVQA automatically reports a separate estimated quality rating on a nominal **1–5** scale alongside the canonical CompressedVQA score when transfer metadata and encoded height are supported. The active selected calibration uses **SDR-only (200 observations)** and **HDR PQ-only (192 observations)** fitting, with separate score normalization and M1 sigmoid + M7 corrections. The SDR model additionally uses a +0.5107733 native-MOS correction; HDR has no post-offset. The historical policy ID remains for protocol compatibility, but the calibration version and coefficient hashes have changed.

**Experimental A2 opt-in (20 November 2026, branch `feature/cvqa-mos-a2-tv123-ai25-20261120`):** A unified TV1+TV2+TV3 A2-TV123 calibration (25% AI amplitude tuning) is available via `--mos-calibration-algorithm a2_tv123_ai25` or `PIXVQA_MOS_CALIBRATION_ALGORITHM`. **Default remains frozen v7+C2.** See [A2-TV123 integration](docs/dev/CVQA_MOS_A2_TV123_INTEGRATION.md). Not production-certified.

**Feature-branch update (19 November 2026):** A separately versioned convex C2 SDR local-tail correction has been added to branch **feature/cvqa-mos-v7-local-tail-006** (not main). It enforces estimated MOS(0)=1, lowers the two SDR CVQA 4.134 anchor values by **0.080 MOS**, and returns exactly to frozen v7 for SDR CVQA at or above **5.65** and HDR PQ CVQA at or above **6.268450**. Frozen fitted v7 coefficients and native-MOS evaluation scores remain unchanged. See [selected local C2 curves](docs/dev/CVQA_MOS_V7_LOCAL_TAIL_CONVEX.md). This is research-only, not production-certified.

These are research-derived estimates, **not viewer-measured MOS or production-certified predictions**. Retrospective SDR54 MAE is 0.2780 after source-held-out correction; HDR35 MAE is 0.6042, worse than the old mixed-data model. Native-rated high-quality HDR MOS ≥4.5 remains unvalidated. This code integration is not production release approval.

For current coefficients, the exact source digest, test protocol and migration details, see [selected CVQA-to-MOS implementation](docs/dev/CVQA_MOS_FINAL_SELECTED_IMPLEMENTATION.md), [v7 transfer-separated calibration](docs/dev/CVQA_MOS_TRANSFER_SEPARATED_V7.md) and [Learned Metrics](docs/LEARNED_METRICS.md). Older reviewed-S1 and mixed-data policy reports remain historical evidence, not current runtime behavior.

## License

MIT © 2026 Pixop ApS. See [Licensing](docs/LICENSING.md) for third-party and trademark notices.
