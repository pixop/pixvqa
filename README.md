<p align="center">
  <img src="docs/brand/pixop-vqa-logo.png" alt="Pixop VQA" width="640">
</p>

# PixVQA

[![CPU CI](https://github.com/pixop/pixvqa/actions/workflows/qa.yml/badge.svg?branch=main)](https://github.com/pixop/pixvqa/actions/workflows/qa.yml) [![FFmpeg](https://img.shields.io/badge/FFmpeg-n8.1.3-blue)](https://ffmpeg.org/) [![libvmaf](https://img.shields.io/badge/libvmaf-74f930d-blue)](https://github.com/Netflix/vmaf) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Video Quality Assessment

PixVQA compares a reference sequence with a distorted sequence and prints text scores for the selected metrics. The default lists each frame and then a summary. There is no JSON mode. It reads raw planar YUV and, when built with FFmpeg, compressed video. Decoding uses the codec FFmpeg reports.

PSNR, XPSNR, SSIM, MS-SSIM, and VMAF compare stored sample codes. PU21-PSNR, PU21-SSIM, and PU21-MS-SSIM compare light after PQ or HLG is interpreted. All eight metrics have a CPU implementation. PSNR, SSIM, MS-SSIM, and XPSNR also have optional CUDA implementations. An optional CUDA build adds NVDEC for HEVC and AV1.

## Supported metrics

| Name | Description |
|---|---|
| `psnr` | Peak signal-to-noise ratio (PSNR) and MSE on stored sample codes |
| `xpsnr` | Extended PSNR (XPSNR). Requires `--fps` |
| `ssim` | Structural similarity index (SSIM) |
| `msssim` | Multi-scale structural similarity (MS-SSIM) |
| `vmaf` | Netflix VMAF v1 through CPU libvmaf. Requires `--model` |
| `pu21-psnr` | PSNR on PU21-encoded display RGB |
| `pu21-ssim` | SSIM on PU21-encoded luminance |
| `pu21-msssim` | MS-SSIM on PU21-encoded luminance |

PU21 is a perceptually uniform encoding of display light for HDR quality assessment. `pu21-xpsnr` and `pu21-vmaf` are not implemented.

Select metrics using `--metrics`. QCS can additionally be enabled using `--qcs`.

### Recommended metrics by video format

| Metric | SDR (BT.709) | PQ / BT.2020 (HDR10) | HLG (BT.2020) |
|---|:---:|:---:|:---:|
| PSNR | ✅ | ⚠️ | ⚠️ |
| XPSNR | ✅ | ⚠️ | ⚠️ |
| SSIM | ✅ | ⚠️ | ⚠️ |
| MS-SSIM | ✅ | ⚠️ | ⚠️ |
| VMAF (SDR models) | ✅ | ❌ | ❌ |
| VMAF (HDR v1.1.8 models) | ❌ | ✅ | ❌ |
| PU21-PSNR | ❌ | ✅ | ✅ |
| PU21-SSIM | ❌ | ✅ | ✅ |
| PU21-MS-SSIM | ❌ | ✅ | ✅ |

**Legend:**

- ✅ Recommended
- ⚠️ Supported, but not generally recommended for HDR perceptual assessment
- ❌ Unsupported metric/model and format combination

**Notes:**
- Conventional PSNR, XPSNR, SSIM, and MS-SSIM operate on encoded sample values without HDR perceptual adaptation.
- SDR VMAF is intended for SDR; HDR VMAF v1.1.8 supports limited-range PQ/BT.2020, but not HLG.
- PU21 metrics support PQ and HLG, but not SDR in PixVQA.
- QCS can be applied to any supported metric using `--qcs`.

## Quality Consistency Score (QCS)

**Publication:** S. Sanz-Rodríguez and J. Frydensbjerg, "[Quality Consistency Score (QCS): A Survival-Based Reliability Descriptor for Video Quality Assessment](https://ieeexplore.ieee.org/document/11629873/)," *IEEE International Conference on Image Processing (ICIP)*, 2026.

Quality Consistency Score (QCS) is a metric-agnostic descriptor of **temporal quality reliability**. It measures the proportion of viewing time during which video quality meets or exceeds specified thresholds, revealing degradations that may be hidden by average quality metrics or conventional rate-distortion (RD/BD-rate) analysis.

### Definition

Let $Q(t)$ represent the frame-level quality scores and $S(x)=P(Q\ge x)$ their survival function. QCS is the normalized area under the survival curve over a quality interval $[a,b]$:
$$
\mathrm{QCS}_{[a,b]} = \frac{1}{b-a}\int_a^b S(x)\,dx
$$
QCS ranges from 0% to 100% when expressed as a percentage. Scores at or below $a$ contribute 0%, scores at or above $b$ contribute 100%, and intermediate scores contribute proportionally. NaN values are excluded. +inf contributes 100% and -inf contributes 0%. A VMAF score above 100 is kept. With the default upper bound of 100 it contributes 100%.

Each frame contributes equally (constant-frame-rate assumption). QCS describes the **distribution of quality scores**, not their temporal ordering.

### Default QCS intervals

| Metric | Interval |
|---|---|
| PSNR / XPSNR | [30, 50] dB |
| SSIM / MS-SSIM | [0.8, 1.0] |
| VMAF (SDR / HDR) | [70, 100] |
| PU21-PSNR | [30, 50] dB |
| PU21-SSIM / PU21-MS-SSIM | [0.8, 1.0] |

VMAF uses one interval for every SDR and HDR model. PU21-PSNR uses the PSNR interval because its peak is 256, about 100 cd/m², which is an 8-bit-style decibel scale. PU21-SSIM and PU21-MS-SSIM use the SSIM interval because they are similarities on that same peak.

### Usage

Enable QCS with `--qcs` and optionally override the default interval using `--qcs-range`.
```bash
pixvqa --metrics psnr,vmaf --model model/vmaf_v1.0.16_3d0h.json \
  --qcs --qcs-range psnr=40:55 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

QCS is computed independently for each selected metric. The name in `--qcs-range` is the `--metrics` name, and the bounds must satisfy lower < upper. Identical PSNR is printed `999.99`, identical XPSNR is printed `inf`, and identical PU21-PSNR is printed `999.99`. All three still count in QCS.

## Quick start

The commands use 1920×1080. The same options apply at 3840×2160 with `-s 3840x2160`. Both inputs must use the same size. In the examples below, `pixvqa` denotes `./bin/pixvqa` after installation, or the same executable on `PATH`. The GPU command requires an installation built with NVIDIA support.

Compressed video, software decoder:
```bash
pixvqa --metrics psnr --decode cpu \
  -r reference.mp4 --input compressed -s 1920x1080 -f 420p:8 \
  -t distorted.mp4 --input compressed -s 1920x1080 -f 420p:8
```

PSNR, XPSNR, SSIM, and MS-SSIM together. XPSNR needs a frame rate:
```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim --fps 24 --threads 1 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

VMAF has no default model. Picture size does not choose a model. An SDR model is refused when the effective transfer is PQ or HLG.
```bash
pixvqa --metrics vmaf --model model/vmaf_v1.0.16_3d0h.json --threads 1 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

HDR VMAF v1.1.8 requires the transfer, primaries, matrix, and range on both inputs. This is the validated limited-range PQ BT.2020 case. The score keeps that model's clip, including values above 100.
```bash
pixvqa --metrics vmaf --model model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json --threads 1 \
  --hdr-transfer pq --hdr-primaries bt2020 --hdr-matrix bt2020nc --hdr-range limited \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

Raw planar YUV. `--input raw` does not probe the file:
```bash
pixvqa --metrics psnr --input raw \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

PQ, then HLG. PU21 uses the transfer. A stored-sample metric such as `psnr` does not.
```bash
pixvqa --metrics pu21-psnr,pu21-ssim --hdr-transfer pq \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```
```bash
pixvqa --metrics pu21-psnr,pu21-msssim --hdr-transfer hlg --hdr-peak-nits 1000 \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

HEVC 10-bit 4:2:0 on NVDEC:
```bash
pixvqa --metrics psnr --decode gpu \
  -r reference10.yuv --input raw -s 1920x1080 -f 420p:10 \
  -t reference10.hevc.mp4 --input compressed -s 1920x1080 -f 420p:10
```

## Installation

PixVQA supports **Linux x86-64**, including Ubuntu 22.04 and 24.04 (native Linux and WSL2). Native Windows builds have not been validated.

Two installation methods are available.

### Option 1 — Automatic installation (recommended)

The guided installer downloads and builds the required dependencies, compiles the feature set you select, and installs that program as one executable.

From the root of the cloned repository, run:

    bash install_pixvqa.sh

    ./bin/pixvqa --version

The installer asks two questions:

1. **Install dav1d for software AV1 decoding?** `[Y/n]` (default: Yes).

2. **Build NVIDIA GPU support (CUDA metrics + NVDEC)?** `[y/N]` (default: No).

**Recommended:** Select Yes for dav1d to enable software decoding of compressed video, and Yes for NVIDIA if you want GPU acceleration.

NVIDIA acceleration requires a compatible NVIDIA driver and **CUDA toolkit 12.8** already installed. The installer does not install the NVIDIA driver or CUDA toolkit. `./bin/pixvqa` finds its private libraries through the executable's RPATH, so a new terminal does not need `LD_LIBRARY_PATH`.

`--version` reports the selected features. A raw-YUV build prints `build: cpu` and `decode: none`. Software decoding prints `decode: software`. NVIDIA support prints `build: cuda` and `decode: software nvdec`.

Files created under `build/` are compilation intermediates. They are removed after installation and smoke QA. Set `PIXVQA_KEEP_BUILD=1` to keep them.

**Optional — Isolated dependency installation**

By default, the installer uses shared dependency directories under `$HOME`. To keep all PixVQA-managed third-party dependencies inside the repository instead, run:

    PIXVQA_DEPS_ROOT="$PWD/deps" bash install_pixvqa.sh

This does not isolate system packages, the NVIDIA driver, or the CUDA toolkit.

The guided installer is currently experimental. For additional requirements and troubleshooting, see [Installation Guide](docs/INSTALL.md).

### Option 2 — Manual installation (advanced)

For custom builds or manual control over dependencies and compilation, follow the step-by-step instructions in:

**[Manual Installation Guide](docs/INSTALL.md)**

The guide explains how to:

- Build the required dependencies, including FFmpeg, libvmaf, and zimg.

- Compile the CPU-only version for raw YUV input.

- Enable software decoding for compressed video, including AV1 and VVC.

- Enable NVIDIA CUDA acceleration and NVDEC hardware decoding.

- Install PixVQA into a custom directory.

- Verify the installation and troubleshoot common issues.

**VMAF:** SDR and HDR VMAF always execute on CPU libvmaf. When GPU execution is requested, PixVQA automatically falls back to CPU for VMAF and displays a warning. PSNR, SSIM, MS-SSIM, and XPSNR retain CUDA acceleration when supported.

For independent clean-install testing of both methods, see [Installation Validation](docs/INSTALL_VALIDATION.md).

## Command-line options

`pixvqa --help` is the full option list. `pixvqa --help-hdr` is the full HDR list, including defaults and the PU21 citation.

| Option | Purpose |
|---|---|
| `-r`, `-t` | Reference path and distorted path |
| `--metrics` | Comma-separated metric names |
| `--decode` | `auto`, `cpu`, or `gpu`. Default `auto` |
| `--metric-device` | `auto`, `cpu`, or `gpu`. Default `auto` |
| `--threads` | `auto` or a positive integer. Default `auto` |
| `-s WxH` | Luma size. Give it again after `-t` |
| `-f` | `420p`, `422p`, or `444p`, with `:8` or `:10` |
| `--input` | `auto`, `raw`, or `compressed` |
| `--fps` | `N` or `N/D`. Required for `xpsnr` |
| `--model` | VMAF model JSON. No default |
| `--hdr-transfer` | `pq`, `hlg`, `sdr`, `linear`, or `auto` |
| `-v` | `1` prints each frame and the summary. `0` prints the summary. Default `1` |
| `--qcs` | Enable QCS for selected supported per-frame metrics |
| `--qcs-range` | `metric=lower:upper`, with lower < upper |

## Formats and limitations

FFmpeg chooses the demuxer and decoder. There is no container or codec allowlist. A picture is scored only when it is already planar 8-bit or 10-bit 4:2:0, 4:2:2, or 4:4:4. Other layouts are rejected and are not converted. The tested set is smaller than the set FFmpeg can open.

| Path | Tested behavior |
|---|---|
| Raw YUV | Planar 8-bit and 10-bit 4:2:0, 4:2:2, and 4:4:4 |
| AV1 | libdav1d, or libaom-av1 if libdav1d is absent. The native `av1` decoder is not used. NVDEC for 8-bit and 10-bit 4:2:0 |
| VVC | FFmpeg native decoder, short 10-bit 4:2:0 clips. No NVDEC |
| HEVC | Software decoding, and NVDEC for 10-bit 4:2:0 |

`--decode` accepts `auto`, `cpu`, or `gpu`. `--decode software` and `--decode nvdec` are rejected, and `cpu` and `gpu` do not substitute for each other. GPU PSNR and GPU SSIM are 10-bit only. GPU XPSNR and GPU MS-SSIM accept 8-bit and 10-bit. VMAF has no GPU path. `--metric-device gpu` with VMAF, alone or with other metrics, runs VMAF on CPU libvmaf and prints `pixvqa: warning: VMAF CUDA computation is unavailable; using CPU libvmaf.` XPSNR, PSNR, SSIM, and MS-SSIM still use the GPU when they have an implementation. The execution summary prints `requested device` and each metric's actual device. VMAF is `cpu libvmaf`, not a GPU backend. A GPU metric that cannot run is an error, with no CPU substitute. `--metric-device auto` uses GPU scoring for 10-bit 3840×2160 or larger NVDEC input when VMAF is not selected. PQ and HLG are not inferred from bit depth, a filename, or BT.2020 primaries. `--hdr-peak-nits` is an evaluation-display value, not MaxCLL. Identical samples print PSNR `999.99`. `--psnr`, `--ssim`, `--msssim`, `--xpsnr`, `--vmaf`, `--xpsnr-backend`, `--vmaf-backend`, `--wang-msssim`, and `--kernel` are rejected. An unknown option exits 1 with `PARSE ERROR`. A flag this version explicitly removed exits 2. Metric selection is `--metrics`.

## References

- PSNR. Peak-signal form of mean squared error on the stored sample codes.

- XPSNR. C. R. Helmrich, M. Siekmann, S. Becker, S. Bosse, D. Marpe, and T. Wiegand, “XPSNR: A Low-Complexity Extension of the Perceptually Weighted Peak Signal-to-Noise Ratio for High-Resolution Video Quality Assessment,” ICASSP 2020. [DOI 10.1109/ICASSP40776.2020.9054089](https://doi.org/10.1109/ICASSP40776.2020.9054089).

- SSIM. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, April 2004. [DOI 10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861).

- MS-SSIM. Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiscale Structural Similarity for Image Quality Assessment,” Asilomar Conference on Signals, Systems and Computers, 2003. [DOI 10.1109/ACSSC.2003.1292216](https://doi.org/10.1109/ACSSC.2003.1292216).

- VMAF. Z. Li, A. Aaron, I. Katsavounidis, A. Moorthy, and M. Manohara, “Toward a Practical Perceptual Video Quality Metric,” Netflix Technology Blog, 2016. [Blog](https://netflixtechblog.com/toward-a-practical-perceptual-video-quality-metric-653f208b9652). Models and attribution: [model/README.md](model/README.md), [Netflix/vmaf](https://github.com/Netflix/vmaf) commit `74f930d` (not a release tag; the previous tag is v3.2.1), [third_party/netflix-vmaf/NOTICE](third_party/netflix-vmaf/NOTICE).

- QCS. S. Sanz-Rodríguez and J. Frydensbjerg, "[Quality Consistency Score (QCS): A Survival-Based Reliability Descriptor for Video Quality Assessment](https://ieeexplore.ieee.org/document/11629873/)," *IEEE ICIP*, 2026.
- PU21. R. K. Mantiuk and M. Azimi, “PU21: A Novel Perceptually Uniform Encoding for Adapting Existing Quality Metrics for HDR,” Picture Coding Symposium, 2021. [DOI 10.1109/PCS50896.2021.9477471](https://doi.org/10.1109/PCS50896.2021.9477471). Reference implementation: [gfxdisp/pu21](https://github.com/gfxdisp/pu21). Notice: [licenses/PU21_BSD.txt](licenses/PU21_BSD.txt).
- HDR VMAF. Netflix, “VMAF v1 models: HDR models,” [resource/doc/models_v1.md](https://github.com/Netflix/vmaf/blob/master/resource/doc/models_v1.md#hdr-models). A. Mavlankar, Z. Li, L. Krasula, and C. Bampis, “All of Netflix’s HDR video streaming is now dynamically optimized,” Netflix Technology Blog, 2023. [Blog](https://netflixtechblog.com/all-of-netflixs-hdr-video-streaming-is-now-dynamically-optimized-e9e0cb15f2ba).
- HDR metric benchmark. P. Hanhart, M. V. Bernardo, M. Pereira, A. M. G. Pinheiro, and T. Ebrahimi, “Benchmarking of objective quality metrics for HDR image quality assessment,” EURASIP Journal on Image and Video Processing, 2015. [DOI 10.1186/s13640-015-0091-4](https://doi.org/10.1186/s13640-015-0091-4).
- PQ and HLG. ITU-R BT.2100, “Image parameter values for high dynamic range television for use in production and international programme exchange.” [ITU-R BT.2100](https://www.itu.int/rec/R-REC-BT.2100).

## License

PixVQA is licensed under the [MIT License](LICENSE) (SPDX `MIT`). Copyright (c) 2026 Pixop ApS. [docs/LICENSING.md](docs/LICENSING.md) explains third-party and model licenses. Dependency notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The logo at `docs/brand/pixop-vqa-logo.png` is included with the repository. The MIT license does not grant unrestricted trademark or logo use. See [docs/brand/README.md](docs/brand/README.md).
