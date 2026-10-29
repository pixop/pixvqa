<p align="center">
  <img src="docs/brand/pixop-vqa-logo-crop.png" alt="Pixop VQA" width="640">
</p>

# PixVQA — A Unified Video Quality Assessment Tool by Pixop

[![CPU CI](https://github.com/pixop/pixvqa/actions/workflows/qa.yml/badge.svg?branch=main)](https://github.com/pixop/pixvqa/actions/workflows/qa.yml) [![FFmpeg](https://img.shields.io/badge/FFmpeg-n8.1.3-blue)](https://ffmpeg.org/) [![libvmaf](https://img.shields.io/badge/libvmaf-74f930d-blue)](https://github.com/Netflix/vmaf) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PixVQA compares a reference sequence with a distorted sequence and prints text scores for the selected metrics. The default lists each frame and then a summary. There is no JSON mode. It reads raw planar YUV and, when built with FFmpeg, compressed video. Decoding uses the codec FFmpeg reports.

PSNR, XPSNR, SSIM, MS-SSIM, and VMAF compare stored sample codes. PU21-PSNR, PU21-SSIM, and PU21-MS-SSIM compare light after PQ or HLG is interpreted. All eight metrics have a CPU implementation. PSNR, SSIM, MS-SSIM, XPSNR, and the three PU21 metrics also have optional CUDA implementations. An optional CUDA build adds NVDEC for HEVC and AV1.

## Supported VQA Metrics

| Name | Description |
|---|---|
| `psnr` | Peak signal-to-noise ratio (PSNR) and MSE on stored sample codes |
| `xpsnr` | Extended PSNR (XPSNR). Uses a constant compressed frame rate, or `--fps` |
| `ssim` | Structural similarity (SSIM): 11×11 Gaussian, sigma 1.5, K1=0.01, K2=0.03 |
| `msssim` | Multi-scale structural similarity, the five-scale product (MS-SSIM) |
| `vmaf` | Netflix VMAF v1 through CPU libvmaf. Requires `--model` |
| `pu21-psnr` | PSNR on PU21-encoded display RGB |
| `pu21-ssim` | SSIM on PU21-encoded luminance |
| `pu21-msssim` | MS-SSIM on PU21-encoded luminance |
| [`--qcs`](docs/QCS.md) | Quality Consistency Score (QCS): temporal quality reliability descriptor derived from per-frame VQA scores |

PU21 is a perceptually uniform encoding of display light for HDR quality assessment. `pu21-xpsnr` and `pu21-vmaf` are not implemented.

**HDR / PU21:** To evaluate PQ or HLG video with PU21-PSNR, PU21-SSIM, or PU21-MS-SSIM, see the [PU21 usage guide](docs/PU21.md) for recommended transfer-function and display settings.

Select VQA metrics using `--metrics`; enable the QCS descriptor separately using `--qcs`.

## Recommended Metrics for SDR and HDR Video

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

**Legend:**

- ✅ Recommended
- ⚠️ Supported, but not generally recommended for HDR perceptual assessment
- ❌ Unsupported metric/model and format combination

**Notes:**
- Conventional PSNR, XPSNR, SSIM, and MS-SSIM operate on encoded sample values without HDR perceptual adaptation.
- SDR VMAF is intended for SDR; HDR VMAF v1.1.8 supports limited-range PQ/BT.2020, but not HLG.
- PU21 metrics support PQ and HLG, but not SDR in PixVQA.
- QCS can be applied to any supported metric using `--qcs`.

## Quality Consistency Score (QCS) - ICIP 2026

QCS is a metric-agnostic **temporal quality reliability descriptor** derived from per-frame VQA scores. It quantifies the proportion of viewing time during which video quality meets or exceeds specified thresholds, complementing conventional average quality metrics.

**Publication:** S. Sanz-Rodríguez and J. Frydensbjerg, "[Quality Consistency Score (QCS): A Survival-Based Reliability Descriptor for Video Quality Assessment](https://ieeexplore.ieee.org/document/11629873/)," *IEEE International Conference on Image Processing (ICIP)*, 2026.

For further details, see the [QCS documentation](docs/QCS.md) or the [QCS reference implementation](https://github.com/pixop/Quality-Consistency-Score).

## Installation

PixVQA supports **Linux x86-64**, including Ubuntu 22.04 and 24.04 (native Linux and WSL2). Native Windows builds have not been validated.

From the root of the cloned repository, run:

```bash
bash install_pixvqa.sh
```

The installer writes `bin/pixvqa` in that checkout.

**Optional**: To keep PixVQA-managed dependencies inside the repository instead of shared directories under `$HOME`, run:

```bash
PIXVQA_DEPS_ROOT="$PWD/deps" bash install_pixvqa.sh
```

The installer downloads the required dependencies, builds PixVQA, and asks whether to enable:

1. **Software decoding (dav1d/FFmpeg):** Recommended for compressed video support.
2. **NVIDIA GPU acceleration (CUDA/NVDEC):** Optional. Requires a compatible NVIDIA driver and CUDA toolkit 12.8 already installed.

After installation, verify:

```bash
./bin/pixvqa --version
python3 scripts/run_qa.py --profile smoke
```

**Troubleshooting and Manual Installation:**

If automatic installation fails, consult the [Manual Installation Guide](docs/INSTALL.md).

The guide provides step-by-step instructions for installing dependencies and compiling PixVQA manually. Follow the numbered sections in order, verifying that each stage completes successfully before continuing.

For clean-install testing and known validation limitations, see [Installation Validation](docs/INSTALL_VALIDATION.md).

**Note:** VMAF always runs on CPU libvmaf. Other supported metrics can use CUDA when NVIDIA acceleration is enabled.


## Command-line Samples

PixVQA supports raw YUV and compressed video. For compressed inputs, resolution and pixel format are detected automatically; raw YUV requires `-s` (resolution) and `-f` (pixel format). Reference and distorted inputs must have matching properties.

XPSNR requires `--fps` unless a reliable constant frame rate can be determined from the compressed inputs. GPU acceleration requires an NVIDIA-enabled build.

The examples use `pixvqa` to refer to `./bin/pixvqa` or an executable on `PATH`. The `samples/` directory contains short Big Buck Bunny excerpts.

**Compressed Reference and Distorted Video**

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_hevc.mp4 \
  -t samples/dist_hevc.mp4
```

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_av1.mp4 \
  -t samples/dist_av1.mp4
```

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_vvc.mp4 \
  -t samples/dist_vvc.mp4
```

**Uncompressed Reference and Compressed Distorted Video**

```bash
pixvqa --metrics psnr,ssim \
  -r reference.yuv --input raw -s 1920x1080 -f 420p:8 \
  -t distorted.mp4
```

**Uncompressed Reference and Distorted Video**

XPSNR requires `--fps` because raw YUV does not contain frame-rate metadata.

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim --fps 24 --threads 1 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

**VMAF**

VMAF requires an explicit model selection using `--model`. Bundled VMAF models are available in the `model/` directory.

```bash
pixvqa --metrics vmaf --model model/vmaf_v1.0.16_3d0h.json --threads 1 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

The bundled HDR VMAF v1.1.8 models assume limited-range 10-bit 4:2:0 PQ with
BT.2020 primaries and the BT.2020 non-constant-luminance matrix. PixVQA applies
that model configuration automatically; ensure both inputs really have those
characteristics. Contradictory metadata or HDR CLI options produce a warning and
do not convert the samples.

```bash
pixvqa --metrics vmaf \
  --model model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

**PU21 Metrics for HDR Video (PQ and HLG)**

PU21 metrics account for HDR transfer functions, unlike conventional metrics that operate on encoded sample values. For PQ and HLG, PixVQA defaults to BT.2020 primaries, BT.2020nc matrix, and limited range when these are unspecified. HLG defaults to a 1000 cd/m² display peak; PQ requires no display peak. See the [PU21 guide](docs/PU21.md) for details.

```bash
pixvqa --metrics pu21-psnr,pu21-ssim --hdr-transfer pq \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

```bash
pixvqa --metrics pu21-psnr,pu21-msssim --hdr-transfer hlg \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

**GPU Decoding (NVDEC)**

Decode the compressed HEVC 10-bit 4:2:0 video using NVIDIA NVDEC while reading the reference as raw YUV.

```bash
pixvqa --metrics psnr --decode gpu \
  -r reference10.yuv --input raw -s 1920x1080 -f 420p:10 \
  -t reference10.hevc.mp4
```

## Command-line Options

`pixvqa --help` lists the options. `pixvqa --help-hdr` lists the HDR options and defaults. QCS is documented in [docs/QCS.md](docs/QCS.md). PQ and HLG setup is documented in [docs/PU21.md](docs/PU21.md).

| Option | Purpose |
|---|---|
| `-r`, `-t` | Reference path and distorted path |
| `--metrics` | Comma-separated metric names |
| `--decode` | `auto`, `cpu`, or `gpu`. Default `auto` |
| `--metric-device` | `auto`, `cpu`, or `gpu`. Default `auto` |
| `--threads` | `auto` or a positive integer. Default `auto` |
| `-s WxH` | Luma size. Required for raw YUV. Checked when set on compressed video |
| `-f` | `420p`, `422p`, or `444p`, with `:8` or `:10`. Required for raw YUV |
| `--input` | `auto`, `raw`, or `compressed`. Default `auto`. `raw` does not probe |
| `--fps` | `N` or `N/D`. Required for `xpsnr` when no constant compressed rate is available. Decimals truncate toward zero |
| `--model` | VMAF model JSON. No default |
| `--hdr-transfer` | `pq`, `hlg`, `sdr`, `linear`, or `auto` |
| `-v` | `1` prints each frame and the summary. `0` prints the summary. Default `1` |
| `--qcs` | Enable QCS for selected supported per-frame metrics |
| `--qcs-range` | `metric=lower:upper`, comma-separated and repeatable, with lower < upper |


## Supported Video Formats

PixVQA supports raw YUV and compressed video through FFmpeg.

| Input format | Support |
|---|---|
| Raw YUV | 8-bit and 10-bit planar 4:2:0, 4:2:2, and 4:4:4 |
| Compressed video | Software decoding through FFmpeg (e.g., H.264, HEVC, AV1, VVC, VP9, MPEG-2, ProRes), subject to decoder availability and supported pixel formats |
| NVIDIA NVDEC | Hardware decoding of HEVC 10-bit 4:2:0 and AV1 8-bit/10-bit 4:2:0 |

Compressed video support depends on the available FFmpeg decoders. PixVQA accepts planar 8-bit and 10-bit YUV in 4:2:0, 4:2:2, and 4:4:4 formats; unsupported pixel formats are rejected rather than converted. 4:2:0 requires even width and height. Samples are planar, little-endian, with the low bits occupied.

For AV1 software decoding, FFmpeg must include libdav1d or libaom-av1. NVIDIA NVDEC is limited to HEVC and AV1; `--decode gpu` does not fall back to software decoding. `--decode auto` prefers a linked software decoder. AV1 uses NVDEC only when no software AV1 decoder is linked, and VVC stays on software. There is no codec or container allowlist. A FIFO stays raw.

The two inputs are decoded independently and paired in presentation order. Timestamps do not drop, repeat, or resynchronize frames. A length or geometry mismatch is an error.

## CPU and GPU Support

| Metric | CPU 8-bit | CPU 10-bit | CUDA 8-bit | CUDA 10-bit |
|---|:---:|:---:|:---:|:---:|
| PSNR | ✅ | ✅ | ✅ | ✅ |
| XPSNR | ✅ | ✅ | ✅ | ✅ |
| SSIM | ✅ | ✅ | ✅ | ✅ |
| MS-SSIM | ✅ | ✅ | ✅ | ✅ |
| VMAF (SDR models) | ✅ | ✅ | — | — |
| VMAF (HDR models) | ❌ | ✅ | — | — |
| PU21-PSNR | ✅ | ✅ | ✅ | ✅ |
| PU21-SSIM | ✅ | ✅ | ✅ | ✅ |
| PU21-MS-SSIM | ✅ | ✅ | ✅ | ✅ |

Bundled HDR VMAF v1.1.8 assumes limited-range 10-bit 4:2:0 PQ/BT.2020 with the
BT.2020 non-constant-luminance matrix; users must ensure the input matches.
PU21 requires PQ or HLG. A dash means that metric has no CUDA implementation.
VMAF still runs on CPU libvmaf when a GPU is requested. CUDA PSNR, SSIM, and
MS-SSIM use FP64.

NVDEC accepts HEVC Main 10 4:2:0 and AV1 8-bit or 10-bit 4:2:0. `--decode gpu` rejects other codecs, including VVC. `--metric-device gpu` has no CPU substitute for a metric that cannot run on the GPU. `--metric-device auto` uses the CPU except one profile: 10-bit, at least 3840×2160, GPU decoding, no VMAF, on a CUDA build. `--threads auto` uses half of the processors visible to this process, at most 8 and at least 1, for metric workers. Software decoding keeps its own pool of 8 threads on each compressed input.

**Limitations:**
- HDR transfer characteristics (PQ/HLG) come from the stream or from `--hdr-transfer`. They are not inferred from bit depth, BT.2020 primaries, or the filename.
- A variable frame rate is not treated as constant. XPSNR then requires `--fps`.
- Unsupported pixel formats are rejected rather than automatically converted.

Option names and defaults are in `pixvqa --help` and `pixvqa --help-hdr`. QCS is in [docs/QCS.md](docs/QCS.md). PQ and HLG setup is in [docs/PU21.md](docs/PU21.md).

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

The logo at `docs/brand/pixop-vqa-logo-crop.png` is included with the repository. The MIT license does not grant unrestricted trademark or logo use. See [docs/brand/README.md](docs/brand/README.md).
