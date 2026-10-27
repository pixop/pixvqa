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
| `psnr` | Peak signal-to-noise ratio, and MSE, on stored sample codes |
| `xpsnr` | XPSNR. Requires `--fps` |
| `ssim` | Single-scale structural similarity |
| `msssim` | One five-scale product MS-SSIM. `ssim` does not select it |
| `vmaf` | Netflix VMAF v1 through libvmaf, on the CPU. Requires `--model` |
| `pu21-psnr` | PSNR on PU21-encoded display RGB |
| `pu21-ssim` | SSIM on PU21-encoded luminance |
| `pu21-msssim` | MS-SSIM on PU21-encoded luminance |

PU21 is a perceptually uniform encoding of display light, so those three metrics can take HDR as input. `pu21-xpsnr` and `pu21-vmaf` are not supported. Pass names to `--metrics`. There is no second MS-SSIM metric.

## Recommended metrics by video format

✅ supported and recommended. ⚠️ supported, but not a perceptual HDR measure. ❌ rejected by PixVQA.

| Metric | SDR (BT.709) | PQ / BT.2020 (HDR10) | HLG / BT.2020 |
|---|:---:|:---:|:---:|
| PSNR | ✅ | ⚠️ | ⚠️ |
| XPSNR | ✅ | ⚠️ | ⚠️ |
| SSIM | ✅ | ⚠️ | ⚠️ |
| MS-SSIM | ✅ | ⚠️ | ⚠️ |
| VMAF, SDR models | ✅ | ❌ (1) | ❌ (1) |
| VMAF, HDR v1.1.8 models | ❌ | ✅ (2) | ❌ |
| PU21-PSNR | ❌ (3) | ✅ (4) | ✅ (4) |
| PU21-SSIM | ❌ (3) | ✅ (4) | ✅ (4) |
| PU21-MS-SSIM | ❌ (3) | ✅ (4) | ✅ (4) |

QCS is not a separate column entry. It is printed for every metric that runs, so it is available exactly where the metric's cell is ✅ or ⚠️, and it inherits that metric's suitability.

PSNR, XPSNR, SSIM, and MS-SSIM compare the stored Y′CbCr sample codes. The transfer does not change their result: the same codes score the same under SDR, PQ, or HLG. SDR gamma coding is close enough to perceptually uniform for these metrics. PQ codes are a perceptual encoding, so these scores are a usable engineering comparison, but PU21 predicts HDR quality better than PQ encoding does. HLG codes are scene-referred and are not tied to display luminance. PixVQA does not run them through a display model.

1. Rejected when the effective transfer of either input is PQ or HLG, from a stream tag or `--hdr-transfer`. PixVQA does not infer PQ or HLG from 10-bit input, BT.2020, or a filename. An untagged raw input is therefore scored as SDR. Netflix states that an SDR model on HDR pictures is not meaningful.
2. Only limited-range 10-bit 4:2:0. Both inputs need PQ, BT.2020 primaries, the bt2020nc matrix, and limited range, each from a stream tag or an explicit option. Assumed values are not used. Full range, 8-bit, 4:4:4, and BT.709 primaries are rejected. HDR10 static metadata (SMPTE ST 2086, MaxCLL, MaxFALL) is not read, and no display mapping is applied. `hdr_vmaf_v1.1.8_3d0h_2160` and `hdr_vmaf_v1.1.8_hfr_3d0h_2160` can score up to 110.
3. PU21 requires a PQ or HLG transfer. SDR, `linear`, and an unspecified transfer are rejected. The PU21 paper describes an SDR path through a display model, but PixVQA does not implement it. Use the conventional metrics for SDR.
4. PU21 needs display luminance. PQ is decoded to absolute luminance up to 10 000 cd/m² and is clipped only when `--hdr-peak-nits` is set. PixVQA does not tone-map to a display. HLG uses the BT.2100 inverse OETF and OOTF with a 1000 cd/m² evaluation peak unless `--hdr-peak-nits` sets another. PU21-PSNR uses display RGB. PU21-SSIM and PU21-MS-SSIM use luminance only. PQ and HLG are also accepted with BT.709 primaries.

The HDR10 column describes the PQ / BT.2020 signal. HDR10 also implies 10-bit samples and static metadata. PixVQA evaluates the signal and ignores that metadata.

## Quality Consistency Score (QCS)

Quality Consistency Score (QCS) is a metric-agnostic descriptor of temporal quality reliability. It measures the proportion of viewing time during which video quality meets or exceeds thresholds within a specified quality range, summarized as the normalized area under the survival curve of frame-level quality scores. QCS complements conventional average quality metrics by revealing the prevalence and severity of quality degradations that averages may obscure.

QCS characterizes the distribution of frame-level quality scores over time, not their temporal ordering. Different temporal arrangements of the same quality scores therefore produce the same QCS.

Each frame is one sample. That matches viewing time when every frame has the same duration. QCS does not read timestamps. A score at or below the lower limit contributes 0%. A score at or above the upper limit contributes 100%, and so does +inf. -inf contributes 0%. NaN is omitted. `--qcs` prints one QCS value for every metric named in `--metrics`, on CPU or GPU. The frame score is the value that metric already produced. QCS does not clip or rescale it first. HDR VMAF, and the SDR models whose clip is 110, can score above 100. With the default upper bound of 100 those scores contribute 100%. Raising the upper bound, for example `--qcs-range vmaf=70:120`, keeps the difference.

Default ranges:

| Metric | Default | Why this band |
|---|---|---|
| `psnr`, `xpsnr` | `[30, 50]` | Decibel scores. Unchanged. |
| `ssim`, `msssim` | `[0.8, 1.0]` | Similarity scores. Unchanged. |
| `vmaf` | `[70, 100]` | The QCS paper's VMAF interval. One default for every SDR and HDR model. |
| `pu21-psnr` | `[30, 50]` | PSNR on PU21 codes with peak 256, about 100 cd/m². That is an 8-bit-style decibel scale, not the VMAF scale. |
| `pu21-ssim`, `pu21-msssim` | `[0.8, 1.0]` | SSIM and MS-SSIM on PU21 luminance with the same peak. The score is still a similarity in about `[0, 1]`. |

`--qcs-range psnr=40:55` sets one range. The name is the `--metrics` name, and the bounds must satisfy lower < upper. Identical PSNR is printed `999.99`, identical XPSNR is printed `inf`, and identical PU21-PSNR is printed `999.99`. All three still count in QCS.

```bash
pixvqa --metrics psnr,ssim,vmaf,pu21-psnr --qcs \
  --model model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json \
  --qcs-range vmaf=60:90 --qcs-range pu21-psnr=30:50 \
  --hdr-transfer pq --hdr-primaries bt2020 --hdr-matrix bt2020nc --hdr-range limited \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

```bash
pixvqa --metrics psnr --qcs --qcs-range psnr=30:50 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

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
pixvqa --metrics vmaf --qcs --model model/vmaf_v1.0.16_3d0h.json --threads 1 \
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
pixvqa --metrics pu21-psnr,pu21-ssim --qcs --hdr-transfer pq \
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
| `--qcs` | Enable QCS for every selected metric |
| `--qcs-range` | `metric=lower:upper`, with lower < upper. Defaults are listed under QCS |

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
- PU21. R. K. Mantiuk and M. Azimi, “PU21: A Novel Perceptually Uniform Encoding for Adapting Existing Quality Metrics for HDR,” Picture Coding Symposium, 2021. [DOI 10.1109/PCS50896.2021.9477471](https://doi.org/10.1109/PCS50896.2021.9477471). Reference implementation: [gfxdisp/pu21](https://github.com/gfxdisp/pu21). Notice: [licenses/PU21_BSD.txt](licenses/PU21_BSD.txt).
- HDR VMAF. Netflix, “VMAF v1 models: HDR models,” [resource/doc/models_v1.md](https://github.com/Netflix/vmaf/blob/master/resource/doc/models_v1.md#hdr-models). A. Mavlankar, Z. Li, L. Krasula, and C. Bampis, “All of Netflix’s HDR video streaming is now dynamically optimized,” Netflix Technology Blog, 2023. [Blog](https://netflixtechblog.com/all-of-netflixs-hdr-video-streaming-is-now-dynamically-optimized-e9e0cb15f2ba).
- HDR metric benchmark. P. Hanhart, M. V. Bernardo, M. Pereira, A. M. G. Pinheiro, and T. Ebrahimi, “Benchmarking of objective quality metrics for HDR image quality assessment,” EURASIP Journal on Image and Video Processing, 2015. [DOI 10.1186/s13640-015-0091-4](https://doi.org/10.1186/s13640-015-0091-4).
- PQ and HLG. ITU-R BT.2100, “Image parameter values for high dynamic range television for use in production and international programme exchange.” [ITU-R BT.2100](https://www.itu.int/rec/R-REC-BT.2100).

## License

PixVQA is licensed under the [MIT License](LICENSE) (SPDX `MIT`). Copyright (c) 2026 Pixop ApS. [docs/LICENSING.md](docs/LICENSING.md) explains third-party and model licenses. Dependency notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The logo at `docs/brand/pixop-vqa-logo.png` is included with the repository. The MIT license does not grant unrestricted trademark or logo use. See [docs/brand/README.md](docs/brand/README.md).
