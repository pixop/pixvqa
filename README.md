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

## Quality Consistency Score (QCS)

Quality Consistency Score (QCS) is a metric-agnostic descriptor of temporal quality reliability. It measures the proportion of viewing time during which video quality meets or exceeds thresholds within a specified quality range, summarized as the normalized area under the survival curve of frame-level quality scores. QCS complements conventional average quality metrics by revealing the prevalence and severity of quality degradations that averages may obscure.

QCS characterizes the distribution of frame-level quality scores over time, not their temporal ordering. Different temporal arrangements of the same quality scores therefore produce the same QCS.

Each frame is one sample. That matches viewing time when every frame has the same duration. QCS does not read timestamps. A score at or below the lower limit contributes 0%. A score at or above the upper limit contributes 100%, and so does +inf. -inf contributes 0%. NaN is omitted. `--qcs` accepts `psnr`, `xpsnr`, `ssim`, and `msssim`. It does not accept `vmaf` or the PU21 metrics. Defaults are `[30,50]` for PSNR and XPSNR, and `[0.8,1.0]` for SSIM and MS-SSIM. `--qcs-range psnr=40:55` sets one range. The name is the base metric, and the bounds must satisfy lower < upper. Identical PSNR is printed `999.99` and identical XPSNR is printed `inf`. Both still count in QCS.

```bash
pixvqa --metrics psnr --qcs --qcs-range psnr=30:50 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

## Quick start

The commands use 1920×1080. The same options apply at 3840×2160 with `-s 3840x2160`. Both inputs must use the same size. `pixvqa` is `build/swdec/pixvqa`, or the installed binary. The GPU command needs `build/full/pixvqa`.

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

The tested environment is Linux on x86-64, including WSL2. Native Windows has not been validated. Builds do not require a CUDA toolkit unless you want NVDEC or GPU metrics. Full commands, FFmpeg flags, and troubleshooting are in [docs/INSTALL.md](docs/INSTALL.md).

Raw YUV only. This binary does not decode compressed video:

```bash
bash build.sh
```

CPU decoding, including AV1 through libdav1d and VVC through FFmpeg. Leave `GCC13` and `CUDA_HOME` unset:

```bash
export FFMPEG_PREFIX=/path/to/ffmpeg
export DAV1D_LIB=/path/to/libdav1d
export GCC_ENV=/path/to/gcc
export LIBVMAF_PREFIX=/path/to/libvmaf
bash build_decode.sh
```

The executable is `build/swdec/pixvqa`. Optional NVDEC and GPU metrics produce `build/full/pixvqa`.

```bash
PREFIX=/opt/pixvqa PIXVQA_BIN=build/swdec/pixvqa bash scripts/install.sh
```

Models are copied to `$PREFIX/share/pixvqa/model`. Pass one file to `--model`. Nothing selects a model for you.

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
| `--qcs` | Optional QCS for `psnr`, `xpsnr`, `ssim`, or `msssim` |
| `--qcs-range` | `metric=lower:upper`, with lower < upper |

## Formats and limitations

FFmpeg chooses the demuxer and decoder. There is no container or codec allowlist. A picture is scored only when it is already planar 8-bit or 10-bit 4:2:0, 4:2:2, or 4:4:4. Other layouts are rejected and are not converted. The tested set is smaller than the set FFmpeg can open.

| Path | Tested behavior |
|---|---|
| Raw YUV | Planar 8-bit and 10-bit 4:2:0, 4:2:2, and 4:4:4 |
| AV1 | libdav1d, or libaom-av1 if libdav1d is absent. The native `av1` decoder is not used. NVDEC for 8-bit and 10-bit 4:2:0 |
| VVC | FFmpeg native decoder, short 10-bit 4:2:0 clips. No NVDEC |
| HEVC | Software decoding, and NVDEC for 10-bit 4:2:0 |

`--decode` accepts `auto`, `cpu`, or `gpu`. `--decode software` and `--decode nvdec` are rejected, and `cpu` and `gpu` do not substitute for each other. GPU PSNR and GPU SSIM are 10-bit only. GPU XPSNR and GPU MS-SSIM accept 8-bit and 10-bit. VMAF has no GPU path: `--metric-device gpu` with only VMAF is an error, and `--metrics xpsnr,vmaf --metric-device gpu` runs XPSNR on the GPU and leaves VMAF on the CPU. A GPU metric that cannot run is an error, with no CPU substitute. `--metric-device auto` uses GPU scoring for 10-bit 3840×2160 or larger NVDEC input when VMAF is not selected. PQ and HLG are not inferred from bit depth, a filename, or BT.2020 primaries. `--hdr-peak-nits` is an evaluation-display value, not MaxCLL. Identical samples print PSNR `999.99`. `--psnr`, `--ssim`, `--msssim`, `--xpsnr`, `--vmaf`, `--xpsnr-backend`, `--vmaf-backend`, `--wang-msssim`, and `--kernel` are rejected. An unknown option exits 1 with `PARSE ERROR`. A flag this version explicitly removed exits 2. Metric selection is `--metrics`.

## References

- PSNR. Peak-signal form of mean squared error on the stored sample codes.
- XPSNR. C. R. Helmrich, M. Siekmann, S. Becker, S. Bosse, D. Marpe, and T. Wiegand, “XPSNR: A Low-Complexity Extension of the Perceptually Weighted Peak Signal-to-Noise Ratio for High-Resolution Video Quality Assessment,” ICASSP 2020. [DOI 10.1109/ICASSP40776.2020.9054089](https://doi.org/10.1109/ICASSP40776.2020.9054089).
- SSIM. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, April 2004. [DOI 10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861).
- MS-SSIM. Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiscale Structural Similarity for Image Quality Assessment,” Asilomar Conference on Signals, Systems and Computers, 2003. [DOI 10.1109/ACSSC.2003.1292216](https://doi.org/10.1109/ACSSC.2003.1292216).
- VMAF. Z. Li, A. Aaron, I. Katsavounidis, A. Moorthy, and M. Manohara, “Toward a Practical Perceptual Video Quality Metric,” Netflix Technology Blog, 2016. [Blog](https://netflixtechblog.com/toward-a-practical-perceptual-video-quality-metric-653f208b9652). Models and attribution: [model/README.md](model/README.md), [Netflix/vmaf](https://github.com/Netflix/vmaf) commit `74f930d` (not a release tag; the previous tag is v3.2.1), [third_party/netflix-vmaf/NOTICE](third_party/netflix-vmaf/NOTICE).
- PU21. R. K. Mantiuk and M. Azimi, “PU21: A Novel Perceptually Uniform Encoding for Adapting Existing Quality Metrics for HDR,” Picture Coding Symposium, 2021. [DOI 10.1109/PCS50896.2021.9477471](https://doi.org/10.1109/PCS50896.2021.9477471). Reference implementation: [gfxdisp/pu21](https://github.com/gfxdisp/pu21). Notice: [licenses/PU21_BSD.txt](licenses/PU21_BSD.txt).

## License

PixVQA is licensed under the [MIT License](LICENSE) (SPDX `MIT`). Copyright (c) 2026 Pixop ApS. [docs/LICENSING.md](docs/LICENSING.md) explains third-party and model licenses. Dependency notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The logo at `docs/brand/pixop-vqa-logo.png` is included with the repository. The MIT license does not grant unrestricted trademark or logo use. See [docs/brand/README.md](docs/brand/README.md).
