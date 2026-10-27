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

Each frame is one sample. That matches viewing time when every frame has the same duration. QCS does not read timestamps. A score at or below the lower limit contributes 0%. A score at or above the upper limit contributes 100%, and so does +inf. -inf contributes 0%. NaN is omitted. QCS supports the base metrics `psnr`, `xpsnr`, `ssim`, and `msssim` (selected with `--metrics`; `--qcs` enables their QCS reporting). It does not accept `vmaf` or the PU21 metrics. Defaults are `[30,50]` for PSNR and XPSNR, and `[0.8,1.0]` for SSIM and MS-SSIM. `--qcs-range psnr=40:55` sets one range. The name is the base metric, and the bounds must satisfy lower < upper. Identical PSNR is printed `999.99` and identical XPSNR is printed `inf`. Both still count in QCS.

```bash
pixvqa --metrics psnr --qcs --qcs-range psnr=30:50 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

## Quick start

The commands use 1920×1080. The same options apply at 3840×2160 with `-s 3840x2160`. Both inputs must use the same size. In the examples below, `pixvqa` denotes a binary on `PATH`; alternatively invoke `build/swdec/pixvqa` from the checkout, or `$HOME/pixvqa/bin/pixvqa-with-libs` after the guided installation. The GPU command requires the full NVIDIA build (`build/full/pixvqa` or its installed equivalent).

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

PixVQA targets **Linux x86-64**, including Ubuntu 22.04/24.04 under WSL2. Native Windows has not been validated. Run installation commands in a **Linux/Ubuntu shell**, not Windows Git Bash or MSYS2. The source tree does not include prebuilt FFmpeg, libdav1d, or CUDA libraries.

Choose **one** of the following installation methods from the root of your PixVQA source checkout.

### Option 1 — Automatic installation (guided; experimental)

From the repository root, run the interactive installer in an Ubuntu/Linux terminal:

```bash
bash install_pixvqa.sh
```

The installer asks two independent questions:

1. **Install dav1d 1.5.1 for software AV1 decoding?** `[Y/n]` (default: yes).
2. **Build NVIDIA GPU support (CUDA metrics + NVDEC)?** `[y/N]` (default: no).

**Choose dav1d = Yes** for the FFmpeg software-decoding build (`build/swdec/pixvqa`). With **dav1d = No**, the current guided installer installs only the raw-YUV CPU build (`build/cpu/pixvqa`), without compressed decoding. Selecting NVIDIA = Yes together with dav1d = No is rejected; this combination has not been validated. The installer does not currently accept command-line switches such as `--with-dav1d` or `--without-cuda`.

For NVIDIA = Yes, the installer checks for an existing compatible NVIDIA driver and **CUDA toolkit 12.8**, which must already be installed. It can install a separate conda-forge **GCC 13.4** toolchain and compatible `ffnvcodec` development headers when missing. It does **not** install the NVIDIA driver or CUDA toolkit. On WSL2, use the Windows-provided NVIDIA driver; do not install a second Linux display driver.

The installer default `PREFIX` is the source checkout. Set `PREFIX` before running it to install somewhere else. It builds pinned libvmaf commit `74f930d` with zimg 3.0.6 (not the historical v3.2.1 recipe), verifies that VMAF is enabled in PixVQA, checks runtime libraries, and runs CPU smoke QA. It reuses matching existing dependency installations where possible, but does not provide a general uninstall command or a guaranteed transaction/rollback mechanism.

**Run PixVQA after installation** using the launcher generated by Revision 6, which supplies private FFmpeg, dav1d and CUDA library paths in a new shell. With the default prefix that launcher is `bin/pixvqa-with-libs` inside the checkout:

```bash
./bin/pixvqa-with-libs --version
```

The installed executable is also at `$PREFIX/bin/pixvqa`, but direct execution from a new shell may need additional runtime-library configuration. Revision 6 does **not** create `$HOME/pixvqa-deps/pixvqa-env.sh` and does not automatically add `pixvqa` to `PATH`.

**Verification status (23 October 2026):** Revision 6 completed on Ubuntu 22.04 WSL2 with dav1d = Yes and NVIDIA = Yes. After rebuilding CUDA for `sm_89 + sm_120`, the CPU smoke profile passed (16/16) and the NVIDIA hardware profile passed (3/3) on an RTX 4070. The VVC hardware regression requires a separately staged VVC fixture and a modern FFmpeg executable capable of decoding VVC; see [fixture instructions](tests/fixtures/vvc/README.md). These results establish a working local configuration, **not** a clean-system installation or coverage of every installer choice. The installer remains experimental until those tests are completed; see [clean-install validation plan](docs/INSTALL_VALIDATION.md).

### Option 2 — Manual installation (advanced/custom builds)

Follow **[docs/INSTALL.md](docs/INSTALL.md)** for pinned dependency versions, FFmpeg configuration, libvmaf, CPU-only/raw-YUV or software-decoding builds, optional CUDA/NVDEC builds, installation, verification, and troubleshooting. The manual procedure saves the checkout root in `PIXVQA_SRC`, allowing you to return to the project after building dependencies in separate directories.

Depending on the build, the executable is `build/cpu/pixvqa` (raw YUV only), `build/swdec/pixvqa` (FFmpeg software decoding), or `build/full/pixvqa` (optional NVIDIA support). SDR VMAF models are stored under `$PREFIX/share/pixvqa/model`. HDR v1.1.8 models are in `hdr_vmaf_v1.1.8` and `hdr_vmaf_v1.1.8_hfr` under that directory. Pass the required JSON model to `--model` explicitly.

For independent clean-environment verification of automatic and manual paths, use [docs/INSTALL_VALIDATION.md](docs/INSTALL_VALIDATION.md). Do not treat a rerun on already-populated prefixes as a clean install.

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
| `--qcs` | Enable QCS for selected supported base metrics (`psnr`, `xpsnr`, `ssim`, `msssim`) |
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
- PU21. R. K. Mantiuk and M. Azimi, “PU21: A Novel Perceptually Uniform Encoding for Adapting Existing Quality Metrics for HDR,” Picture Coding Symposium, 2021. [DOI 10.1109/PCS50896.2021.9477471](https://doi.org/10.1109/PCS50896.2021.9477471). Reference implementation: [gfxdisp/pu21](https://github.com/gfxdisp/pu21). Notice: [licenses/PU21_BSD.txt](licenses/PU21_BSD.txt).

## License

PixVQA is licensed under the [MIT License](LICENSE) (SPDX `MIT`). Copyright (c) 2026 Pixop ApS. [docs/LICENSING.md](docs/LICENSING.md) explains third-party and model licenses. Dependency notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The logo at `docs/brand/pixop-vqa-logo.png` is included with the repository. The MIT license does not grant unrestricted trademark or logo use. See [docs/brand/README.md](docs/brand/README.md).
