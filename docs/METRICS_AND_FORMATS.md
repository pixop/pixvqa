# Metrics, Formats and Hardware Support

| Name | Description |
|---|---|
| `psnr` | Peak signal-to-noise ratio (PSNR) and MSE on stored sample codes |
| `xpsnr` | Extended PSNR (XPSNR). Uses a constant compressed frame rate, or `--fps` |
| `ssim` | Structural similarity (SSIM): 11×11 Gaussian, sigma 1.5, K1=0.01, K2=0.03 |
| `msssim` | Multi-scale structural similarity, the five-scale product (MS-SSIM) |
| `vmaf` | Netflix VMAF through CPU libvmaf. Automatically selects a bundled SDR/PQ model when transfer, geometry and frame rate are known; `--model` overrides selection |
| `pu21-psnr` | PSNR on PU21-encoded display RGB |
| `pu21-ssim` | SSIM on PU21-encoded luminance |
| `pu21-msssim` | MS-SSIM on PU21-encoded luminance |
| [`cvvdp`](LEARNED_METRICS.md) | ColorVideoVDP JOD, optional external Python backend. Full-reference |
| [`cvqa-hdr-nr`](LEARNED_METRICS.md) | CompressedVQA-HDR NR, optional external Python backend. No-reference |
| [`simplevqa`](LEARNED_METRICS.md) | SimpleVQA NR UGC score; compressed SDR only. Pooled-video scalar |
| [`hdr-chipqa`](LEARNED_METRICS.md) | HDR-ChipQA NR dark-scene MOS-like score; SDR and PQ validated, HLG experimental. Pooled-video scalar |
| [`--qcs`](QCS.md) | Quality Consistency Score (QCS): temporal quality reliability descriptor derived from per-frame VQA scores |

PU21 is a perceptually uniform encoding of display light for HDR quality assessment. `pu21-xpsnr` and `pu21-vmaf` are not implemented.

**HDR / PU21:** To evaluate PQ or HLG video with PU21-PSNR, PU21-SSIM, or PU21-MS-SSIM, see the [PU21 usage guide](PU21.md) for recommended transfer-function and display settings.

Select VQA metrics using `--metrics`; enable the QCS descriptor separately using `--qcs`.

## Video Format Support by Metric

| Metric | Type | SDR – BT.709 | PQ / HDR10 – BT.2020 | HLG – BT.2020 |
|---|:---:|:---:|:---:|:---:|
| PSNR | FR | ✅ | ✅ | ✅ |
| XPSNR | FR | ✅ | ✅ | ✅ |
| SSIM | FR | ✅ | ✅ | ✅ |
| MS-SSIM | FR | ✅ | ✅ | ✅ |
| VMAF (SDR v1.0.16 models) | FR | ✅ | ❌ | ❌ |
| VMAF (HDR v1.1.8 models) | FR | ❌ | ✅ | ❌ |
| PU21-PSNR | FR | ❌ | ✅ | ✅ |
| PU21-SSIM | FR | ❌ | ✅ | ✅ |
| PU21-MS-SSIM | FR | ❌ | ✅ | ✅ |
| ColorVideoVDP | FR | ✅ | ✅ | ✅ |
| CompressedVQA-HDR | NR | ✅ | ✅ | ❌ |
| SimpleVQA | NR | ✅ | ❌ | ❌ |
| HDR-ChipQA | NR | ✅ | ✅ | 🔬 |

**Legend:** ✅ Supported · 🔬 Experimental support (validation incomplete) · ❌ Unsupported.

**PU21 metrics:** PSNR, SSIM and MS-SSIM can be computed after applying the PU21 perceptually uniform transfer function. PU21 transforms HDR luminance values to better approximate the nonlinear sensitivity of human vision. As a result, PU21 provides a perceptually motivated signal representation for conventional full-reference metrics.

**Notes:**
- Conventional PSNR, XPSNR, SSIM, and MS-SSIM operate on encoded sample values without HDR perceptual adaptation.
- SDR VMAF is intended for SDR; HDR VMAF v1.1.8 supports limited-range PQ/BT.2020, but not HLG.
- PU21 metrics support PQ and HLG, but not SDR in PixVQA.
- **SimpleVQA** (`simplevqa`): compressed **SDR** only; PQ/HLG streams and explicit PQ/HLG `--hdr-transfer` are rejected (`src/main.cpp`, Python adapter). Raw YUV is rejected. One pooled score per video; **QCS unsupported**.
- **HDR-ChipQA** (`hdr-chipqa`): YUV **4:2:0** 8- or 10-bit, raw or compressed. SDR uses BT.709 gamma-coded YUV (`color_space=BT709`); PQ HDR uses PQ-coded 10-bit YUV with stream PQ tags or `--hdr-transfer pq` (`color_space=BT2020`). Raw PQ/HLG requires `-f 420p:10` and `--fps`. **HLG** uses the same BT2020 feature path without EOTF and is **experimental** (monotonicity sanity checks only). One pooled score; **QCS unsupported**. See [CHIPQA_REFERENCE.md](dev/CHIPQA_REFERENCE.md).
- QCS applies to per-frame or sampled metrics only; `simplevqa` and `hdr-chipqa` are pooled-only. See [QCS](QCS.md).
- Metric groups (`sdr`, `pq`, `hlg`) select baseline native metrics and transfer defaults; **HDR-ChipQA** is opt-in because of host-RAM demands. **SimpleVQA** is incompatible with `pq`/`hlg` groups (SDR-only). **CVVDP** is also opt-in for groups because of GPU cost (e.g. `--metrics pq,cvvdp`); `all` still includes CVVDP. Add HDR-ChipQA explicitly (e.g. `--metrics pq,hdr-chipqa`). See [CLI Usage](CLI_USAGE.md#metric-groups).

## Metric Capabilities

PixVQA supports native and Python-based quality metrics. Metrics can be combined in a single `--metrics` list, and their input requirements are validated before processing.

| Metric | Reference (`-r`) | Test (`-t`) | QCS | Execution |
|---|---|---|---|---|
| Native metrics | Raw YUV or compressed video | Raw YUV or compressed video | Per-frame scores; metric-specific default ranges | PixVQA |
| `cvvdp` | Raw YUV420p10le (SDR/PQ/HLG) or compressed | Raw YUV420p10le (SDR/PQ/HLG) or compressed | Per-frame JOD-equivalent scores; default range [0, 10] | Python adapter |
| `cvqa-hdr-nr` | Not required | Compressed video only | Sampled-frame scores; default range [0, 10] | Python adapter |
| `simplevqa` | Not required | Compressed SDR video only | Pooled-video scalar; QCS unsupported | Python adapter (`python -m pixvqa_simplevqa`; optional `--use-worker` for CUDA batch) |
| `hdr-chipqa` | Not required | Raw YUV420 8/10-bit or compressed | Pooled-video scalar; QCS unsupported | Python adapter |

**Decoding:** Native metrics require `make swdec` or `make full` for compressed video. Python-based metrics decode compressed video independently of the build.

## Supported Video Formats

PixVQA supports raw YUV and compressed video through FFmpeg.

| Input format | Support |
|---|---|
| Raw YUV | 8-bit and 10-bit planar 4:2:0, 4:2:2, and 4:4:4 |
| Compressed video | Software decoding through FFmpeg (e.g., H.264, HEVC, AV1, VVC, VP9, MPEG-2, ProRes), subject to decoder availability and supported pixel formats |
| NVIDIA NVDEC | Hardware decoding of HEVC 10-bit 4:2:0 and AV1 8-bit/10-bit 4:2:0 |

Compressed video support depends on the available FFmpeg decoders. PixVQA accepts planar 8-bit and 10-bit YUV in 4:2:0, 4:2:2, and 4:4:4 formats; unsupported pixel formats are rejected rather than converted. 4:2:0 requires even width and height. Samples are planar, little-endian, with the low bits occupied.

AV1 software decoding requires an FFmpeg build with an AV1 decoder (for example, dav1d). NVDEC supports selected HEVC/AV1 formats; `--decode gpu` has no software fallback. `--decode auto` selects a supported decoder when available.

Inputs are paired in presentation order. If lengths differ, full-reference metrics evaluate the common frame prefix and emit a warning; incompatible geometry remains an error. PixVQA does not resynchronize timestamps.

## Metric Execution Backends

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

A ✅ shows where each metric runs. The table refers to **metric computation**, not video decoding. VMAF runs on CPU through `libvmaf`, while HDR-ChipQA uses Python with NumPy and SciPy. ColorVideoVDP uses PyTorch but is not a neural-network model. For SimpleVQA, the persistent CUDA worker is available through its standalone Python module, not yet through the PixVQA CLI.

**Input bit depth:** PSNR, XPSNR, SSIM, MS-SSIM, and PU21 metrics accept 8-bit and 10-bit input on CPU and CUDA. PU21 requires PQ or HLG video with the correct HDR settings. VMAF runs on CPU: SDR models accept 8-bit and 10-bit input, while HDR models require 10-bit 4:2:0 PQ/BT.2020 video (limited range, non-constant luminance). For Python-based metrics, see [Metric Capabilities](#metric-capabilities) and [Learned Metrics](LEARNED_METRICS.md).

`--device auto` uses CUDA for supported metrics when the build, GPU and input format permit; otherwise it uses CPU. `--device gpu` requests GPU execution without silently substituting CPU for GPU-only requests. `--decode gpu` supports selected HEVC/AV1 formats via NVDEC. CPU and GPU floating-point results may differ slightly.

**Limitations:**
- HDR transfer characteristics (PQ/HLG) come from the stream or from `--hdr-transfer`. They are not inferred from bit depth, BT.2020 primaries, or the filename.
- A variable frame rate is not treated as constant. XPSNR then requires `--fps`.
- Unsupported pixel formats are rejected rather than automatically converted.

For options see [CLI Usage](CLI_USAGE.md), `pixvqa --help` and `pixvqa --help-hdr`. HDR usage: [PU21](PU21.md).
