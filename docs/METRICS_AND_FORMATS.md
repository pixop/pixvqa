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
| [`--qcs`](QCS.md) | Quality Consistency Score (QCS): temporal quality reliability descriptor derived from per-frame VQA scores |

PU21 is a perceptually uniform encoding of display light for HDR quality assessment. `pu21-xpsnr` and `pu21-vmaf` are not implemented.

**HDR / PU21:** To evaluate PQ or HLG video with PU21-PSNR, PU21-SSIM, or PU21-MS-SSIM, see the [PU21 usage guide](PU21.md) for recommended transfer-function and display settings.

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
| CVVDP (`sdr` / `pq` / `hlg` interpretation) | ✅ | ✅ | ✅ |
| CompressedVQA-HDR NR | ✅ | ✅ | ❌ |

**Legend:**

- ✅ Recommended
- ⚠️ Supported, but not generally recommended for HDR perceptual assessment
- ❌ Unsupported metric/model and format combination

**Notes:**
- Conventional PSNR, XPSNR, SSIM, and MS-SSIM operate on encoded sample values without HDR perceptual adaptation.
- SDR VMAF is intended for SDR; HDR VMAF v1.1.8 supports limited-range PQ/BT.2020, but not HLG.
- PU21 metrics support PQ and HLG, but not SDR in PixVQA.
- QCS is available for supported native and learned metrics; see [QCS](QCS.md).
- Metric groups configure transfer/display defaults and include additional conventional metrics even for HDR; see [CLI Usage](CLI_USAGE.md#metric-groups).

## Metric Capabilities

PixVQA supports native and Python-based quality metrics. Metrics can be combined in a single `--metrics` list, and their input requirements are validated before processing.

| Metric | Reference (`-r`) | Test (`-t`) | QCS | Execution |
|---|---|---|---|---|
| Native metrics | Raw YUV or compressed video | Raw YUV or compressed video | Per-frame scores; metric-specific default ranges | PixVQA |
| `cvvdp` | Raw YUV420p10le (SDR/PQ/HLG) or compressed | Raw YUV420p10le (SDR/PQ/HLG) or compressed | Per-frame JOD-equivalent scores; default range [0, 10] | Python adapter |
| `cvqa-hdr-nr` | Not required | Compressed video only | Sampled-frame scores; default range [0, 10] | Python adapter |

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

Bundled HDR VMAF requires limited-range 10-bit 4:2:0 PQ/BT.2020 non-constant-luminance samples. PU21 requires PQ or HLG. A dash indicates no CUDA implementation. VMAF always executes on CPU libvmaf.

`--device auto` uses CUDA for supported metrics when the build, GPU and input format permit; otherwise it uses CPU. `--device gpu` requests GPU execution without silently substituting CPU for GPU-only requests. `--decode gpu` supports selected HEVC/AV1 formats via NVDEC. CPU and GPU floating-point results may differ slightly.

**Limitations:**
- HDR transfer characteristics (PQ/HLG) come from the stream or from `--hdr-transfer`. They are not inferred from bit depth, BT.2020 primaries, or the filename.
- A variable frame rate is not treated as constant. XPSNR then requires `--fps`.
- Unsupported pixel formats are rejected rather than automatically converted.

For options see [CLI Usage](CLI_USAGE.md), `pixvqa --help` and `pixvqa --help-hdr`. HDR usage: [PU21](PU21.md).
