# Metrics, Formats and Hardware Support

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
| [`cvvdp`](COLORVIDEOVDP.md) | ColorVideoVDP JOD, optional external Python backend. Full-reference |
| [`cvqa-hdr-nr`](COMPRESSEDVQA_HDR.md) | CompressedVQA-HDR NR, optional external Python backend. No-reference |
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
- QCS can be applied to any metric using `--qcs`. CVVDP QCS integrates a per-frame JOD-equivalent sequence taken from the official full-video pass. CompressedVQA-HDR NR QCS uses about one sampled frame per second. Both default to [0, 10], which is used only for QCS and never clamps the reported scores.
- `--metrics sdr`, `--metrics pq`, and `--metrics hlg` select a fixed set per transfer and fill its colorimetry and display. The set is not the ✅ column: PQ and HLG also include PSNR, XPSNR, SSIM, and MS-SSIM, which stay ⚠️ for HDR perceptual assessment. CVVDP HLG uses the upstream `standard_hdr_hlg` display. CompressedVQA-HDR NR is included for SDR and PQ and excluded for HLG. See [metric groups](CLI_USAGE.md#metric-groups).

## Metric Capabilities

All metrics can be mixed in one `--metrics` list. PixVQA checks every metric's requirements before running any metric.

| Metric | Reference (`-r`) | Raw YUV | Compressed video | `--frames` | QCS | Runs in |
|---|---|---|---|---|---|---|
| Native metrics | Required | ✅ | FFmpeg builds (`make swdec`, `make full`) | ✅ | Per frame, default ranges | PixVQA |
| `cvvdp` | Required | YUV420p10le PQ or HLG, both inputs | ✅ (both inputs) | ✅ | Per-frame JOD-equivalent, default [0, 10] | Python adapter |
| `cvqa-hdr-nr` | Not used; ignored when it is the only metric | ❌ | ✅ | ❌ | Sampled frames, default [0, 10] | Python adapter |

The `make cpu` and `make cuda` builds have no FFmpeg decoder, so native metrics there accept raw YUV only. Compressed input to a native metric on that build fails with a clear error before any metric runs. `cvvdp` and `cvqa-hdr-nr` decode in their own Python environments and work on compressed video with any build.

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

Option names and defaults are in `pixvqa --help` and `pixvqa --help-hdr`. QCS is in [docs/QCS.md](QCS.md). PQ and HLG setup is in [docs/PU21.md](PU21.md).
