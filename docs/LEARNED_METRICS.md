# Optional Learned Metrics

PixVQA supports four optional Python-based video quality metrics. They are **not required** for native metrics such as PSNR, SSIM, XPSNR, VMAF, and PU21.

| Metric | Type | Supported video |
|---|---|---|
| `cvvdp` (ColorVideoVDP) | Full-reference, perceptual JOD | SDR, PQ and HLG; compressed and supported raw 10-bit YUV420 |
| `cvqa-hdr-nr` (CompressedVQA-HDR NR) | No-reference | Compressed SDR and HDR10; automatic provisional MOS 1–5 for supported metadata |
| `simplevqa` (SimpleVQA) | No-reference | Compressed SDR only |
| `hdr-chipqa` (HDR-ChipQA) | No-reference, one score per video | SDR and PQ; experimental HLG; compressed or supported raw YUV420 |

For complete format details, see [Metrics & Formats](METRICS_AND_FORMATS.md).

**Learned NR production defaults (2026-11-10):** SimpleVQA and CVQA-HDR-NR
decode video exclusively through a bounded-memory FFmpeg BGR24 pipe;
OpenCV remains used for pixel resizing/color conversion only. Deterministic
PyTorch inference is enabled by default. No decoder-selection macro or
development environment variables are required. The installer provisions pinned
FFmpeg n8.1.3 `ffmpeg` and `ffprobe` into the learned `.venv/bin` directory;
learned decoding resolves those binaries directly (system `PATH` is not used).
See [Production operations](operations/LEARNED_NR_PRODUCTION.md)
and [decoder bug-fix report](dev/LEARNED_METRIC_DECODER_BUGFIX_REPORT_2026-11-10.md).


## Install

From the PixVQA checkout, after building the native executable:

```bash
bash scripts/install_learned_metrics.sh
```

Alternatively, `install_pixvqa.sh` offers optional learned-metric installation during guided setup (answer **yes** when prompted).

### CUDA and Python requirements

PixVQA separates native GPU processing from optional AI-based quality metrics:

- **Native PixVQA — CUDA Toolkit 12.8:** Required with a compatible NVIDIA driver only to build the CUDA/NVDEC-enabled `full` variant. CPU-only and software-decoding builds do not require CUDA.
- **Learned metrics — CUDA 13.0 (PyTorch wheels):** SimpleVQA, CVQA-HDR-NR and other learned metrics share **Python 3.12** with **PyTorch 2.13.0+cu130** and **torchvision 0.28.0+cu130**. CUDA 13.0 runtime components are supplied through Python packages; **you do not need to install a second system-wide CUDA 13.0 toolkit**.
- **HDR-ChipQA — Python 3.10 (regression only):** Feature extraction uses the shared Python 3.12 environment. Its pretrained scikit-learn regression model runs in a small isolated **Python 3.10 / scikit-learn 1.0.2** environment (`.venv-chipqa-inference`). No additional CUDA installation is needed.

The guided installer configures these environments when learned metrics are enabled. **Existing Python environments are preserved rather than upgraded automatically.** If their installed versions do not match the dependency lockfile, the installer explains how to create a replacement by setting `PIXVQA_LEARNED_ENV_DIR` to a new absolute path. Reuse runs `scripts/verify_learned_env_pins.sh`, which checks every pin in `learned-env.txt` except ColorVideoVDP (commit stamp is authoritative). CUDA stack wheels (`nvidia-*`, `cuda-*`, `triton`) are required only when the installed `torch` already matches the lock; older stacks that bundle those libraries inside the `torch` wheel are rejected on the `torch` / `torchvision` pins instead of false “not installed” noise.

The installer automatically:

1. Creates a repository-local Python environment (`.venv`).
2. Installs the Python dependencies.
3. Downloads or reuses the upstream metric implementations and pretrained model checkpoints, verifying the downloaded assets.
4. Builds and installs pinned FFmpeg n8.1.3 `ffmpeg` and `ffprobe` into `.venv/bin`.
5. Configures the metrics for use with `pixvqa`.

HDR-ChipQA also uses a separate `.venv-chipqa-inference` environment, created automatically. Re-running the installer reuses verified assets.

**Main packages:** PyTorch, torchvision, PyTorchVideo, NumPy, SciPy, OpenCV, scikit-image, pandas, matplotlib, and scikit-learn (in HDR-ChipQA's separate inference environment), plus metric-specific dependencies. See the [complete pinned dependency list](../python/requirements/learned-env.txt).

An internet connection is needed to obtain missing models. Normally you do **not** need to download repositories or checkpoints manually; if a download fails, the installer reports an error.

## Run

```bash
# Compressed SDR, no reference:
./bin/pixvqa --metrics simplevqa -t distorted.mp4

# Compressed SDR or HDR10, no reference:
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4

# ColorVideoVDP, with a reference:
./bin/pixvqa --metrics pq,cvvdp -r reference.mp4 -t distorted.mp4

# HDR-ChipQA, with explicit selection:
./bin/pixvqa --metrics hdr-chipqa -t distorted.mp4
```

The `sdr`, `pq`, and `hlg` groups intentionally exclude **CVVDP** (potentially high GPU memory use) and **HDR-ChipQA** (potentially high system memory use). Add either explicitly (`--metrics sdr,cvvdp` or `--metrics pq,hdr-chipqa`). The legacy `all` selection includes CVVDP; see [Metric groups](CLI_USAGE.md#metric-groups).

## Playback frame counts and MP4 edit lists

CVQA-HDR-NR and SimpleVQA use the pinned FFmpeg BGR24 playback decoder, without frame-rate conversion or an OpenCV decoder fallback. MP4/MOV container `nb_frames` can include coded preroll excluded by the edit list. The shared frame-count resolver scans packet flags and counts; if discard flags or a packet-count discrepancy are present, it checks that the unrestricted coded track count matches the container and obtains the actual presented count using ffprobe with the edit list enabled. For ordinary MP4s, this is a compressed-packet scan rather than an extra video decode. Counts for unchanged local regular files are cached within each process. Both adapters plan samples from the reconciled presentation count; the original container count and count basis remain in JSON provenance. For edited media the extra exact count requires a decode-only preflight pass. Playback never disables edit lists.

A full capture validates the decoder's exit status and exact byte count on the final expected frame. Partial frames, unreconciled count discrepancies, probe errors and nonzero decoder exits fail closed rather than padding a failed decode. Explicit frame-limited reads still release early. The B08 coefficients and temporal midpoint formula are unchanged.

Regression: `python scripts/test_ffmpeg_bgr_capture.py` in the learned environment with an encoder-capable host FFmpeg (set `PIXVQA_TEST_FIXTURE_FFMPEG` if needed). The regular full QA suite also runs the synthetic edit-list case with its host FFmpeg pair; the hosted CPU workflow runs the decoder contracts without models.

## Concurrent jobs

For multiple PixVQA processes on one GPU, `--scheduler-admission` enables opt-in global admission for **SimpleVQA** and **CVQA-HDR-NR**. It queues learned-metric inference according to shared resource reservations and concurrency limits. It does not control NVENC jobs or guarantee that memory use stays below a hard limit. The default is off; users running a single evaluation do not need this option.

```bash
./bin/pixvqa --scheduler-admission --metrics simplevqa,cvqa-hdr-nr -t distorted.mp4
```

For existing automated job scripts, `PIXVQA_SCHEDULER_ADMISSION=1` is the equivalent environment-variable option. See [CLI Usage](CLI_USAGE.md#concurrent-learned-metric-jobs) and [Advanced scheduler configuration](dev/METRIC_SCHEDULER.md).

## SimpleVQA rate–quality limitations

SimpleVQA is a **no-reference UGC-trained prediction model**, not a distortion measure against a pristine reference. **Higher encoding bitrate is not guaranteed to produce a higher SimpleVQA score.** In a controlled RTX 5080 HEVC BarScene 1080p test at 10/15/20/25/30 Mb/s, its scores were non-monotonic. The 10 Mb/s and 30 Mb/s endpoint scores were each **bit-identical across five independent CUDA evaluations**, and the original upstream implementation reproduced the inverse 10-versus-30 Mb/s ranking on the same encoded files. Other test content, including BasketballDrive, showed increasing scores with bitrate.

This is consistent with content-dependent model sensitivity and possibly out-of-distribution generalization for high-quality, contribution-like encoding; **the training-bitrate distribution has not been verified**, so that causal hypothesis is not established. Do not use SimpleVQA as the only RD/BD-rate optimization criterion or assume score monotonicity. Numerical reproducibility and subjective-validity/monotonicity are separate evaluation questions. See [production acceptance](operations/LEARNED_NR_PRODUCTION.md) and the local experiment report `reports/bar_scene_simplevqa_5080_20261110/REPORT.md` (benchmark artifacts are not distributed).

## Performance Considerations

**CVVDP uses GPU memory (VRAM).** High resolutions and concurrent jobs may need substantial VRAM. Monitor usage with `nvidia-smi`; `--cvvdp-gpu-mem GB` supplies an optional budget but cannot guarantee that a workload fits.

**HDR-ChipQA can use substantial system RAM**, particularly with parallel jobs. Start with one job on memory-constrained systems and monitor RAM before increasing concurrency. Detailed benchmark observations belong in [Advanced Setup Reference](dev/LEARNED_METRICS_ADVANCED.md).

For environment customization, offline installation, WSL-specific details, and upstream packaging notes, see [Advanced Setup Reference](dev/LEARNED_METRICS_ADVANCED.md). For algorithm-specific behavior, see [HDR-ChipQA Reference](dev/CHIPQA_REFERENCE.md) and [SimpleVQA Reference](dev/SIMPLEVQA_REFERENCE.md).

## CVQA-HDR-NR estimated perceptual quality (provisional research)

CVQA-HDR-NR continues to report the canonical `quality_score` (nominal 0–10, values above 10 allowed) from the official NR logistic calibration. On every successful NR run, when ffprobe stream metadata **trustworthily** identifies **SDR** (`bt709` and related SDR tags) or **HDR PQ** (`smpte2084` / `st2084`) at **1080** or **2160** encoded height, PixVQA also emits a separate **estimated perceptual quality 1–5** field (`mos_calibrated_cvqa` in protocol/JSON for compatibility). ffprobe failures, metadata errors, or unsupported domains only suppress that field; the canonical score is always unchanged. Unexpected calibration errors are caught at a narrow boundary and likewise suppress MOS output without affecting the canonical score.

### Selected calibration (automatic)

The sole active mapping is `a2_tv123_ai25_h1_b08_w005`: frozen A2-TV123 (historical human sigmoid and AI25 amplitude) plus a uniform, anchored full-range 5% H1 shape correction. H1 adds direct five-category human ACR evidence: 224 ratings on 28 HDR PQ 2160p clips, 196 retained after observer screening. The correction transports this evidence to the other routes as an engineering assumption, not direct subjective validation.

See [canonical B08 algorithm and validation](dev/CVQA_MOS_A2_TV123_INTEGRATION.md). Input CVQA is not clamped, values above 10 remain valid, and estimated MOS is bounded to [1,5]. The legacy calibration modules and selectors have been removed. No calibration algorithm or weight argument is needed, and the old MOS selector environment variables are not read.

**Upstream gate:** estimated perceptual quality is emitted only if git commit, checkpoint SHA-256, and profile SHA-256 all match pinned values in `pixvqa_cvqahdr.__init__`. Otherwise `reason=cvqa_upstream_incompatible`. Preflight still checks git commit only for canonical CVQA scoring.

- **PQ or SDR content** → `A2_TV123_H1_B08` (fixed 5% correction, all four routes).
- **HLG**, bare `bt2020`, missing/unknown transfer, unsupported heights → MOS suppressed; canonical score unchanged.
- Regression: `PYTHONPATH=python python3 scripts/test_cvqahdr_mos_calibration.py` (independent B08 formula oracle, standalone parity, domain and upstream gates, provenance, selector removal and optional-stage safety).

The detailed JSON run record includes `mos_calibration` metadata (`calibration_version`, `coefficients_digest_sha256`, `unified_sigmoid`, `h1_shape_weight`, evidence/extrapolation provenance) and, when available, `mos_calibrated_cvqa`.
