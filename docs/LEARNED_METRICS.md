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

CVQA-HDR-NR continues to report the canonical **0–10** `quality_score` from the official NR logistic calibration. On every successful NR run, when ffprobe stream metadata **trustworthily** identifies **SDR** (`bt709` and related SDR tags) or **HDR PQ** (`smpte2084` / `st2084`) at **1080** or **2160** encoded height, PixVQA also emits a separate **estimated perceptual quality 1–5** field (`mos_calibrated_cvqa` in protocol/JSON for compatibility). ffprobe failures, metadata errors, or unsupported domains only suppress that field; the canonical score is always unchanged. Unexpected calibration errors are caught at a narrow boundary and likewise suppress MOS output without affecting the canonical score.

### Selected calibration (automatic)

PixVQA applies one scientifically selected transfer — policy `selected_s1_sdr_p1_hdr_c0` — automatically when metadata is supported. No environment variable is required.

**Updated 16 November 2026:** the models are now trained separately: 200 SDR observations only (30 recorded sources), and 192 HDR PQ observations only (35 recorded sources). The original M1 sigmoid + M7 metadata-correction structure is unchanged. The calibration version and provenance hashes distinguish these new coefficients from the old mixed-data policy.

Normalization is separate for each transfer, using `z_T = (CVQA − mean_T) / std_T`:

| Transfer | Mean | Standard deviation |
|---|---:|---:|
| SDR | 8.010332079849347 | 1.241145006178454 |
| HDR PQ | 8.769753592063894 | 0.9333470039280067 |

Estimated MOS (input CVQA is never clipped):

`MOS = clip(1 + 4·σ(a_T + b_T·z_T) + d_{T,h}, 1, 5)`

| Transfer | Height | a_T | b_T | d_{T,h} |
|---|---|---:|---:|---:|
| SDR | 1080p | −0.49117629 | 1.09778388 | +0.51077329 |
| SDR | 2160p | −0.49117629 | 1.09778388 | +0.59620846 |
| HDR PQ | 1080p | +0.07227164 | 0.82933048 | −0.13103674 |
| HDR PQ | 2160p | +0.07227164 | 0.82933048 | +0.09786364 |

**Feature-branch mapping (19 November 2026):** On branch **feature/cvqa-mos-v7-local-tail-006**, the formula above describes the **unchanged fitted v7 baseline**. A new separately versioned, monotone local C² tail replaces only its low CVQA extrapolation: SDR joins exact v7 at CVQA **5.5**, with the two SDR CVQA 4.134 markers each **0.060 MOS lower**, and HDR PQ joins exact v7 at CVQA **6.268450** using the earlier localized HDR C² curves. All routes satisfy **MOS(CVQA=0)=1**, while the canonical score remains unchanged. [Four-route curves, full coefficient values, integrity hashes and limitations](dev/CVQA_MOS_V7_LOCAL_TAIL_006.md). This feature is **not merged to main**.

The SDR native-MOS offset is +0.5107732905216783; no offset is applied to PQ. Training score support is **4.1345–10.0383 for SDR** and **6.2685–10.0515 for PQ**. Values outside a route's range trigger its own out-of-distribution flags but are not clamped. The MOS output remains within [1,5].

The selected models were evaluated retrospectively: SDR54 MAE 0.2780 with source-family-held-out offsets, and HDR35 MAE 0.6042 without an offset (worse than 0.5552 for the old mixed-data model). These are provisional **estimated** ratings, not observed MOS. High-quality HDR MOS ≥4.5 remains unvalidated.

Full-precision independent research coefficients and evaluation: `perceptual_optimization/calibration/production_readiness_20261113/CVQA_TO_MOS_TRANSFER_SEPARATED_DELIVERABLE_20261116/`. The frozen source copy, retained historical golden fixture, new oracle and implementation validation are described in [Transfer-Separated v7 Implementation](dev/CVQA_MOS_TRANSFER_SEPARATED_V7.md).

`PIXVQA_CVQA_MOS_POLICY` is **obsolete**. Unset or `selected_s1_sdr_p1_hdr_c0` is accepted; any other non-empty value (including legacy `legacy_target_lambda1_both` or `reviewed_s1_hdr0_sdr1`) **fail closed** with `obsolete_policy_env:…` so alternate formulas cannot be activated.

**Upstream gate:** estimated perceptual quality is emitted only if git commit, checkpoint SHA-256, and profile SHA-256 all match pinned values in `pixvqa_cvqahdr.__init__`. Otherwise `reason=cvqa_upstream_incompatible`. Preflight still checks git commit only for canonical CVQA scoring.

- **PQ content** → `U0_M7_Bias`; **SDR** → `U_v2_M7_Bias` (stable calibrator identifiers).
- **HLG**, bare `bt2020`, missing/unknown transfer, unsupported heights → MOS suppressed; canonical score unchanged.
- Regression: `PYTHONPATH=python python3 scripts/test_cvqahdr_mos_calibration.py` (68-row v7 golden fixture, 89 native per-clip comparisons, VD6 fixed-offset transfer and independent numerical oracle).

The detailed JSON run record includes `mos_calibration` metadata (`calibration_version`, `coefficients_digest_sha256`, `route_sigmoid`, provenance) and, when available, `mos_calibrated_cvqa`.
