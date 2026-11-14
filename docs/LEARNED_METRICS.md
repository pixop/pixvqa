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

### Calibration policy (select one)

| Policy | Env `PIXVQA_CVQA_MOS_POLICY` | HDR PQ | SDR |
|---|---|---|---|
| **Legacy (default)** | unset or `legacy_target_lambda1_both` | Original_U target λ=1.0 | U_v2 target λ=1.0 |
| **Reviewed S1 (opt-in)** | `reviewed_s1_hdr0_sdr1` | Original_U baseline λ=0.0 | U_v2 target λ=1.0 |
| **Selected composite (opt-in)** | `selected_s1_sdr_p1_hdr_c0` | Reviewed S1 C0 (no post-offset) | Reviewed S1 + **+0.4105018** MOS offset; **unclamped input** |

```bash
# Reviewed S1 F1-392 pair (independently selected; HDR differs, SDR identical to legacy SDR):
export PIXVQA_CVQA_MOS_POLICY=reviewed_s1_hdr0_sdr1
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4

# User-selected research composite (shadow-not-prod; not default):
export PIXVQA_CVQA_MOS_POLICY=selected_s1_sdr_p1_hdr_c0
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4
```

Invalid policy values **fail closed** (`mos_calibration_status=unavailable`, reason `invalid_policy:…`); the default is **not** substituted silently.

**Reviewed and selected upstream gate:** when `PIXVQA_CVQA_MOS_POLICY` is `reviewed_s1_hdr0_sdr1` or `selected_s1_sdr_p1_hdr_c0`, estimated perceptual quality is emitted only if git commit, checkpoint SHA-256, and profile SHA-256 all match pinned values in `pixvqa_cvqahdr.__init__` (explicit SHA comparison; `commit_verified` alone is insufficient). Otherwise `reason=cvqa_upstream_incompatible`. Legacy default is not gated. Preflight still checks git commit only for canonical CVQA scoring.

See [reviewed S1 integration notes](dev/CVQA_MOS_REVIEWED_S1_INTEGRATION.md) and [selected composite implementation](dev/CVQA_MOS_FINAL_SELECTED_IMPLEMENTATION.md).

**Legacy coefficients (default, 2026-11-14):** full-392 cohort, **target-MOS reweighting λ=1.0** (+100%) on both routes — human label *domain-aware logistic MOS calibration*, `weight_kind=target_mos`. Serialized from `mos_target_reweight_20261113/artifacts/endpoints.csv` (`original_ids`, not the prior baseline `final_fitted_models.json`).

- **PQ content** → `U0_M7_Bias` (research arm Original_U, M1 backbone + M7 gamma); **SDR** → `U_v2_M7_Bias` (U-v2 arm).
- Ridge **0.03**, cohort **`full-392-20261113`**; four-parameter M1 logistic with `b_eff=max(0,b)`; legacy and reviewed retain training-support input clamping; **selected** allows unbounded extrapolation with OOD warnings; final MOS **[1, 5]** output bounding on all policies.
- **HLG**, bare `bt2020`, missing/unknown transfer, unsupported heights, or untrusted tags → `MOS-calibrated CVQA: unavailable` (canonical score unchanged). **No filename heuristics.**
- U-domain target remapping is **training-only** and **not** applied at inference.

**Research caveats (see `evidence_caveat` in run JSON):** exploratory native AVT ratings previously informed U-v2 routing and are not blind confirmation; HDR target weighting worsens HDR AVT error vs baseline — HDR λ=1.0 is a **user policy choice**, not empirically superior global accuracy (SDR AVT error improves). NightTraffic alias-merged CV (55 source groups) is a research sensitivity only — **not** proof of identical stimuli; PixVQA runtime grouping is unchanged. Scale-bridge pseudo-pairs (120, three families) are not verified stimulus-identical. M7-Bias was a later challenger; initial validation selected plain M7, not M7-Bias — calibrator names are stable identifiers only.

**Selected v2 (2026-11-14):** bilateral input clamp removed for `selected_s1_sdr_p1_hdr_c0` only (`input_support_policy=unbounded_extrapolation`). Historical support bounds are observational; CVQA may exceed 10. Legacy/reviewed retain input clamp. MOS 4.5 output cap remains unimplemented.

The detailed JSON run record includes `mos_calibration` metadata (`calibration_version`, `coefficients_digest_sha256`, `weight_kind`, `weight_lambda`, `m1_params`, provenance) and, when available, `mos_calibrated_cvqa`.
