# Optional Learned Metrics

PixVQA supports four optional Python-based video quality metrics. They are **not required** for native metrics such as PSNR, SSIM, XPSNR, VMAF, and PU21.

| Metric | Type | Supported video |
|---|---|---|
| `cvvdp` (ColorVideoVDP) | Full-reference, perceptual JOD | SDR, PQ and HLG; compressed and supported raw 10-bit YUV420 |
| `cvqa-hdr-nr` (CompressedVQA-HDR NR) | No-reference | Compressed SDR and HDR10 |
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

## Performance considerations

**CVVDP uses GPU memory (VRAM).** High resolutions and concurrent jobs may need substantial VRAM. Monitor usage with `nvidia-smi`; `--cvvdp-gpu-mem GB` supplies an optional budget but cannot guarantee that a workload fits.

**HDR-ChipQA can use substantial system RAM**, particularly with parallel jobs. Start with one job on memory-constrained systems and monitor RAM before increasing concurrency. Detailed benchmark observations belong in [Advanced Setup Reference](dev/LEARNED_METRICS_ADVANCED.md).

For environment customization, offline installation, WSL-specific details, and upstream packaging notes, see [Advanced Setup Reference](dev/LEARNED_METRICS_ADVANCED.md). For algorithm-specific behavior, see [HDR-ChipQA Reference](dev/CHIPQA_REFERENCE.md) and [SimpleVQA Reference](dev/SIMPLEVQA_REFERENCE.md).
