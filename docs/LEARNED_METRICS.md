# Optional Learned Metrics

PixVQA supports four optional Python-based video quality metrics. They are **not required** for native metrics such as PSNR, SSIM, XPSNR, VMAF, and PU21.

| Metric | Type | Supported video |
|---|---|---|
| `cvvdp` (ColorVideoVDP) | Full-reference, perceptual JOD | SDR, PQ and HLG; compressed and supported raw 10-bit YUV420 |
| `cvqa-hdr-nr` (CompressedVQA-HDR NR) | No-reference | Compressed SDR and HDR10 |
| `simplevqa` (SimpleVQA) | No-reference | Compressed SDR only |
| `hdr-chipqa` (HDR-ChipQA) | No-reference, one score per video | SDR and PQ; experimental HLG; compressed or supported raw YUV420 |

For complete format details, see [Metrics & Formats](METRICS_AND_FORMATS.md).

## Install

From the PixVQA checkout, after building the native executable:

```bash
bash scripts/install_learned_metrics.sh
```

Alternatively, `install_pixvqa.sh` offers optional learned-metric installation during guided setup (answer **yes** when prompted).

The installer creates a local **`.venv`** Python environment, installs dependencies, downloads or reuses the upstream repositories and pretrained checkpoints, verifies their integrity, and configures PixVQA. It also creates **`.venv-chipqa-inference`** for HDR-ChipQA's separate inference requirements. A second run reuses verified assets.

Key Python packages include **PyTorch, torchvision, PyTorchVideo, NumPy, SciPy, OpenCV, scikit-image, pandas, matplotlib, and scikit-learn** (the latter in HDR-ChipQA's separate inference environment). Exact pinned versions and additional dependencies are in [the requirements file](../python/requirements/learned-env.txt) and installer scripts.

The installer needs network access for missing assets. It reports an error if a required checkpoint cannot be downloaded or verified. No manual checkpoint download is normally necessary.

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

The `sdr`, `pq`, and `hlg` groups include the baseline metrics but **not CVVDP or HDR-ChipQA**. Add either one explicitly when needed (`--metrics sdr,cvvdp` or `--metrics pq,hdr-chipqa`). See [CLI Usage](CLI_USAGE.md#metric-groups).

## Performance considerations

**CVVDP uses GPU memory (VRAM).** High resolutions and concurrent jobs may need substantial VRAM. Monitor usage with `nvidia-smi`; `--cvvdp-gpu-mem GB` supplies an optional budget but cannot guarantee that a workload fits.

**HDR-ChipQA uses substantial system RAM.** Five concurrent 1080p/60, ~20-second jobs triggered Linux out-of-memory kills on WSL2 configured with 10 GB RAM and 4 GB swap. The benchmark campaign worked after removing HDR-ChipQA. For memory-constrained multi-job campaigns, select it separately and measure peak RAM before increasing concurrency. This observation does not establish 4K memory requirements.

For environment customization, offline installation, WSL-specific details, and upstream packaging notes, see [Advanced Setup Reference](dev/LEARNED_METRICS_ADVANCED.md). For algorithm-specific behavior, see [HDR-ChipQA Reference](dev/CHIPQA_REFERENCE.md) and [SimpleVQA Reference](dev/SIMPLEVQA_REFERENCE.md).
