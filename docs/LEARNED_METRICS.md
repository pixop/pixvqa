# Optional Learned Metrics

Two separately installed Python-backed metrics can run alongside native PixVQA metrics:

| Metric | Type | Supported inputs |
|---|---|---|
| `cvvdp` ([ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP)) | Full-reference JOD | Compressed or raw YUV420p10le, including mixed raw/compressed |
| `cvqa-hdr-nr` ([CompressedVQA-HDR](https://github.com/sunwei925/CompressedVQA-HDR)) | No-reference | Compressed SDR and HDR10; not raw YUV or HLG |

## Setup

The guided installer does **not** install these models. Install the Python environment and provide required model files:

```bash
scripts/setup_learned_env.sh ~/pixvqa-learned-env ~/src/ColorVideoVDP
cp config/learned_metrics.example.toml config/learned_metrics.local.toml
```

Edit `config/learned_metrics.local.toml` with your paths; it is Git-ignored. CompressedVQA-HDR NR needs its upstream checkout, `.pth` checkpoint and `.npy` calibration profile. The [example configuration](../config/learned_metrics.example.toml) lists the available fields. CVVDP needs its installed executable; for compressed files, `ffmpeg` and `ffprobe` must be on `PATH`.

> ⚠️ **CVVDP GPU resource usage:** High-resolution video and concurrent CVVDP jobs can require substantial GPU memory and compute. Monitor GPU utilization and available VRAM (for example, with `nvidia-smi`) and verify that the workload fits your GPU before running large batches. CVVDP has been verified on NVIDIA GeForce RTX 4070 and RTX 5080; this does not guarantee sufficient resources for every resolution, configuration or degree of parallelism. The optional `--cvvdp-gpu-mem GB` setting provides a memory budget, but is not a substitute for monitoring actual usage.

## Examples

```bash
# Full-reference ColorVideoVDP (PQ):
./bin/pixvqa --metrics cvvdp --cvvdp-interpretation pq \
  --cvvdp-display standard_hdr_pq -r reference.mp4 -t distorted.mp4

# CompressedVQA-HDR NR:
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4

# Combine with native metrics:
./bin/pixvqa --metrics psnr,cvvdp,cvqa-hdr-nr --hdr-transfer pq \
  -r reference.mp4 -t distorted.mp4
```

Compressed video metadata may supply transfer; when it cannot, set `--hdr-transfer` or `--cvvdp-interpretation`. Specify a suitable SDR display explicitly (e.g. `--cvvdp-display standard_fhd`). Raw CVVDP input requires geometry, format, rate and correct transfer/color interpretation.

`--frames N` caps both learned metrics. In a mixed native/learned run, all metrics use the common successfully compared frame prefix. The NR model samples at approximately 1 Hz by default; adjust using `--cvqa-nr-sample-interval-sec` and `--cvqa-nr-chunk-size-sec`.

See [CLI Usage](CLI_USAGE.md) for options and [Metrics & Formats](METRICS_AND_FORMATS.md) for compatibility.
