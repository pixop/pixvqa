# Learned metrics

PixVQA integrates two optional Python-based metrics. Install their dependencies and model files separately; the native build does not install them.

| Metric | Purpose | Inputs | Upstream |
|---|---|---|---|
| `cvvdp` | **ColorVideoVDP**, full-reference perceptual quality (JOD) | Raw 10-bit 4:2:0 PQ/HLG or compressed video, including mixed raw/compressed pairs | [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP) |
| `cvqa-hdr-nr` | **CompressedVQA-HDR NR**, no-reference video quality | Distorted compressed video only; supports SDR and HDR10 | [CompressedVQA-HDR](https://github.com/sunwei925/CompressedVQA-HDR) |

Both can be combined with native metrics in `--metrics`. The `sdr`, `pq`, and `hlg` metric groups include compatible learned metrics. PixVQA resolves transfer interpretation from a metric group or reliable video metadata. PQ/HLG groups also set the CVVDP display; for SDR, select a display explicitly. Incompatible combinations are rejected.

## Setup

The upstream repositories provide the model implementations and assets. For the tested Python dependencies, create the shared environment using:

```bash
scripts/setup_learned_env.sh ~/pixvqa-learned-env ~/src/ColorVideoVDP
cp config/learned_metrics.example.toml config/learned_metrics.local.toml
```

The CompressedVQA-HDR NR checkout, model checkpoint (`.pth`), and calibration profile (`.npy`) must be supplied separately. Edit the local TOML paths (it is ignored by Git). For example:

```toml
[learned]
python = "~/pixvqa-learned-env/bin/python"
adapter_path = "../python"

[cvqa-hdr-nr]
repository = "~/src/CompressedVQA-HDR"
checkpoint = "~/models/NR_HDR_VQA.pth"
profile = "~/models/NR_HDR_VQA.npy"
```

CVVDP uses the installed `cvvdp` executable, and needs `ffmpeg` and `ffprobe` on `PATH` for compressed video. Consult [`config/learned_metrics.example.toml`](../config/learned_metrics.example.toml) for optional paths.

## Examples

```bash
# Full-reference CVVDP; infer PQ/HLG/SDR from tagged compressed streams.
./bin/pixvqa --metrics cvvdp -r reference.mp4 -t distorted.mp4

# Explicit interpretation/display when input metadata is insufficient.
./bin/pixvqa --metrics cvvdp -r reference.mp4 -t distorted.mp4 \
  --cvvdp-interpretation pq --cvvdp-display standard_hdr_pq

# No-reference metric; samples the whole compressed clip.
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4

# Combine native and learned metrics, with PQ defaults.
./bin/pixvqa --metrics psnr,cvvdp,cvqa-hdr-nr --hdr-transfer pq \
  -r reference.mp4 -t distorted.mp4
```

`cvvdp` requires `-r` and `-t`. `cvqa-hdr-nr` requires only `-t`; an NR-only `-r` is ignored. `--frames` limits CVVDP and native metrics, but is ignored with a warning by `cvqa-hdr-nr`, which evaluates the whole clip. CVVDP raw YUV requires declared geometry/format and frame rate when both inputs are raw.

Use `--device auto|cpu|gpu` (default `auto`) to select an execution device. For all other options, run `./bin/pixvqa --help-advanced` or `./bin/pixvqa --help-hdr`.
