# CLI Usage

Run `./bin/pixvqa` from the checkout (or `pixvqa` if installed on `PATH`). Set `--metrics`, a distorted input (`-t`), and a reference (`-r`) for full-reference metrics.

## Examples

**Compressed video** (requires FFmpeg-enabled build):

```bash
./bin/pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_hevc.mp4 -t samples/dist_hevc.mp4
```

**Raw YUV** (explicit size and planar format):

```bash
./bin/pixvqa --metrics psnr,ssim,xpsnr --fps 50 \
  -r reference.yuv -t distorted.yuv -s 1920x1080 -f 420p:10
```

**PQ/HDR** (PU21 metrics):

```bash
./bin/pixvqa --metrics pu21-psnr,pu21-msssim --hdr-transfer pq \
  -r reference.yuv -t distorted.yuv -s 1920x1080 -f 420p:10
```

For raw XPSNR and automatic VMAF selection, specify `--fps` when rate is not available from compressed metadata. Raw YUV has no embedded color metadata: supply `--hdr-transfer` for HDR and for VMAF auto-selection.

### VMAF models

**PixVQA automatically selects the bundled VMAF JSON model** (default `--model auto`); you do not normally need `--model`. This is PixVQA's model-resolution layer; libvmaf itself still loads a specific model file. The choice uses:

- **Transfer:** SDR or PQ from explicit settings, the metric group, or unambiguous compressed-stream tags.
- **Resolution:** exactly 1920×1080 (3H) or 3840×2160 (1.5H).
- **Constant frame rate:** at most 30 fps or above 30 fps (HFR model); use `--fps` for raw input or to override conflicting/unknown rate metadata.

Auto-selection does not support HLG, other resolutions, ambiguous transfer, or unknown rate. To use another suitable model, set `--model PATH`. An explicit path **replaces** auto-selection, but PixVQA still checks model accessibility and compatibility. Bundled HDR VMAF is designed for limited-range YUV420p10 PQ with BT.2020 primaries and a non-constant-luminance matrix. PixVQA does not convert incompatible input samples.

```bash
./bin/pixvqa --metrics vmaf --hdr-transfer sdr --fps 50 \
  -r reference.yuv -t distorted.yuv -s 1920x1080 -f 420p:10

# Explicit model override:
./bin/pixvqa --metrics vmaf --model model/vmaf_v1.0.16_3d0h.json \
  -r reference.yuv -t distorted.yuv -s 1920x1080 -f 420p:8
```

### Optional learned metrics

Install one Python environment for **ColorVideoVDP**, **CompressedVQA-HDR NR**, and **SimpleVQA** — supply upstream repos and checkpoints separately (PixVQA does not download weights). Use `pixvqa --metrics simplevqa` for SDR compressed video, or `python -m pixvqa_simplevqa` for the reference module. See [Learned Metrics](LEARNED_METRICS.md) and [SIMPLEVQA_REFERENCE.md](dev/SIMPLEVQA_REFERENCE.md).

```bash
# ColorVideoVDP (full-reference):
./bin/pixvqa --metrics cvvdp -r reference.mp4 -t distorted.mp4 \
  --cvvdp-interpretation pq --cvvdp-display standard_hdr_pq

# CompressedVQA-HDR NR (no reference; compressed SDR or HDR10 only):
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4
```

CVVDP accepts compressed, raw 10-bit YUV420, and mixed raw/compressed inputs; raw input needs geometry, frame rate and appropriate color interpretation. CVQA-HDR NR does not accept raw YUV or HLG; default sampling is approximately 1 Hz. Both accept `--frames`. When native metrics are selected, learned metrics use the successfully compared frame prefix.

For high-resolution or concurrent CVVDP runs, check GPU memory and utilization; see the [GPU usage warning](LEARNED_METRICS.md#setup).

## Metric groups

Groups select several metrics and transfer/display defaults. A group can be combined with individual metric names, but not with a different group. All selected backends must be installed.

| Group | Includes | Interpretation |
|---|---|---|
| `sdr` | PSNR, XPSNR, SSIM, MS-SSIM, VMAF, CVVDP, CVQA-HDR NR | SDR; specify an SDR CVVDP display (e.g. `standard_fhd`) |
| `pq` | SDR group metrics plus PU21-PSNR/SSIM/MS-SSIM | PQ/BT.2020 non-constant luminance, limited range; HDR CVVDP display |
| `hlg` | PSNR, XPSNR, SSIM, MS-SSIM, PU21-PSNR/SSIM/MS-SSIM, CVVDP | HLG/BT.2020 non-constant luminance, limited range; no VMAF or NR |

Metric groups assign interpretation; they do not convert pixel values. See [Metrics & Formats](METRICS_AND_FORMATS.md).

## Compatible inputs

| Metric | Raw YUV | Compressed video | `--frames` |
|---|---|---|---|
| Native metrics | Planar 8/10-bit 4:2:0, 4:2:2, 4:4:4 | With FFmpeg build | Yes |
| `cvvdp` | 10-bit 4:2:0 with declared format/rate | Yes, including mixed pairs | Yes |
| `cvqa-hdr-nr` | No | SDR or HDR10 only | Yes |

Full-reference metrics compare matching frames in presentation order, stopping at the **shorter** input with a warning. Decode/I/O errors, truncated raw frames, and incompatible geometry remain errors. An NR-only invocation ignores `-r`.

## Command-line options

This is the everyday option reference. `--help-advanced` and `--help-hdr` list additional per-input HDR controls and compatibility aliases.

| Option | Purpose / default |
|---|---|
| `--metrics LIST` | Required; comma-separated metrics (`psnr,xpsnr,ssim,msssim,vmaf,pu21-psnr,pu21-ssim,pu21-msssim,cvvdp,cvqa-hdr-nr`) or `all,sdr,pq,hlg`. Repeatable. |
| `-r PATH`, `-t PATH` | Reference (full-reference metrics) and required distorted video. |
| `--input auto\|raw\|compressed` | Input-type override for subsequent inputs; default `auto`. |
| `-s WxH`, `-f 420p\|422p\|444p[:8\|:10]` | Raw-video size and planar format; omitted bit depth means 10. |
| `--fps N\|N/D` | Override frame rate; required when the input cannot supply a reliable constant rate for XPSNR/VMAF auto-selection. |
| `-n N`, `--frames N` | Maximum evaluated frames across selected metrics; default: common available duration. |
| `-v 0..9` | Verbosity; `0` summary, `1` per-frame plus summary (default `1`). |
| `--threads auto\|N` | Native-metric worker threads; default `auto`. |
| `--decode auto\|cpu\|gpu` | Native decoder; default `auto`. GPU requires NVDEC. |
| `--device auto\|cpu\|gpu` | Metric device preference; default `auto`; VMAF uses CPU libvmaf. |
| `--model PATH\|auto` | VMAF model; default `auto`. Explicit model overrides selection. |
| `--hdr-transfer auto\|pq\|hlg\|sdr\|linear` | Transfer interpretation; default `auto`. |
| `--hdr-primaries`, `--hdr-matrix`, `--hdr-range` | Color interpretation overrides; values in `--help-hdr`. |
| `--hdr-chroma-loc`, `--hdr-chroma-up`, `--hdr-peak-nits`, `--hdr-black-nits`, `--hdr-hlg-gamma` | Optional HDR/display settings; values in `--help-hdr`. |
| `--cvvdp-interpretation sdr\|pq\|hlg`, `--cvvdp-display NAME` | CVVDP interpretation/display; otherwise resolved where possible from group, options and metadata. |
| `--cvvdp-gpu-mem GB` | Optional positive CUDA memory budget (decimal GB). |
| `--cvvdp-python PATH`, `--cvvdp-bin PATH`, `--cvvdp-output-json PATH` | CVVDP interpreter, executable, result output overrides. |
| `--cvqahdr-python PATH`, `--cvqahdr-repo PATH`, `--cvqahdr-checkpoint PATH`, `--cvqahdr-profile PATH`, `--cvqahdr-output-json PATH` | NR interpreter, model assets and result output. |
| `--cvqa-nr-sample-interval-sec N`, `--cvqa-nr-chunk-size-sec N` | NR sampling interval (default 1 second) and inference chunk (default 8 seconds). |
| `--learned-config PATH` | Optional learned-metric TOML file; auto-loads `config/learned_metrics.local.toml` when present. |
| `--qcs`, `--qcs-range metric=lower:upper` | Enable QCS; override metric interval (repeatable/comma-separated). See [QCS](QCS.md). |
| `--help`, `--help-advanced`, `--help-hdr`, `--version` | Help and version. |

Legacy aliases (`--cs`, `--tf`, `--ref-cs`, `--dist-cs`, `--ref-tf`, `--dist-tf`, `--hlg-peak`) remain accepted; avoid mixing conflicting aliases and modern options.

**Exit codes:** `0` success; `2` configuration/preflight error; `1` runtime/backend failure.
