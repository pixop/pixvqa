# CLI Usage

Use `./bin/pixvqa` from the repository (or `pixvqa` if it is on your `PATH`). Choose one or more metrics with `--metrics` and supply the video to evaluate with `-t`. Full-reference (FR) metrics also need an undistorted reference with `-r`; no-reference (NR) metrics do not.

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

### VMAF Models

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

### Optional Learned Metrics

To use the optional Python-based metrics, run `bash scripts/install_learned_metrics.sh` once. It sets up the Python environments and obtains the model assets automatically when downloads are available. Native metrics do **not** require this step. See [Learned Metrics](LEARNED_METRICS.md).

**SDR video is not limited to SimpleVQA.** You can evaluate SDR using native metrics or other compatible learned metrics. SimpleVQA itself accepts only compressed SDR video. All supported metrics use the same `pixvqa` command.

```bash
# No-reference SDR quality:
./bin/pixvqa --metrics simplevqa -t distorted.mp4

# No-reference SDR or HDR10 quality:
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4

# Full-reference ColorVideoVDP (opt-in, GPU-intensive):
./bin/pixvqa --metrics cvvdp --hdr-transfer pq -r reference.mp4 -t distorted.mp4

# HDR-ChipQA (opt-in, host-memory-intensive):
./bin/pixvqa --metrics hdr-chipqa -t distorted.mp4
```

**Resource limits:** CVVDP can use substantial **GPU VRAM**; HDR-ChipQA can use substantial **host RAM**, especially with several jobs in parallel. See [Performance considerations](LEARNED_METRICS.md#performance-considerations).

### Concurrent learned-metric jobs

For multiple independent PixVQA processes (for example, GNU Parallel jobs), the optional **global admission scheduler** coordinates the learned GPU backends and queues work when its resource reservations or concurrency caps would be exceeded. It is **off by default**. Enable it with the public CLI flag:

```bash
./bin/pixvqa --scheduler-admission --metrics simplevqa,cvqa-hdr-nr -t distorted.mp4
```

For existing batch scripts that cannot easily add a CLI flag to every invocation, the equivalent environment variable is `PIXVQA_SCHEDULER_ADMISSION=1` (inherited by subprocesses). The scheduler uses a per-user daemon, which starts automatically on first use. It does not limit the number of encoders or native PixVQA processes launched by GNU Parallel: it controls **admission of learned-metric inference** only. Reservations and defaults are not hard GPU-memory guarantees. Use `--scheduler-admission` for explicit user commands and the environment variable for automation. See [Metric resource scheduler](dev/METRIC_SCHEDULER.md) for limits, implementation and advanced settings.

## Metric Groups

A metric *group* selects several metrics in one command and sets the corresponding SDR/HDR interpretation. You can also select metrics individually or add one to a group (for example, `--metrics pq,cvvdp`).

| Group | Included metrics | Color interpretation |
|---|---|---|
| `sdr` | PSNR, XPSNR, SSIM, MS-SSIM, VMAF, CVQA-HDR NR | SDR |
| `pq` | SDR group's metrics + PU21-PSNR, PU21-SSIM, PU21-MS-SSIM | PQ / BT.2020, limited range |
| `hlg` | PSNR, XPSNR, SSIM, MS-SSIM, PU21-PSNR, PU21-SSIM, PU21-MS-SSIM | HLG / BT.2020, limited range |

**CVVDP and HDR-ChipQA are not part of the `sdr`, `pq`, or `hlg` groups** because they can require significant GPU VRAM or system RAM. Select them explicitly if needed. SimpleVQA can be added to `sdr`, but not to PQ/HLG groups.

**Note:** The legacy `all` selection is broader than these groups: it includes CVVDP but excludes HDR-ChipQA. To control resource use, prefer a specific group or an explicit metric list.

When CVVDP is added to an SDR, PQ, or HLG group, the group supplies its corresponding interpretation and default display preset (`standard_fhd`, `standard_hdr_pq`, or `standard_hdr_hlg`, respectively). Explicit CVVDP settings are checked for compatibility; they do not silently replace incompatible group settings. See `--help-hdr` for related controls.

Groups interpret samples; they do not convert video pixel values. With `--qcs`, eligible frame-based metrics produce QCS results; pooled-video metrics such as HDR-ChipQA and SimpleVQA do not. See [QCS](QCS.md) and [Metrics & Formats](METRICS_AND_FORMATS.md).

## Compatible Inputs

| Metric | Raw YUV | Compressed video | `--frames` |
|---|---|---|---|
| Native metrics | Planar 8/10-bit 4:2:0, 4:2:2, 4:4:4 | With FFmpeg build | Yes |
| `cvvdp` | 10-bit 4:2:0 with declared format/rate | Yes, including mixed pairs | Yes |
| `cvqa-hdr-nr` | No | SDR or HDR10 only | Yes |
| `simplevqa` | No | Compressed SDR only (PQ/HLG rejected) | Yes |
| `hdr-chipqa` | YUV420 8/10-bit (`-s`, `-f`, `--fps` for raw PQ/HLG) | Yes (SDR, PQ; HLG experimental) | Yes |

Full-reference metrics compare matching frames in presentation order, stopping at the **shorter** input with a warning. Decode/I/O errors, truncated raw frames, and incompatible geometry remain errors. An NR-only invocation ignores `-r`.

## Command-Line Options

Common options are listed below. For less frequently used backend, HDR, and compatibility settings, use `./bin/pixvqa --help-advanced` or `./bin/pixvqa --help-hdr`.

| Option | Purpose / default |
|---|---|
| `--metrics LIST` | Required; comma-separated metrics (`psnr,xpsnr,ssim,msssim,vmaf,pu21-psnr,pu21-ssim,pu21-msssim,cvvdp,cvqa-hdr-nr,simplevqa,hdr-chipqa`) or `all,sdr,pq,hlg`. Repeatable. |
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
| `--learned-config PATH` | Advanced: custom learned-metric configuration; normally detected automatically. |
| `--qcs`, `--qcs-range metric=lower:upper` | Enable QCS; override metric interval (repeatable/comma-separated). See [QCS](QCS.md). |
| `--help`, `--help-advanced`, `--help-hdr`, `--version` | Help and version. |

For backend-specific settings and legacy option aliases, see `--help-advanced` and `--help-hdr`.

**Exit codes:** `0` success; `2` configuration/preflight error; `1` runtime/backend failure.
