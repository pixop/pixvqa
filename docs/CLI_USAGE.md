# CLI Usage

Use `./bin/pixvqa` (shown below as `pixvqa`). PixVQA supports raw YUV and, in an FFmpeg-enabled build, compressed video. Compressed-input metadata normally supplies size, format and frame rate; raw YUV requires `-s` and `-f`.

## Examples

### Compressed video

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_hevc.mp4 -t samples/dist_hevc.mp4
```

### Raw YUV

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim --fps 60 \
  -r reference.yuv -t distorted.yuv -s 1920x1080 -f 420p:10
```

`--fps` is required for XPSNR when a reliable constant rate is unavailable (always for two raw inputs). Raw inputs must match the declared geometry and format.

### VMAF model selection

**PixVQA selects a bundled VMAF model automatically by default** (`--model auto`). Although libvmaf needs a model, PixVQA resolves its path from:

- **Transfer:** SDR or PQ, supplied by `--hdr-transfer`, the `sdr`/`pq` metric group, or reliable compressed-stream metadata. Untagged raw input needs an explicit transfer declaration; bit depth alone does not imply HDR.
- **Resolution:** exactly 1920×1080 (3H) or 3840×2160 (1.5H).
- **Frame rate:** known constant rate; selects the HFR variant above 30 fps. For raw YUV, specify `--fps`.

Automatic selection is unavailable for HLG, unsupported resolutions, missing/ambiguous transfer or unknown frame rate. **Pass `--model PATH`** to select a specific model or override automatic selection. `--vmaf-model` is a deprecated alias; if both are supplied they must agree. Explicit selection bypasses the automatic model-choice restrictions, **not** the model's format/transfer compatibility checks. Bundled HDR VMAF models require limited-range YUV420p10 PQ, BT.2020 primaries and BT.2020 non-constant-luminance matrix; PixVQA does not convert mismatched samples.

```bash
pixvqa --metrics vmaf --hdr-transfer sdr --fps 60 \
  -r reference.yuv -t distorted.yuv -s 1920x1080 -f 420p:10

# Explicit model, for example to override auto-selection:
pixvqa --metrics vmaf --model model/vmaf_v1.0.16_3d0h.json \
  -r reference.yuv -t distorted.yuv -s 1920x1080 -f 420p:8
```

### Optional Python-backed metrics

[Setup and model assets](LEARNED_METRICS.md) must be configured separately.

```bash
# Full-reference ColorVideoVDP: use explicit transfer/display when metadata is missing.
pixvqa --metrics cvvdp -r reference.mp4 -t distorted.mp4 \
  --cvvdp-interpretation pq --cvvdp-display standard_hdr_pq

# No-reference CompressedVQA-HDR NR: compressed test input only.
pixvqa --metrics cvqa-hdr-nr -t distorted.mp4

# Mix native and Python-backed metrics.
pixvqa --metrics psnr,vmaf,cvvdp,cvqa-hdr-nr --hdr-transfer pq \
  -r reference.mp4 -t distorted.mp4
```

ColorVideoVDP supports compressed input and raw YUV420p10le PQ/HLG (declare size/format and `--fps` for raw/raw pairs). SDR use requires a suitable `--cvvdp-display`. CompressedVQA-HDR NR supports compressed SDR or HDR10, not raw YUV or HLG; it ignores `--frames` with a warning and evaluates the full video. NR defaults to one sample per second and eight-second inference chunks.

### PQ / HLG PU21

```bash
pixvqa --metrics pu21-psnr,pu21-msssim --hdr-transfer pq \
  -r ref10.yuv -t dist10.yuv -s 1920x1080 -f 420p:10
pixvqa --metrics pu21-psnr,pu21-msssim --hdr-transfer hlg \
  -r ref10.yuv -t dist10.yuv -s 1920x1080 -f 420p:10
```

See [PU21](PU21.md) for HDR interpretation and display defaults.

## Metric groups

A group selects metric names and transfer-related defaults. A group can be combined with individual metrics, but not a different group. Conventional PSNR/SSIM/MS-SSIM/XPSNR measure encoded sample differences; HDR light-domain evaluation is provided by PU21.

| Group | Included metrics | Default interpretation and restrictions |
|---|---|---|
| `sdr` | PSNR, XPSNR, SSIM, MS-SSIM, VMAF, CVVDP, CVQA-HDR NR | SDR VMAF model auto-selected; specify an SDR `--cvvdp-display` (such as `standard_fhd`) |
| `pq` | Same as `sdr` plus PU21-PSNR/SSIM/MS-SSIM | PQ, BT.2020, BT.2020nc, limited range; HDR VMAF; CVVDP `standard_hdr_pq` |
| `hlg` | PSNR, XPSNR, SSIM, MS-SSIM, PU21-PSNR/SSIM/MS-SSIM, CVVDP | HLG, BT.2020, BT.2020nc, limited range; CVVDP `standard_hdr_hlg`; no VMAF/NR |

All group metrics must be installed and compatible. For mixed inputs with missing metadata, supply the appropriate HDR/color options; metric groups do not convert pixels.

## Compatible inputs

| Metric | Raw YUV | Compressed video | Frame cap (`-n` / `--frames`) |
|---|---|---|---|
| Native metrics | Yes | FFmpeg-enabled build (`swdec` / `full`) | Supported |
| `cvvdp` | 10-bit 4:2:0 PQ/HLG, with geometry and rate | Yes (also mixed raw/compressed) | Supported |
| `cvqa-hdr-nr` | No | SDR or HDR10 | Ignored (full clip) |

`-t` is always required; add `-r` for any full-reference metric. When only NR is selected, `-r` is ignored. Native metrics need FFmpeg support for compressed inputs; Python-backed metrics decode using their own dependencies.

## Command-line options

Options below were checked against `src/main.cpp` (`--help`, `--help-advanced`, `--help-hdr` and argument handling). Defaults refer to the current source, not libvmaf's standalone CLI.

| Option | Description / default |
|---|---|
| `--metrics LIST` | Required. Comma-separated names: `psnr,xpsnr,ssim,msssim,vmaf,pu21-psnr,pu21-ssim,pu21-msssim,cvvdp,cvqa-hdr-nr`; or `all,sdr,pq,hlg`. May be repeated. |
| `-r PATH`, `-t PATH` | Reference and distorted video. `-t` required; `-r` for full-reference metrics. |
| `--input auto|raw|compressed` | Input detection; default `auto`. Applies to following `-r`/`-t` side when placed after it. |
| `-s WxH`, `-f 420p|422p|444p[:8|:10]` | Raw-video geometry/format; format without depth defaults to 10-bit. Specify before both paths to apply to both sides. |
| `--fps N|N/D` | Frame rate for raw VMAF auto-selection and XPSNR when no reliable rate is available. |
| `-n N`, `--frames N` | Maximum frames for native/CVVDP; NR ignores with warning. |
| `-v 0|1` | Summary only / per-frame plus summary; default `1` (higher verbosity values are accepted). |
| `--threads auto|N` | Native metric threads; default `auto`. |
| `--decode auto|cpu|gpu` | Native decoding device; default `auto`; GPU needs NVDEC. |
| `--device auto|cpu|gpu` | Metric device preference; default `auto`. VMAF always runs on CPU. |
| `--model PATH|auto` | VMAF model; default `auto`. Explicit path overrides automatic selection. |
| `--hdr-transfer auto|pq|hlg|sdr|linear` | Transfer interpretation; default `auto`. |
| `--hdr-primaries`, `--hdr-matrix`, `--hdr-range` | Color overrides; use `--help-hdr` for accepted values and per-input forms. |
| `--hdr-chroma-loc`, `--hdr-chroma-up`, `--hdr-peak-nits`, `--hdr-black-nits`, `--hdr-hlg-gamma` | Additional HDR options; use `--help-hdr`. |
| `--cvvdp-interpretation sdr|pq|hlg` | Optional explicit CVVDP transfer; otherwise resolve from group/HDR options/metadata. |
| `--cvvdp-display NAME` | CVVDP display; auto default for PQ/HLG; specify for SDR when not otherwise resolved. |
| `--cvvdp-gpu-mem GB` | Optional positive decimal-GB CUDA budget. |
| `--cvvdp-python`, `--cvvdp-bin` | Override CVVDP Python or executable path. |
| `--cvvdp-output-json PATH` | Persist CVVDP result record. |
| `--cvqahdr-python`, `--cvqahdr-repo`, `--cvqahdr-checkpoint`, `--cvqahdr-profile` | NR interpreter, checkout, checkpoint and calibration; use configured defaults or provide paths. |
| `--cvqahdr-output-json PATH` | Persist NR result record. |
| `--cvqa-nr-sample-interval-sec N` | NR temporal interval in seconds; default `1`. |
| `--cvqa-nr-chunk-size-sec N` | NR chunk duration in seconds; default `8`. |
| `--learned-config PATH` | Optional TOML settings for Python-backed metrics; default `config/learned_metrics.local.toml` in checkout when present. |
| `--qcs` | Enable supported temporal Quality Consistency Scores. |
| `--qcs-range metric=lower:upper` | Configurable QCS range; comma-separated or repeated; requires `--qcs`. Defaults and semantics: [QCS](QCS.md). |
| `--tf`, `--cs`, `--ref-tf`, `--dist-tf`, `--ref-cs`, `--dist-cs`, `--hlg-peak`, `--vmaf-model` | Deprecated compatibility options; see `--help-advanced` and `--help-hdr`. Do not mix conflicting old/new options. |
| `-h`, `--help`, `--help-advanced`, `--help-hdr`, `--version` | Help and version. |

Exit codes: `0` success, `2` invalid setup/preflight, `1` runtime error. All selected metrics must complete for success.
