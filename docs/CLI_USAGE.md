# Command-line Usage

PixVQA supports raw YUV and compressed video. For compressed inputs, resolution and pixel format are detected automatically; raw YUV requires `-s` (resolution) and `-f` (pixel format). Reference and distorted inputs must have matching properties.

XPSNR requires `--fps` unless a reliable constant frame rate can be determined from the compressed inputs. GPU acceleration requires an NVIDIA-enabled build.

The examples use `pixvqa` to refer to `./bin/pixvqa` or an executable on `PATH`. The `samples/` directory contains short Big Buck Bunny excerpts.

**Compressed Reference and Distorted Video**

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_hevc.mp4 \
  -t samples/dist_hevc.mp4
```

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_av1.mp4 \
  -t samples/dist_av1.mp4
```

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim \
  -r samples/ref_vvc.mp4 \
  -t samples/dist_vvc.mp4
```

**Uncompressed Reference and Compressed Distorted Video**

```bash
pixvqa --metrics psnr,ssim \
  -r reference.yuv --input raw -s 1920x1080 -f 420p:8 \
  -t distorted.mp4
```

**Uncompressed Reference and Distorted Video**

XPSNR requires `--fps` because raw YUV does not contain frame-rate metadata.

```bash
pixvqa --metrics psnr,xpsnr,ssim,msssim --fps 24 --threads 1 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

**VMAF**

VMAF requires an explicit model selection using `--model`. Bundled VMAF models are available in the `model/` directory.

```bash
pixvqa --metrics vmaf --model model/vmaf_v1.0.16_3d0h.json --threads 1 \
  -r reference.yuv -s 1920x1080 -f 420p:8 \
  -t distorted.yuv -s 1920x1080 -f 420p:8
```

The bundled HDR VMAF v1.1.8 models assume limited-range 10-bit 4:2:0 PQ with
BT.2020 primaries and the BT.2020 non-constant-luminance matrix. PixVQA applies
that model configuration automatically; ensure both inputs really have those
characteristics. Contradictory metadata or HDR CLI options produce a warning and
do not convert the samples.

```bash
pixvqa --metrics vmaf \
  --model model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

**ColorVideoVDP (optional external backend)**

CVVDP is installed separately from PixVQA. It runs as `cvvdp` in `--metrics`,
alone or [mixed with other metrics](#mixed-native-and-learned-metrics).
It requires explicit sample interpretation and an official display model; PixVQA
does not infer HDR parameters for it. Raw input in this phase is limited-range
planar YUV420p10le PQ, passed with `-s`, `-f 420p:10`, and `--fps`.

```bash
pixvqa --metrics cvvdp \
  -r reference.mkv \
  -t distorted.mkv \
  --cvvdp-interpretation pq \
  --cvvdp-display standard_hdr_pq \
  --cvvdp-output-json result.cvvdp.json
```

```bash
pixvqa --metrics cvvdp \
  -r reference.yuv \
  -t reconstructed.yuv \
  -s 1920x1080 -f 420p:10 --fps 60 \
  --cvvdp-interpretation pq \
  --cvvdp-display standard_hdr_pq
```

See [COLORVIDEOVDP.md](COLORVIDEOVDP.md) for isolated installation, supported
displays, the JSON result record, and incompatible native options.

**CompressedVQA-HDR NR (optional external backend)**

CompressedVQA-HDR NR is installed separately from PixVQA. It runs as
`cvqa-hdr-nr` in `--metrics`, alone or mixed with other metrics. It is
no-reference, so it requires only `-t`, and it accepts compressed video only.
The official implementation's OpenCV sampling constraints are validated before
inference. Raw YUV and `--frames` are rejected. When it is the only metric, a
supplied `-r` is ignored with an informational message and is never opened.

```bash
pixvqa --metrics cvqa-hdr-nr \
  -t distorted.mp4 \
  --cvqahdr-python /home/me/venvs/cvqahdr/bin/python \
  --cvqahdr-repo /home/me/src/CompressedVQA-HDR \
  --cvqahdr-checkpoint /home/me/models/NR_HDR_VQA.pth \
  --cvqahdr-profile /home/me/models/NR_HDR_VQA.npy \
  --cvqahdr-output-json result.cvqahdr.json
```

See [COMPRESSEDVQA_HDR.md](COMPRESSEDVQA_HDR.md) for runtime pins, output
metadata, clip-duration constraints, sampled-frame QCS, and validation.

### Mixed native and learned metrics

Native metrics, `cvvdp`, and `cvqa-hdr-nr` can be selected together, in one
`--metrics` list or across repeated `--metrics` options:

```bash
pixvqa --metrics psnr,vmaf,cvvdp,cvqa-hdr-nr \
  --model model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json \
  -r reference.mp4 -t distorted.mp4 \
  --cvvdp-interpretation pq --cvvdp-display standard_hdr_pq \
  --cvqahdr-repo /home/me/src/CompressedVQA-HDR \
  --cvqahdr-checkpoint /home/me/models/NR_HDR_VQA.pth \
  --cvqahdr-profile /home/me/models/NR_HDR_VQA.npy
```

The `--cvqahdr-*` and interpreter options can be omitted when
`config/learned_metrics.local.toml` provides them; see
[learned-metric config](LEARNED_METRICS_CONFIG.md).

**Reference rules.** `-t` is always required. `-r` is required when any
full-reference metric is selected: every native metric and `cvvdp`. If
`cvqa-hdr-nr` is the only metric, a supplied `-r` is ignored with a
`pixvqa: info:` message on stderr and is never opened.

**Compatible inputs.** Each selected metric must accept the inputs. Every
check finishes before any metric runs.

| Metric | Raw YUV | Compressed video | `--frames` |
|---|---|---|---|
| Native | Yes | `make swdec` or `make full` only | Yes |
| `cvvdp` | Both inputs raw YUV420p10le PQ with `-s`, `-f 420p:10`, `--fps` | Both inputs compressed; decoded by pycvvdp | Yes |
| `cvqa-hdr-nr` | No | Yes; decoded by OpenCV | No |

`cvvdp` rejects one raw input mixed with one compressed input. The `make cpu`
and `make cuda` builds have no FFmpeg decoder. Native metrics on compressed
input therefore need `make swdec` or `make full`. The learned backends decode
in their own Python environments, so any build can run `cvvdp` and
`cvqa-hdr-nr` on compressed video when no native metric is selected.
Native-only options, such as `--decode`, `--threads`, `-v`, `--model`, and the
HDR options, are rejected when no native metric is selected.

**Execution.** PixVQA validates every option and format, then runs each
learned backend's `--check` preflight: interpreter, imports, model assets,
video metadata, and NR sampling constraints. Only then does any metric run.
Native metrics run first and keep their per-frame output. CVVDP runs next,
then NR. One summary follows:

```text
Summary:

[ PSNR  y:... ]
[ CVVDP jod:7.415657997131348 frames:120 display:standard_hdr_pq interpretation:pq ]
[ CVQA-HDR-NR quality_score:4.444974268667124 raw_model_score:-1.1983554363250732 samples:2 frames:120 density:0.999/s ]
[ PSNR-QCS value:... range:[30,50] ]
[ CVVDP-QCS value:74.247569 range:[0,10] frames:120 basis:per-frame-jod-equivalent ]
[ CVQA-HDR-NR-QCS value:... range:[0,10] samples:2 basis:sampled-frames ]

Total frames:120  fps:...
```

Learned scores are printed with full precision, as the shortest decimal text
that round-trips to the same double. CVVDP coverage is `frames:N`: the test
length, capped by `--frames`. NR coverage gives the sample count, the OpenCV
frame count, and the sampling density. Provenance, such as runtime versions,
hashes, commands, and sample timestamps, stays in each backend's JSON record.
Records are temporary unless `--cvvdp-output-json` or `--cvqahdr-output-json`
is given. A metric that fails prints `status:failed`, for example
`[ CVVDP status:failed ]`. If a native metric fails during the frame loop, its
per-frame lines stay, the learned metrics still run, and the summary starts
with `[ NATIVE status:failed ]`.

**Output streams.** Reports go to stdout. Backend warnings, progress, and
diagnostics go to stderr; PixVQA forwards them as they arrive and drains both
pipes concurrently. A single learned metric without `--qcs` keeps its
standalone compact JSON line on stdout. Every other combination prints the
text report.

**Exit status.**

- 0: every selected metric completed.
- 2: invalid configuration, incompatible inputs, or failed preflight. No metric
  ran.
- 1: a runtime failure. Completed results are still printed.

**PU21 Metrics for HDR Video (PQ and HLG)**

PU21 metrics account for HDR transfer functions, unlike conventional metrics that operate on encoded sample values. For PQ and HLG, PixVQA defaults to BT.2020 primaries, BT.2020nc matrix, and limited range when these are unspecified. HLG defaults to a 1000 cd/m² display peak; PQ requires no display peak. See the [PU21 guide](PU21.md) for details.

```bash
pixvqa --metrics pu21-psnr,pu21-ssim --hdr-transfer pq \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

```bash
pixvqa --metrics pu21-psnr,pu21-msssim --hdr-transfer hlg \
  -r reference10.yuv -s 1920x1080 -f 420p:10 \
  -t distorted10.yuv -s 1920x1080 -f 420p:10
```

**GPU Decoding (NVDEC)**

Decode the compressed HEVC 10-bit 4:2:0 video using NVIDIA NVDEC while reading the reference as raw YUV.

```bash
pixvqa --metrics psnr --decode gpu \
  -r reference10.yuv --input raw -s 1920x1080 -f 420p:10 \
  -t reference10.hevc.mp4
```

### HDR master command

`--preset hdr` is the single command for two HDR10/PQ inputs. With no
`--metrics` list it runs every supported metric, in the usual order: native,
then `cvvdp`, then `cvqa-hdr-nr`. It fills options the existing pipeline
already understands. It does not change metric math.

| What it sets | Value |
|---|---|
| Metrics | `psnr`, `xpsnr`, `ssim`, `msssim`, `vmaf`, `pu21-psnr`, `pu21-ssim`, `pu21-msssim`, `cvvdp`, `cvqa-hdr-nr` |
| Transfer | `pq`, BT.2020 primaries, `bt2020nc`, and limited range, unless you already passed those |
| VMAF model | `model/hdr_vmaf_v1.1.8/hdr_vmaf_v1.1.8_3d0h.json` when `--model` is omitted. Run from the repository root, or pass `--model` |
| CVVDP | `--cvvdp-interpretation pq` and `--cvvdp-display standard_hdr_pq` when omitted |
| Learned QCS interval | `[0, 10]` for both `cvvdp` and `cvqa-hdr-nr` (the default for every run, not only the preset) |

`--metrics all` selects the same metrics without the HDR defaults.
`--preset hdr --metrics psnr,pu21-psnr` keeps your list and still applies the
HDR defaults. A non-PQ transfer or `--cvvdp-interpretation sdr` is an error. `--frames` is an error because NR reads the whole
clip. Native metrics on compressed video need `make swdec` or `make full`,
and `vmaf` needs a build with libvmaf. PSNR, XPSNR, SSIM, and MS-SSIM still
score stored sample codes. PU21, HDR VMAF, and CVVDP do the HDR processing.

`--qcs` adds one QCS line per metric to the same summary. Pass `-v 0` when
you want the summary without per-frame native lines.

With `config/learned_metrics.local.toml` in place, the whole HDR10 run is:

```bash
./bin/pixvqa --preset hdr --qcs -v 0 \
  -r reference_hdr.mp4 \
  -t distorted_hdr.mp4
```

Each metric keeps its own summary line. The block under `Summary:` is the
consolidated result. Backend warnings stay on stderr.

### Metric groups

A group name in `--metrics` selects one transfer's metric set and fills that
transfer's colorimetry and display. A comma-separated list of metric names
still selects only those metrics. A group can be combined with extra metric
names, but not with another group. PSNR, XPSNR, SSIM, and MS-SSIM still score
stored sample codes; the transfer applies to PU21, VMAF, and CVVDP.

| Group | Metrics | What it fills | What it refuses |
|---|---|---|---|
| `sdr` | `psnr`, `xpsnr`, `ssim`, `msssim`, `vmaf`, `cvvdp`, `cvqa-hdr-nr` | SDR VMAF model `model/vmaf_v1.0.16_3d0h.json`; `--cvvdp-interpretation sdr`. `--cvvdp-display` (`standard_fhd` or `standard_4k`) is required | Every PU21 metric. It does not set `--hdr-transfer sdr`, so a PQ or HLG stream is not rescored as SDR |
| `pq` | `psnr`, `xpsnr`, `ssim`, `msssim`, `vmaf`, `pu21-psnr`, `pu21-ssim`, `pu21-msssim`, `cvvdp`, `cvqa-hdr-nr` | Transfer `pq`, BT.2020 primaries, `bt2020nc`, limited range; HDR VMAF model; `--cvvdp-interpretation pq`, `--cvvdp-display standard_hdr_pq` | A transfer, primaries, matrix, range, CVVDP interpretation, or CVVDP display from another family |
| `hlg` | `psnr`, `xpsnr`, `ssim`, `msssim`, `pu21-psnr`, `pu21-ssim`, `pu21-msssim`, `cvvdp` | Transfer `hlg`, BT.2020 primaries, `bt2020nc`, limited range. PU21 keeps its 1000 cd/m² HLG peak. `--cvvdp-interpretation hlg`, `--cvvdp-display standard_hdr_hlg` (upstream model: 1500 cd/m², BT.2020-HLG) | `vmaf` (the installed HDR VMAF model is PQ-only) and `cvqa-hdr-nr` (HDR10 and SDR only) |

HLG is never run through the PQ or SDR display. `--preset hdr` combines with
`--metrics pq` only. The VMAF model is looked up in the working directory and
then in the checkout that holds the executable. `--model` overrides it.

```bash
./bin/pixvqa --metrics sdr --cvvdp-display standard_fhd --qcs \
  -r reference_sdr.mp4 -t distorted_sdr.mp4
./bin/pixvqa --metrics hlg -r reference_hlg.mp4 -t distorted_hlg.mp4
```

## Command-line Options

`pixvqa --help` lists the options. `pixvqa --help-hdr` lists the HDR options and defaults. QCS is documented in [QCS.md](QCS.md). PQ and HLG setup is documented in [PU21.md](PU21.md).

| Option | Purpose |
|---|---|
| `-r`, `-t` | Reference path and distorted path. `-t` is always required. `-r` is required by full-reference metrics and ignored when only `cvqa-hdr-nr` is selected |
| `--metrics` | Comma-separated metric names. Native, `cvvdp`, and `cvqa-hdr-nr` can be mixed; repeated options accumulate. `all` selects every metric. `sdr`, `pq`, and `hlg` are [metric groups](#metric-groups) |
| `--preset hdr` | HDR10/PQ master run. See [HDR master command](#hdr-master-command) |
| `--decode` | `auto`, `cpu`, or `gpu`. Default `auto` |
| `--metric-device`, `--device` | `auto`, `cpu`, or `gpu`. Default `auto`. The two spellings are aliases and must agree when both are given. `gpu` never falls back to the CPU, except VMAF, which has no GPU path. Learned metrics resolve `auto` to CUDA when PyTorch sees a device, else CPU; see [LEARNED_DEVICES.md](LEARNED_DEVICES.md) |
| `--threads` | `auto` or a positive integer. Default `auto` |
| `-s WxH` | Luma size. Required for raw YUV. Checked when set on compressed video |
| `-f` | `420p`, `422p`, or `444p`, with `:8` or `:10`. Required for raw YUV |
| `--input` | `auto`, `raw`, or `compressed`. Default `auto`. `raw` does not probe |
| `--fps` | `N` or `N/D`. Required for `xpsnr` when no constant compressed rate is available. Decimals truncate toward zero |
| `--model` | VMAF model JSON. No default |
| `--cvvdp-interpretation` | Required for CVVDP: `sdr`, `pq`, or `hlg`; never inferred from the video. A group or `--preset hdr` fills it |
| `--cvvdp-display` | Required official CVVDP display model. `pq` fills `standard_hdr_pq`; `hlg` fills `standard_hdr_hlg` |
| `--cvvdp-python` | Interpreter for the adapter. Overrides `PIXVQA_CVVDP_PYTHON` and the config |
| `--cvvdp-bin` | Official CVVDP executable. Overrides `[cvvdp] executable` |
| `--cvvdp-gpu-mem GB` | Optional CVVDP CUDA-memory cap in decimal GB. It is passed as upstream `--gpu-mem`, which limits its frame-block budget to `min(cap, free memory minus its safety reserve)`. Omit it to preserve upstream's free-memory heuristic |
| `--cvvdp-output-json` | Persistent CVVDP machine-readable run record |
| `--cvqahdr-python` | Interpreter for CompressedVQA-HDR NR. Overrides `PIXVQA_CVQAHDR_PYTHON` and the config |
| `--cvqahdr-repo` | Upstream CompressedVQA-HDR checkout. Required unless set in the config |
| `--cvqahdr-checkpoint` | NR checkpoint. Required unless set in the config |
| `--cvqahdr-profile` | NR calibration profile. Required unless set in the config |
| `--cvqahdr-output-json` | Persistent CompressedVQA-HDR NR run record |
| `--learned-config` | Learned-metric TOML config. Default `config/learned_metrics.local.toml` in the executable's checkout. See [learned-metric config](LEARNED_METRICS_CONFIG.md) |
| `--hdr-transfer` | `pq`, `hlg`, `sdr`, `linear`, or `auto` |
| `-v` | `1` prints each frame and the summary. `0` prints the summary. Default `1` |
| `--qcs` | Enable QCS for every selected metric. `cvvdp` uses per-frame JOD-equivalents and `cvqa-hdr-nr` uses sampled frames |
| `--qcs-range` | `metric=lower:upper`, comma-separated and repeatable, with lower < upper. `cvvdp` and `cvqa-hdr-nr` default to [0, 10]; the interval never clamps reported scores |
