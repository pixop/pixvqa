# CompressedVQA-HDR NR

`cvqa-hdr-nr` is an optional no-reference backend for the official
CompressedVQA-HDR NR model. PixVQA invokes an isolated Python interpreter as a
safe argv subprocess; the native build has no PyTorch dependency. It can run
alone or together with native metrics and `cvvdp`; see
[mixed execution](CLI_USAGE.md#mixed-native-and-learned-metrics).

The adapter imports the unmodified `NR/NR_model.py` and `NR/VQA_NR.py` from an
upstream checkout pinned to `76bff6d5c078d51be06eddf1ed11591f3678145a`. It
builds the SigLIP2 model with `pretrained=False` and strictly loads the supplied
checkpoint. The default sampling interval is one second; PixVQA additionally
supports configurable frame sampling and chunked inference. Selected frames are
processed independently in batches, their raw scores are averaged across the
whole video, and the upstream logistic calibration is applied once to that mean.
At the default one-second interval, the adapter uses upstream video
preprocessing; at other intervals it follows a custom OpenCV/PIL frame-selection
path with the same spatial transforms. It imports the upstream modules with
bytecode writing disabled, so the checkout gains no `__pycache__`.

## Invocation

```bash
pixvqa --metrics cvqa-hdr-nr \
  -t distorted.mp4 \
  --cvqahdr-python /home/me/venvs/cvqahdr/bin/python \
  --cvqahdr-repo /home/me/src/CompressedVQA-HDR \
  --cvqahdr-checkpoint /home/me/models/NR_HDR_VQA.pth \
  --cvqahdr-profile /home/me/models/NR_HDR_VQA.npy \
  --cvqa-nr-sample-interval-sec 1 \
  --cvqa-nr-chunk-size-sec 8 \
  --cvqahdr-output-json result.cvqahdr.json
```

`-r` is not needed. When `cvqa-hdr-nr` is the only selected metric, a supplied
`-r` is ignored with a `pixvqa: info:` line on stderr; the reference path is
not opened or validated and is not passed to the backend. When a full-reference
metric (any native metric or `cvvdp`) is also selected, `-r` is required and
the NR backend still never receives it.

The metric accepts compressed video only. Raw YUV (a `.yuv` test path, or a
raw test input detected by a native run) is rejected. For NR-only runs, supplied raw-video options (`-s`, `-f`, `--fps`) are
accepted but ignored with an informational message. Other shared or native
options may be accepted for command-line compatibility but do not configure
the NR model; in particular, HDR metadata options do not change the upstream
OpenCV preprocessing. `--frames` is rejected whenever `cvqa-hdr-nr` is
selected: NR evaluates the full input clip.
`--device cpu` (alias of `--metric-device`) runs the model on the CPU,
`--device gpu` requires CUDA and fails in preflight without it, and the
default `auto` uses CUDA when PyTorch sees a device, else the CPU with a
notice. The summary line and JSON record report the effective device. CPU and
CUDA scores agree within 1e-6; see [LEARNED_DEVICES.md](LEARNED_DEVICES.md).

The interpreter is selected with `--cvqahdr-python`, then
`PIXVQA_CVQAHDR_PYTHON`, then `[cvqa-hdr-nr] python`, then `[learned] python`
in the [learned-metric config](LEARNED_METRICS_CONFIG.md), then `python3`. The
repository, checkpoint, and profile are required, either as options or as the
`[cvqa-hdr-nr]` keys `repository`, `checkpoint`, and `profile`; an option
overrides the config. With a config in place, the command reduces to
`pixvqa --metrics cvqa-hdr-nr -t distorted.mp4`. With no
`--cvqahdr-output-json`, the native launcher creates and removes a private JSON
record.

## Required Python environment

The adapter itself has no dependencies. Its selected interpreter must contain
the official model runtime, including PyTorch, torchvision, OpenCV, Pillow,
open_clip, and timm. The upstream repository does not pin open_clip or timm;
the validated target stack is PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128,
open_clip_torch 3.1.0, timm 1.0.19, OpenCV 4.11.0.86, Pillow 11.3.0, and NumPy
2.5.2 on Python 3.12. This stack supports CUDA 12.8 and RTX 5080. The shared
[learned-metric environment](LEARNED_ENVIRONMENT.md) pins exactly this stack
together with CVVDP.

The checkpoint and calibration profile are external model assets. PixVQA never
downloads them or any pretrained weights. The adapter sets no download flags;
the `pretrained=False` model construction must remain offline.

Install only the dependency-free PixVQA adapter package into that already
prepared environment:

```bash
/home/me/venvs/cvqahdr/bin/python -m pip install -e /path/to/pixvqa/python
```

For an uninstalled source checkout only, set
`PYTHONPATH=/path/to/pixvqa/python` when running `pixvqa`.

## Preflight

`python -m pixvqa_cvqahdr --check ...` validates without inference: the video,
checkpoint, and profile exist and are readable; the upstream checkout contains
`NR/NR_model.py` and its commit and clean-tree status can be read; OpenCV,
NumPy, PyTorch, torchvision, and open_clip import; CUDA is available when
requested; the profile has shape `(4,)` and finite values; the upstream modules
import; and OpenCV metadata satisfies the sampling constraints below. In mixed
runs PixVQA runs this check for every learned backend before any metric runs.
Preflight failures exit 2; inference failures exit 1.

## Configurable sampling and chunking

- `--cvqa-nr-sample-interval-sec N` sets the requested time between sampled
  frames, in seconds (default **1**). For example, `1`, `0.5`, and `0.25`
  select approximately 1, 2, and 4 samples/s.
- `--cvqa-nr-chunk-size-sec N` sets the nominal video duration represented by
  each GPU inference batch (default **8** seconds). The adapter converts this
  to a batch size of sampled frames. Smaller chunks generally reduce peak
  inference VRAM; larger chunks may improve GPU batching efficiency.

Both values must be finite and positive. Sampling is implemented in integer
frame steps: `interval_frames = max(1, floor(N * round(opencv_fps) + 0.5))`.
The adapter selects the midpoint of each *complete* interval, starting at
`interval_frames // 2`, and plans `frame_count // interval_frames` samples.
The actual interval is `interval_frames / opencv_fps`; the trailing incomplete
interval is not analyzed. The default one-second path uses upstream
preprocessing; nondefault intervals use adapter frame selection.

The adapter currently decodes and prepares the selected frames for the whole
prerecorded clip **before** running sequential GPU chunks. Chunking bounds the
GPU inference batch size, but does not yet provide streaming input, bounded
CPU-side frame buffering, or live score delivery. Faster-than-real-time
offline throughput must not be described as validated live operation.

Chunking changes the organization of inference, not the intended pooled metric:
frame predictions are averaged across **all** sampled frames before logistic
calibration. Each frame is analyzed independently; the model does not jointly
model motion across sampled frames. Per-frame numerical values may differ
slightly with batch size or device.

## Output

The JSON record keeps full-precision `quality_score` and `raw_model_score`,
the OpenCV sampling plan, requested/effective sampling interval, chunk settings,
model/preprocessing/inference timings, runtime versions, checkpoint and profile SHA-256
hashes, CUDA device information, and the upstream commit/clean-tree status.
Its `temporal` object holds one entry per sampled frame (`index`,
`frame_index`, `time_seconds`, `raw_model_score`, `quality_score`), the sample
count, the sampling density, and the parity and padding checks described
below.

Standalone runs (NR alone, no `--qcs`) print one compact JSON result on stdout,
unchanged from earlier releases. Python warnings, including open_clip's
"No pretrained weights loaded" notice, go to stderr. Mixed and QCS runs report
in the PixVQA text summary:

```text
[ CVQA-HDR-NR quality_score:7.40969689895428 raw_model_score:-0.1277415007352829 samples:2 frames:120 density:0.999/s ]
```

Scores are printed as the backend's shortest round-trip decimal text, so the
summary carries the same double as the JSON record.

The model uses OpenCV's decoder and its published 384-pixel resize/center crop
with ImageNet normalization. It does not explicitly process HDR metadata,
linearize PQ, or convert BT.2020. HDR10 decoder output therefore depends on
the selected OpenCV/FFmpeg build and is generally 8-bit decoded pixels.

## Optional GPU profiling

Set `PIXVQA_CVQA_NR_PROFILE=1` to enable per-chunk inference timings, processed
sample counts, peak PyTorch CUDA allocations/reservations, and periodic
`nvidia-smi` GPU-utilization/VRAM samples (when available). The information is
printed to stderr and included under `performance` in the persistent JSON
record (use `--cvqahdr-output-json`). The GPU-utilization and total-memory
measurements are **device-wide**, not exclusive to this metric; on multi-GPU
systems, the current sampler combines device rows rather than selecting only
the inference device. Profiling is disabled by default. The reported stage
timings do not by themselves constitute end-to-end wall time or live latency.

## Sampled-frame QCS

`--qcs` with `cvqa-hdr-nr` uses the default interval **[0, 10]**, the same as
`cvvdp`, so `--qcs-range` is not needed. `--qcs-range cvqa-hdr-nr=lower:upper`
overrides it. The interval is used only for QCS threshold integration: each
calibrated sample is mapped into it by the existing QCS integral. The pooled
`quality_score` and the per-sample calibrated scores are reported unchanged,
never clamped or rescaled.

The adapter observes the official `quality1` head with a forward hook; the
forward pass itself is not modified. The hook yields one float32 raw
prediction per sampled frame. Each is calibrated independently with the
official `logistic_func` and the float64 profile values. PixVQA feeds that
sequence into the same QCS integral as the native metrics:

```text
[ CVQA-HDR-NR-QCS value:... range:[0,10] samples:2 basis:sampled-frames ]
```

- **Video score unchanged.** `quality_score` is still `logistic(mean(raw))`,
  the official pooled score. The logistic is nonlinear, so the mean of the
  calibrated samples differs from it (4.44293 against 4.44497 on
  BasketballGame CRF40).
- **Parity check.** The adapter recomputes the official mean from the hooked
  samples and records `pooled_parity`. On a mismatch QCS is reported as
  `value:unavailable reason:pooled-parity-mismatch`.
- **No padded samples.** When OpenCV decodes fewer frames than it reported,
  upstream pads the buffer by repeating the last sample. In QCS runs the
  adapter counts decoded frames, and any padded sample makes QCS
  `value:unavailable reason:padded-samples`. Samples are never interpolated
  or repeated.
- **Sampled frames only.** QCS describes consistency only across selected
  frames (default approximately one sample per second), not all video frames.
  A 2-second clip yields two samples at the default interval.

Per-sample calibration extends the published model, which was trained and
validated as a video-level predictor. It is not an officially validated
per-frame quality predictor. Choose the QCS range for your own use.

## SDR and HLG

The upstream README lists the supported color spaces as HDR10 and SDR. There
is one NR checkpoint, `NR_HDR_VQA.pth`, and one calibration profile. The
official `video_processing` reads frames with OpenCV, converts BGR to RGB, and
resizes. It never reads a transfer, primaries, or matrix tag, and it does not
apply a PQ or HLG EOTF. PixVQA calls that function unchanged.

An SDR file is therefore scored as the picture OpenCV decoded, not
reinterpreted as PQ. The same checkpoint scores SDR and HDR10. `--metrics sdr`
includes `cvqa-hdr-nr` for that reason.

HLG is not a supported color space in the upstream README, and nothing in the
preprocessing distinguishes HLG from a mis-decoded picture. OpenCV's default
decode does not preserve HLG the way the HDR10 training frames were produced,
and this checkout has no HLG clip with a known score. `--metrics hlg` excludes
`cvqa-hdr-nr`. Selecting it beside that group is an error. `--metrics
cvqa-hdr-nr` on its own still runs the official preprocessor on whatever file
you pass; that is the published behavior, not an HLG mode.

## Validation

The official model has reported `7.4097` for
`BasketballGame_1080p_crf20.mp4` and `4.4450` for
`BasketballGame_1080p_crf40.mp4`. These four-decimal values are useful smoke
checks only. Compare the adapter's full-precision JSON value with a
full-precision official-script harness on the same environment before claiming
numerical parity. On the validated stack the unified CLI returns
`7.40969689895428` (raw `-0.1277415007352829`) and `4.444974268667124` (raw
`-1.1983554363250732`), and the hooked samples reproduce the pooled raw score
exactly.
