# Learned-metric reproducibility investigation (2026-11-09)

Tracking issue: https://github.com/pixop/pixvqa/issues/24

## Background

On an i9-9900K with RTX 4070 and PyTorch 2.4.1+cu121, repeated scoring
of the same Aerial 1080p60 HEVC 10 Mb/s file (1199 frames, 24,573,997 bytes)
gave identical native VMAF/XPSNR but variable learned-metric outputs.

* CVQA-HDR-NR raw score switched between `0.6922507286071777` and
  `0.6880412697792053` (calibrated scores `9.085405712488637` and
  `9.077878369538553`).
* Prior to correction, SimpleVQA scores spanned ~0.00506 across three repeats.
  The upstream demo instantiated the pretrained SlowFast motion backbone in
  training mode. Calling `torch.no_grad()` does **not** turn off training
  behavior; the spatial quality regressor alone was explicitly set to eval.
* Commit d072d11 corrected SlowFast to `eval()` in all staged inference paths.
  Following that correction, three observed SimpleVQA scores were
  `3.543121337890625`, `3.542750835418701`, `3.543274402618408`
  (range ~0.000524). This is improved, **not yet deterministic**.
  Pre-fix and post-fix SimpleVQA scores are **not comparable** in RD analysis.

## Deterministic diagnostic mode (opt-in)

To characterize the remaining variance, set:

```bash
PIXVQA_LEARNED_DETERMINISTIC=1
```

For a subprocess-per-video evaluation, this is inherited by each backend. It
enables on both learned backends:

* fixed Python, NumPy, and PyTorch seed 0;
* `CUBLAS_WORKSPACE_CONFIG=:4096:8` when not already provided;
* `torch.backends.cudnn.benchmark = False` and
  `torch.backends.cudnn.deterministic = True`;
* `torch.use_deterministic_algorithms(True)`.

The CUDA setting must be applied before initializing CUDA; fresh metric
processes satisfy this more reliably than already-running daemons. **Restart
persistent or shared-GPU workers** when changing the setting. Shared workers
must not be reused across deterministic/ordinary comparisons. A deterministic
operator may be unsupported and raise an error. That is diagnostic information,
not a reason to silently fall back.

This mode is **not enabled by default** pending repeatability, score-parity and
performance validation on the target hardware. It may change the numerical
baseline or reduce throughput. Do not mix outputs across configurations.

## Recommended experiment

1. Use one immutable encoded file, not an encode-and-evaluate shell command.
   Record `sha256sum`, `ffprobe` stream properties, checkpoint hashes,
   backend modes and `git rev-parse HEAD`. Confirm that the source video path
   and checkpoint are identical across runs.
2. With no other benchmark jobs active, evaluate the **same encoded file** at
   least five times sequentially with normal settings (admission off).
   Store each full backend JSON and individual per-sample CVQA scores.
3. Repeat five times with `PIXVQA_LEARNED_DETERMINISTIC=1`, admission off.
   Run in fresh processes; use distinct output filenames.
4. Repeat with global admission on, still sequential and then with multiple
   jobs. Isolate the admission socket. Compare against (2)/(3).
5. Compare independent SimpleVQA, persistent-worker and shared-GPU-worker
   modes separately, restarting workers and capturing their backend metadata.
6. For CVQA, first determine whether the **19 individual sampled-frame raw
   values** change. If so, identify the first differing frame; if not, inspect
   pooling/calibration. An exactly two-state raw score pattern is not proven
   to be mere floating-point roundoff.
7. Separately test at low and high compression to establish quality
   sensitivity, only once per-file repeatability is understood.

### Interpretation

If deterministic mode fixes the variation on a quiet host, inspect the
runtime overhead and operator compatibility before making it default.
If nondeterminism persists even in deterministic mode, investigate upstream
implementations, preprocessing, checkpoint provenance, decoder behavior and
hardware/kernel details. GPU admission and queueing should change *when* a
backend runs, not *what* score a fixed decoded video receives. The temporal
association with recent scheduler changes is a hypothesis requiring an A/B
comparison, **not an established cause**.

## Code-level check

```bash
cd ~/codeccomp/metrics/pixvqa
PYTHONPATH=python python scripts/test_simplevqa_motion_eval.py
```

This validates staging/textual injection but **does not** demonstrate numerical
repeatability or compare models on the RTX 4070.

## SimpleVQA stage-by-stage trace (single-process / persistent-worker path)

To locate the first source of residual variation, PixVQA can hash the model
inputs, the SlowFast outputs for each motion window, the assembled motion
features and final score. This is an **opt-in diagnostic** that copies tensors
to host memory and can significantly slow execution. It does not change tensor
values, but should not be used for performance benchmarking. The shared-GPU
path is not instrumented by this trace yet.

```bash
cd ~/codeccomp/metrics/pixvqa
git pull --ff-only origin main
PYTHONPATH=python python scripts/test_simplevqa_motion_eval.py

# In the shell where the exact same single-video benchmark is invoked:
export PIXVQA_LEARNED_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PIXVQA_SIMPLEVQA_TRACE_DIR="$(mktemp -d /tmp/pixvqa-svqa-trace-XXXXXX)"
echo "$PIXVQA_SIMPLEVQA_TRACE_DIR"

# Execute the existing evaluation command three times sequentially, without
# concurrent jobs and without re-encoding the input; use the same MP4 each time.
# Then compare the traces:
PYTHONPATH="$HOME/codeccomp/metrics/pixvqa/python" \
  python "$HOME/codeccomp/metrics/pixvqa/scripts/compare_simplevqa_traces.py" \
  "$PIXVQA_SIMPLEVQA_TRACE_DIR"
```

One JSONL file is created per independent inference process. If the trace
contains only model input differences, investigate OpenCV and preprocessing.
If the inputs match but SlowFast features differ, investigate SlowFast GPU
kernels and checkpoint/state. If SlowFast outputs match but `score` differs,
investigate spatial regressor operations including the `torch.std` reduction.

A clean run means every trace file ends with a `score` stage. Do not mix
traces from different input videos, inference backends, or model checkpoints.
Verify the setting actually reached the child by checking `deterministic=true`
in trace output. Clear these environment variables afterward.

```bash
unset PIXVQA_LEARNED_DETERMINISTIC CUBLAS_WORKSPACE_CONFIG PIXVQA_SIMPLEVQA_TRACE_DIR
```

## FFmpeg BGR24 decoder change (2026-11-10)

The pinned 1920x1080 Aerial 10-bit HEVC MP4 exhibited nondeterministic
OpenCV 4.11.0 / FFmpeg-backend BGR pixels across repeated `VideoCapture.read()`
calls, even with one decode thread. Differences were concentrated in the
bottom boundary rows (within the bottom 4 rows on the sampled frames).
The bitstream remained unchanged. Five independent FFmpeg CLI invocations
produced byte-identical BGR24 frames for the first 32 decoded frames.
This is a **decoder-specific observation**, not proof that all FFmpeg inputs
and platforms are deterministic.

The shared `pixvqa_video.ffmpeg_bgr.FFmpegCapture` backend substitutes
`cv2.VideoCapture` *for pixel decoding* in both learned metric preprocessors.
It uses FFprobe for metadata, launches FFmpeg lazily on the first read,
streams packed BGR24 pixels through a pipe, sets decoder threads to one and
disables implicit timestamp-based frame duplication. SimpleVQA retains
its upstream and rolling-window temporal sampling and existing PIL/torchvision
transforms. CVQA-HDR-NR retains the official midpoint sampler for its default
1 Hz mode and the custom-interval sampler for other configurations.
OpenCV remains available for explicit comparisons; CVQA preflight still uses
OpenCV only for container frame count / FPS metadata.

The change is **not numerically parity-validated yet**: FFmpeg CLI and OpenCV
may produce different 8-bit BGR values and hence learned quality scores.

```bash
cd ~/codeccomp/metrics/pixvqa
git pull --ff-only origin main

# Required if the vqa environment uses a non-editable installed package:
python -m pip install -e python

PYTHONPATH=python python scripts/test_simplevqa_motion_eval.py
PYTHONPATH=python python scripts/test_ffmpeg_bgr_capture.py

# For the repeatability benchmark, select the FFmpeg binary already tested:
export PIXVQA_LEARNED_DECODER=ffmpeg
export PIXVQA_FFMPEG_BIN=/home/sergio/codeccomp/encoders/nvenc/ffmpeg
export PIXVQA_FFPROBE_BIN=ffprobe
export PIXVQA_LEARNED_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Repeat the exact SAME MP4 evaluation at least 5 times,
# without re-encoding; compare raw score and intermediate tensor hashes.

# For control measurements only:
export PIXVQA_LEARNED_DECODER=opencv
```

For persistent or shared-GPU SimpleVQA workers, restart the worker/daemon after
changing the decoding environment or updating the code. The standalone
subprocess starts fresh each time. Keep `PIXVQA_SIMPLEVQA_TRACE_DIR` set only
when diagnosing differences and disable it for speed measurements.

Do not prematurely declare CVQA-HDR-NR fixed. The previously observed
two-/three-state CVQA raw scores require new full-video repeated runs using
the corrected decoder and full per-sample JSON if variation remains. Record
decoder versions, checksum and provenance with any final RD results.
