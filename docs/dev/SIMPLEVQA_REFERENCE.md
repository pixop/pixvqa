# SimpleVQA reference runner

**Status:** standalone upstream reference runner (`python -m pixvqa_simplevqa`) and
**public CLI dispatch** via `pixvqa --metrics simplevqa` (SDR-only NR; pooled score
semantics; QCS integrates the pooled score only). Numerical parity is verified on the
official upstream clip below; no speedup claims.

## Upstream and dependencies

- Official source: https://github.com/sunwei925/SimpleVQA
- Source commit inspected: `017edda50e8cdbeb48113a001dbb9af0dbd2030b`.
  Checkout this revision for reproducible runs.
- Apache-2.0 source repository; separately verify the checkpoint's redistribution
  terms. Do not commit any model weights or third-party videos to PixVQA.
- Official model checkpoint: downloaded separately from the link in upstream README.
  The file must be readable and correspond to `UGC_BVQA_model.pth`.
- SlowFast pretrained model: upstream `test_demo.py` calls
  `slowfast_r50(pretrained=True)`. **This wrapper does not open sockets or run
  downloader code** — but the upstream process may still pull PyTorchVideo hub
  weights into the local cache when that line executes. There is no guard here
  against a randomly initialized backbone; if the hub/cache path fails, treat it
  as an upstream/runtime failure, not a validated reference.
- Python: PyTorch, torchvision, PyTorchVideo, opencv-python, Pillow, NumPy,
  pandas, and their transitive dependencies. Prefer the **same** repo-local
  `.venv` pinned in [`python/requirements/learned-env.txt`](../../python/requirements/learned-env.txt)
  as for cvvdp/cvqa-hdr-nr ([Learned Metrics](../LEARNED_METRICS.md)); do not claim
  validated inference until you run your own checks with real checkpoints.

## Running the unmodified upstream reference

All paths are expanded and resolved before validation and before passing `--dist`
to upstream, so the demo sees a single deterministic absolute input path.

With the official upstream repo and its trained checkpoint available:

```bash
PYTHONPATH=python /path/to/learned-env/bin/python -m pixvqa_simplevqa \
  --repo /path/to/SimpleVQA \
  --checkpoint /path/to/UGC_BVQA_model.pth \
  --test /path/to/sdr_compressed_video.mp4 \
  --device cpu --check

PYTHONPATH=python /path/to/learned-env/bin/python -m pixvqa_simplevqa \
  --repo /path/to/SimpleVQA \
  --checkpoint /path/to/UGC_BVQA_model.pth \
  --test /path/to/sdr_compressed_video.mp4 \
  --device cuda
```

The runner stages the upstream `test_demo.py` in an isolated temporary working
directory. **CUDA** (`--device cuda` / `--is_gpu` upstream) uses PixVQA's
rolling-window streaming overlay: separate spatial decode plus yielded 32-frame
SlowFast motion windows. **CPU** stages an overlay that drops unconditional
`DataParallel` and loads the checkpoint with `map_location` (upstream lines
191–194 otherwise fail on CPU). `model/` is always symlinked; checkpoint via
`ckpts/UGC_BVQA_model.pth`. The upstream checkout is never modified. JSON output
includes `upstream_test_demo` (`cpu-overlay` vs `cuda-streaming-motion-overlay`),
the pooled quality score, model SHA-256, upstream revision, **runtime stack**
(pip `importlib.metadata` versions for torch/torchvision/pytorchvideo/NumPy/OpenCV/Pillow when installed, plus Python implementation/version), and input metadata (including
`duration_seconds` and `color_transfer` when ffprobe reports them). Failure
exits: 2 for setup, 1 for runtime/inference. At present it runs the upstream
processing without a frame cap, optimizer, temporal score trace or QCS.

### Persistent CUDA worker (opt-in)

For batch scoring on CUDA, `--use-worker` keeps one child process alive and
reuses the loaded PyTorch/SlowFast stack across clips. Both the worker and default
CUDA subprocess stage the streaming overlay; the worker still uses
`cuda-single-decode-worker` provenance metadata.

```bash
PYTHONPATH=python .venv/bin/python -m pixvqa_simplevqa \
  --repo deps/SimpleVQA --checkpoint deps/SimpleVQA/ckpts/UGC_BVQA_model.pth \
  --test /path/to/clip.mp4 --device cuda --use-worker
```

`--use-worker` requires `--device cuda`. `pixvqa --metrics simplevqa` does
**not** forward this flag today; use the module above or wire
`--simplevqa-use-worker` in a future CLI revision for native dispatch.

Lifecycle: JSON-line protocol on stdin/stdout; dedicated stderr drainer threads
avoid pipe deadlocks; per-request UUID correlation; startup and per-score
timeouts bound waits; crash or protocol failure kills the child (next request
restarts). See `scripts/test_simplevqa_worker.py` for fast mock coverage.

### Preflight (ffprobe)

- `ffprobe` must succeed for stream + format queries. OSError, non-zero exit,
  or invalid JSON is a **setup** error (exit 2).
- **SDR only:** `color_transfer` values indicating PQ or HLG (e.g. `smpte2084`,
  `arib-std-b67`, and common aliases) are rejected. Unknown/missing transfer is
  allowed for preflight only; do not assume HDR safety from metadata gaps.
- **Length:** upstream effectively needs at least ~one second at the reported
  frame rate. If `nb_frames` is present, it must parse as a non-negative
  integer; malformed values fail preflight. If both duration and `nb_frames` are
  missing from metadata, preflight fails (indeterminate length). Sub-second
  duration (when known) or frame counts below one rounded-FPS second are
  rejected.
- **Raw filenames:** `.yuv` / `.raw` inputs are rejected regardless of metadata
  because transport is ambiguous for OpenCV; use tagged compressed derivatives.

If the upstream Python process exits non-zero, stderr (or a short stdout tail
when stderr is empty) is included in the error text for diagnosis.

## Known upstream limitations to address before integration

- The CPU reference path still allocates buffers proportional to the *entire decoded video*. The CUDA streaming overlay keeps at most 32 transformed motion frames in a deque while yielding each SlowFast window; spatial samples still scale with duration.
- It uses approximately one spatial sample and one SlowFast temporal clip
  per second, with whole-video pooling; a video shorter than one second is
  rejected by the wrapper. To validate the CUDA optimization, compare against
  the pinned upstream `test_demo.py` on identical videos and checkpoints.
  Target identical scores; accept only negligible, measured differences.
- The demo reads via OpenCV; this initial runner only supports compressed SDR
  video, not raw YUV. Raw sources in the NR corpus must be converted to
  format-tagged, deterministic compressed derivatives first.
- The model is designed for SDR UGC; PQ and HLG are unsupported.
- The reference **upstream** process may download pretrained SlowFast weights unless
  pre-populated in the expected PyTorchVideo cache; this repo’s wrapper does not
  add downloads of its own.
- The wrapper test uses a **mock upstream script** to verify invocation and
  result parsing, not numerical validation. Actual parity needs authentic
  checkpoints, inputs, and verified reference scores.

## Validation gates for promotion to public CLI

1. Resolve upstream source revision, model checkpoint hash, SlowFast weight
   artifact hash, and **pinned pip versions** (see `runtime_stack` in JSON and
   [`python/requirements/learned-env.txt`](../../python/requirements/learned-env.txt))
   in the experiment manifest.
2. Generate bounded SDR reference clips from the local real-video corpus.
3. Compare direct official demo output to wrapper output for CPU/CUDA with
   identical inputs and pinned dependencies.
4. Profile memory on representative clips; retain score parity.
5. Add native CLI dispatch/group and compatibility behavior, learned-metric
   configuration, failure semantics, and mixed FR+NR regression tests.
6. Only after those gates, benchmark ONNX Runtime/TensorRT against PyTorch.

The unrelated Superposition `ENOMEM` native-read failure remains open.
Do not run experiments on the active separate benchmarking computer.

## Memory optimization (PR #22, merged path)

PixVQA's **default CUDA path** uses a **rolling-window FP32 overlay** that:

- Decodes spatial keyframes via the same `video_processing_spatial` routine as upstream.
- Yields each 32-frame SlowFast motion window from `video_processing_motion_stream`
  without retaining all windows simultaneously (deque `maxlen=32`).

448×448 spatial samples, model weights, FP32 operations, and the final whole-video
regressor are unchanged. This is **not** the Pixop Prime salient-region/partial-video variant.

**Validated FP32 parity** (exact scores, tolerance 1e-4) on:

| Clip | Duration | Score (all modes) |
|------|----------|-------------------|
| Official 960×540 | 8 s | 2.9507508277893066 |
| BasketballDrive 1080p50 | 10 s | 3.8700046882629395 |
| BasketballDrive looped | 60 s | 3.8700499534606934 |

Peak host RSS (60 s 1080p): upstream ~3267 MiB; streaming default ~1730 MiB (−47%);
legacy bounded single-decode ~2602 MiB (−20%).

### Legacy bounded overlay (internal only, TODO remove)

An earlier PR #22 single-decode path (`video_processing_unified`) remains behind
`PIXVQA_SIMPLEVQA_LEGACY_BOUNDED=1` for regression comparison. It is **not** a
supported user-facing mode and will be removed once streaming is sole default.

### Benchmark harness

From the PixVQA checkout with CUDA and learned dependencies:

```bash
.venv/bin/python scripts/benchmark_simplevqa_memory_parity.py \
  --videos /path/to/clip1.mp4 /path/to/clip2.mp4 \
  --output /tmp/simplevqa_parity.json
```

The harness compares pinned **unmodified upstream**, default **streaming**, and
optional **legacy bounded**. Requires `/usr/bin/time`, the pretrained SlowFast
cache, and reports full-precision scores, absolute differences, process peak RSS,
and elapsed time. It **does not** measure process-exclusive GPU VRAM.

### Reduced-precision research (not enabled)

FP16/BF16 evaluation is a separate opt-in experiment only. See
[`reports/pr22_validation_20261109/FP16_BF16_RESEARCH.md`](../../reports/pr22_validation_20261109/FP16_BF16_RESEARCH.md)
for benchmark design, risks, and acceptance gates. Default inference stays FP32;
do not enable reduced precision in production CLI until measured deviation,
ranking stability, and memory gains are approved.
