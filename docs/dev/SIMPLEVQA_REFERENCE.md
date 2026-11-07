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

The runner executes upstream `test_demo.py` in an isolated temporary working
directory using symlinks to the original sources and a symlink for
`ckpts/UGC_BVQA_model.pth` pointing at your checkpoint. It does not edit the
upstream checkout or copy full trees. Output is one JSON record containing the
pooled quality score, model SHA-256, upstream revision, **runtime stack**
(pip `importlib.metadata` versions for torch/torchvision/pytorchvideo/NumPy/OpenCV/Pillow when installed, plus Python implementation/version), and input metadata (including
`duration_seconds` and `color_transfer` when ffprobe reports them). Failure
exits: 2 for setup, 1 for runtime/inference. At present it runs the upstream
processing without a frame cap, optimizer, temporal score trace or QCS.

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

- The demo allocates buffers proportional to the *entire decoded video*,
  including one 224x224 RGB tensor per frame. Do **not** use unbounded
  UHD/long-video inputs for initial runs.
- It uses approximately one spatial sample and one SlowFast temporal clip
  per second, with whole-video pooling; a video shorter than one second is
  rejected by the wrapper.
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
4. Profile memory; implement streaming and optional max-frame windows while
   retaining score parity.
5. Add native CLI dispatch/group and compatibility behavior, learned-metric
   configuration, failure semantics, and mixed FR+NR regression tests.
6. Only after those gates, benchmark ONNX Runtime/TensorRT against PyTorch.

The unrelated Superposition `ENOMEM` native-read failure remains open.
Do not run experiments on the active separate benchmarking computer.
