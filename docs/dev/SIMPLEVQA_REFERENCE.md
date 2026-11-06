# SimpleVQA reference runner: integration in progress

**Status:** standalone upstream reference runner; **not yet exposed by**
`pixvqa --metrics simplevqa`. No numerical parity or speedup has been claimed.

## Upstream and dependencies

- Official source: https://github.com/sunwei925/SimpleVQA
- Source commit inspected: `017edda50e8cdbeb48113a001dbb9af0dbd2030b`.
  Checkout this revision for reproducible runs.
- Apache-2.0 source repository; separately verify the checkpoint's redistribution
  terms. Do not commit any model weights or third-party videos to PixVQA.
- Official model checkpoint: downloaded separately from the link in upstream README.
  The file must be readable and correspond to `UGC_BVQA_model.pth`.
- SlowFast pretrained model: upstream `test_demo.py` calls
  `slowfast_r50(pretrained=True)`; it may require online download or an
  existing PyTorch hub cache. A successful import preflight does not guarantee
  the pretrained SlowFast weights are cached.
- Python: PyTorch, torchvision, PyTorchVideo, opencv-python, Pillow, NumPy
  and their transitive dependencies. Check an isolated compatible environment.

## Running the unmodified upstream reference

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
directory using symlinks to the original sources/checkpoint. It does not edit
the upstream checkout. Output is one JSON record containing the pooled quality
score, model SHA-256, revision and input metadata. Failure exits: 2 for setup,
1 for runtime/inference. At present it runs the upstream processing without
a frame cap, optimizer, temporal score trace or QCS.

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
- The reference process may download pretrained SlowFast weights unless
  pre-populated in the expected PyTorchVideo cache.
- The wrapper test uses a **mock upstream script** to verify invocation and
  result parsing, not numerical validation. Actual parity needs authentic
  checkpoints, inputs, and verified reference scores.

## Validation gates for promotion to public CLI

1. Resolve upstream source revision, model checkpoint hash, and SlowFast weight
   artifact hash in the experiment manifest.
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
