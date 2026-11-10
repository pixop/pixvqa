# Production configuration — SimpleVQA and CVQA-HDR-NR

**Status (2026-11-10):** Production-default code on `main`; Aerial end-to-end validation of the *pre-cleanup* FFmpeg code passed. **Do not tag a production release until the post-cleanup GPU/worker acceptance gates below pass.**

## No diagnostic variables required

After regular optional learned-metric installation, the supported invocation is simply:

```bash
cd /path/to/pixvqa
./bin/pixvqa --metrics simplevqa,cvqa-hdr-nr -t /path/to/video.mp4
```

This evaluates both NR learned metrics with FFmpeg decoding, deterministic inference, the pinned upstream architectures and verified checkpoints. Neither a macro/compile-time toggle nor any of these historical exports is required:

- `PIXVQA_LEARNED_DECODER` — **removed**. There is no OpenCV pixel decoder fallback.
- `PIXVQA_SIMPLEVQA_LEGACY_BOUNDED` — **removed**. Use the rolling-window sampler.
- `PIXVQA_LEARNED_DETERMINISTIC=1` — **default is now enabled**.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` — selected automatically inside the learned adapters before CUDA initialization, if not already set.
- `PIXVQA_SIMPLEVQA_TRACE_DIR` — diagnostic tracing only, off by default; not a production requirement.
- `PIXVQA_CVQA_NR_PROFILE` — optional performance profiling, off by default.

The learned installer provisions pinned FFmpeg n8.1.3 `ffmpeg` and `ffprobe` into
the managed `.venv/bin`. Learned adapters resolve those binaries directly
(stamp file `PIXVQA_LEARNED_FFMPEG_COMMIT` plus version check). System `PATH` is
not searched. `PIXVQA_FFMPEG_BIN` and `PIXVQA_FFPROBE_BIN` remain explicit
overrides; they must still report FFmpeg n8.1.3. `PIXVQA_CVVDP_FFPROBE` and
`PIXVQA_CVQAHDR_FFPROBE` remain legacy metadata-only overrides.

## Install / update

```bash
cd /path/to/pixvqa
git pull --ff-only origin main
python -m pip install -e python --no-deps
bash scripts/install_learned_metrics.sh   # or answer yes in install_pixvqa.sh
.venv/bin/ffmpeg -version
.venv/bin/ffprobe -version
```

For a fresh installation follow `docs/LEARNED_METRICS.md` and `scripts/install_learned_metrics.sh` to install/check the pinned upstream repositories, pretrained checkpoints and Python environment. Editable installation is useful for an existing environment; use the project's managed installer when required packages or assets are absent. The PixVQA native C++ binary does not require rebuilding for these Python adapter changes.

## Processing contract

- Decoder: external FFmpeg, single decoder thread, first video stream, sequential decode in presentation order with passthrough frame synchronization, BGR24 output, no implicit frame duplication. Metadata comes from FFprobe. The decoder is lazily launched and cleaned up when sampling exits.
- `SimpleVQA`: spatial keyframes follow the upstream FPS-rounded schedule; SlowFast receives 32-frame windows with bounded buffering. SlowFast and quality regressor both use `eval()`.
- `CVQA-HDR-NR`: midpoint-of-interval keyframes, default 1 Hz (19 samples for the 1,199-frame Aerial test), upstream-equivalent `cv2.resize`, PIL Resize/CenterCrop, ToTensor and normalization, chunked model inference. OpenCV is an image-processing dependency, **not a decoder**.
- Both metrics: deterministic PyTorch operations are the default. A GPU operator lacking deterministic implementation must fail rather than silently execute nondeterministically. This may reduce throughput compared with non-deterministic kernels.
- Scores from historical OpenCV-decoded experiments are **not automatically numerically comparable** to new FFmpeg-decoded scores. Recompute RD evaluation results consistently where needed.

## Acceptance checklist before release tag

1. Run `python -m compileall -q python/pixvqa_video python/pixvqa_simplevqa python/pixvqa_cvqahdr`.
2. Run `PYTHONPATH=python python scripts/test_ffmpeg_bgr_capture.py`, `scripts/test_simplevqa_motion_eval.py`, and `scripts/test_cvqa_decoder_override.py`.
3. On original i9-9900K/RTX 4070, run the unchanged Aerial MP4 **three times with no diagnostic exports** (preserve SHA-256 `60437fd57bcee4fa7e87a1c2e39c084afee178d340bda5fe50cbfa5d744524de`). Expect SimpleVQA `3.524660348892212` and CVQA raw `0.7270195484161377` on every run; 19 CVQA samples. Compare exact JSON and protocol score values.
4. Run an 8–10 s SDR smoke and the Aerial sequence in both SimpleVQA persistent-worker and shared-GPU worker modes, verifying exact score parity with standalone. Test multiple concurrent jobs, service teardown/restart, decode errors, exit status and bounded RSS/VRAM.
5. Evaluate additional SDR/sports motion and HDR10 (CVQA only) for stability, checking range/matrix metadata and whether unsupported transfer characteristics are rejected correctly.
6. Record/checkpoint the production PixVQA commit, FFmpeg/FFprobe version, torch/torchvision/PyTorchVideo versions, CUDA driver/toolkit, CPU/GPU identifier, upstream model pin and checkpoint digest.

**Current validated baseline (pre-cleanup FFmpeg integration):** three sequential Aerial evaluations on the i9 yielded bit-identical 79-stage SimpleVQA traces and identical scores for both learned metrics; independent FFmpeg on `Workstation` reproduced native YUV10 and BGR24 for all 1,199 frames, and matched explicit BT.709 limited-range conversion.

For incident history, exact hashes, scores and root-cause evidence, see [bug-fix report](../dev/LEARNED_METRIC_DECODER_BUGFIX_REPORT_2026-11-10.md) and [issue #24](https://github.com/pixop/pixvqa/issues/24).
