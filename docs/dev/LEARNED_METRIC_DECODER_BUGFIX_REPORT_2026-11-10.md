# Bug-fix report — learned NR metric score nondeterminism

**Date:** 2026-11-10  
**Repository:** `pixop/pixvqa` (`main`)  
**Issue:** [#24](https://github.com/pixop/pixvqa/issues/24)  
**Affected metrics:** SimpleVQA, CompressedVQA-HDR-NR  
**Status:** Corrected and independently validated for the pinned Aerial test input; production refactor merged to `main`, pending repeat end-to-end GPU acceptance of refactored production defaults and worker/concurrency modes.

## Executive summary

Repeated evaluation of a bit-identical HEVC Main 10 Aerial video produced inconsistent learned no-reference quality scores, while the native FR metrics were stable. There were **two independently identified SimpleVQA issues**: (1) the upstream SlowFast motion backbone was not switched to `eval()` mode, and (2) OpenCV's FFmpeg-backed `VideoCapture` sometimes produced **different BGR pixel values at the bottom image boundary**, even when configured with one decoding thread. FFmpeg CLI decoded the same file deterministically to both native 10-bit YUV420 and BGR24 across the full 1,199-frame sequence. We replaced the learned metrics' pixel decoding backend with streaming FFmpeg BGR24, preserved the model preprocessing, and enabled deterministic PyTorch inference as the default.

### Reproduction input

- Filename: `Aerial_1920x1080_60_F0-1198_420_10bit_sdr_bt709l_yuv_tbr10000_ip60_bfauto_420_10b_auto_hq_hrd2000_aqs0_aqt0_laauto_sg0_brefauto_wpauto_hevc_nvenc_4070_sdk13.mp4`
- SHA-256: `60437fd57bcee4fa7e87a1c2e39c084afee178d340bda5fe50cbfa5d744524de`
- Size: 24,573,997 bytes
- Format: HEVC Main 10, `yuv420p10le`, 1920×1080, 60/1 fps, 1,199 frames; BT.709 primaries/transfer/matrix, limited (`tv`) range.
- Filesystem location during independent validation: `D:\projects\pixvqa-public` on the connected `Workstation` host. Original scoring: `i9-9900K`, RTX 4070, PyTorch 2.4.1+cu121.

## Evidence and root-cause analysis

1. **Initial mismatch:** same encoded file, stable XPSNR/VMAF, changing SimpleVQA and CVQA raw scores. CVQA raw values previously included `0.6774996519088745`, `0.6880412697792053`, and `0.6922507286071777`; SimpleVQA after the SlowFast eval fix still varied at the fourth decimal place.
2. **Upstream SimpleVQA model state:** official `test_demo.py` set the spatial regressor to eval but did not set `model_motion.eval()`. Added explicit eval in staged inference paths. This reduced, but did not eliminate, SimpleVQA variation.
3. **First differing SimpleVQA tensor:** SHA-256 stage trace showed `spatial_input` unchanged while `motion_slow_0` differed before SlowFast inference. Thus the remaining variation reached the network through preprocessed input pixels.
4. **Isolated OpenCV decoding:** repeatedly decoding the same MP4 with OpenCV 4.11.0 (`CAP_FFMPEG`) produced different BGR hashes. The same occurred with `CAP_PROP_N_THREADS=1`. A compared frame had differences restricted to the last two rows (rows 1078–1079), including 3,658 differing pixels, 8,227 differing channel values, and a maximum absolute channel difference of 254. A separate test found 32/32 decoded frames differed full-frame but 0/32 differed when the bottom four rows were excluded. The precise OpenCV/FFmpeg integration defect was **not** isolated to a specific function or buffer; uninitialized padding/edge handling remains a hypothesis, not a verified root cause.
5. **Independent FFmpeg CLI control:** two decodes of the complete file to `yuv420p10le` generated identical per-frame MD5s for **1,199/1,199 frames**. Two decodes to BGR24 did likewise. A third BGR24 decode using explicit `scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709,format=bgr24` matched the default BGR24 conversion for **1,199/1,199 frames**. This rules out an FFmpeg default range/matrix mismatch for this exact clip. FFmpeg on the validation host: `n8.1.3-20260924`.
6. **Full metric reproducibility after FFmpeg integration:** three sequential i9-9900K CUDA executions of the identical encoded MP4 (SHA matching above) produced:
   - **SimpleVQA:** `3.524660348892212` on all three runs.
   - **CVQA-HDR-NR raw:** `0.7270195484161377` on all three runs; calibrated `9.147192303602765` on all three.
   - **SimpleVQA trace:** all 79 stages (inputs, motion features, final score) byte-identical.
   - Evaluation directory: `~/codeccomp/results/261108/simplevqa_repeatability_20261110_011411_77017`.
   - These runs explicitly selected deterministic execution; production configuration now selects it by default.

## Implemented correction

- Shared `pixvqa_video.ffmpeg_bgr.FFmpegCapture`: FFprobe metadata + sequential, single-threaded FFmpeg BGR24 streaming over a pipe; frames retained only as needed for existing spatial/motion sampling; frame-rate passthrough.
- **SimpleVQA:** SlowFast `eval()`, FFmpeg-backed upstream spatial sampling, CUDA rolling 32-frame motion-window processing, deterministic inference on by default. Removed optional OpenCV decoder and legacy bounded single-decode feature switch.
- **CVQA-HDR-NR:** default and custom schedules now use the local FFmpeg-backed sampler, retaining upstream-equivalent resizing, center crop, normalization, midpoint selection, and model inference. Eliminated temporary process-wide monkeypatching of `cv2.VideoCapture`. Deterministic inference on by default.
- **No OpenCV video decode fallback** in either learned NR metric. OpenCV can remain a dependency for `resize` and `cvtColor`, which are not video decoding operations.
- `PIXVQA_LEARNED_DETERMINISTIC`, `CUBLAS_WORKSPACE_CONFIG`, and `PIXVQA_LEARNED_DECODER` do **not** need to be exported for normal production operation. The legacy decoder selector has been removed; `PIXVQA_LEARNED_DETERMINISTIC=0` is an explicit diagnostic/performance opt-out, not a production requirement.
- Existing trace support remains available for engineering incident diagnosis but is off by default.

## What is *not* claimed

- OpenCV vs FFmpeg scores are **not** numerically parity-preserving. In fact, the score shift is expected when OpenCV passes unstable input pixels. Under the new decoder, old baseline score values must not be silently compared with post-fix scores without noting this methodological change.
- End-to-end deterministic behavior is established for one tested Aerial bitstream and environment, not all HEVC streams, chroma formats, GPUs or PyTorch versions.
- The CVQA-HDR-NR **final score** is stable across three runs; unlike SimpleVQA, a per-layer CVQA trace was not run.
- The post-cleanup production-default refactor still needs acceptance on i9-9900K, especially standalone/worker/shared GPU execution and parallel jobs.

## Regression and release gates

1. Run `python -m compileall -q python/pixvqa_video python/pixvqa_simplevqa python/pixvqa_cvqahdr`.
2. Run `PYTHONPATH=python python scripts/test_simplevqa_motion_eval.py`, `scripts/test_cvqa_decoder_override.py`, and `scripts/test_ffmpeg_bgr_capture.py` (with ffmpeg/ffprobe installed).
3. With **all diagnostic environment variables unset**, evaluate the **same** 1,199-frame Aerial MP4 at least three times: require exact SimpleVQA raw score identity, exact CVQA-HDR-NR raw score identity, correct sampling count (19), and no decode errors.
4. Confirm that post-refactor scores still match the previously validated FFmpeg results `3.524660348892212` (SimpleVQA) and `0.7270195484161377` (CVQA raw) in the original i9 environment. Any departure is a release blocker pending explanation.
5. Validate persistent worker and shared-GPU modes, concurrency, and two additional streams: fast motion/sports and HDR10 for CVQA. Check peak RSS/VRAM, deadlock/pipe cleanup on error, latency and throughput.
6. Pin/record FFmpeg binary build, model checkpoint digest, Python/PyTorch/CUDA versions and PixVQA commit for future RD comparisons.

**Release classification:** decoder bug fix verified on Aerial; hardening changes implemented but require post-refactor metric and worker acceptance before tagging a production release.
