# SimpleVQA staged-script import-path regression — 2026-11-10

Branch: `security/resolve-learned-env-20261110` (unmerged review branch).

## Root cause

The normal SimpleVQA adapter executes an overlay of upstream `test_demo.py` in a temporary working directory. The CUDA overlay imports `pixvqa_video.ffmpeg_bgr`, but the child interpreter's `sys.path` is initialized from the temporary directory, not from the parent adapter's import path. A separately exported `PYTHONPATH=<repo>/python` concealed the bug. The persistent worker also inherited the parent's environment without adding the PixVQA import root. The shared GPU daemon already provided the root.

## Patch

- `python/pixvqa_simplevqa/adapter.py`: introduce `staged_demo_env()`; prepend its own `python/` package root to child `PYTHONPATH` while preserving other caller entries and avoiding duplicate root; pass this environment to the staged subprocess.
- `python/pixvqa_simplevqa/worker.py`: use the same environment in the persistent worker process.
- `scripts/test_simplevqa_staged_import_path.py`: launch a real separate interpreter from a temporary directory with no exported `PYTHONPATH`, verify the staged FFmpeg import, plus environment-order assertions.

## RTX 5080 evidence

- Isolated regression: `PASS: staged SimpleVQA import without exported PYTHONPATH`.
- Overlay regression: CPU, CUDA, shared-GPU overlays all PASS.
- Real GPU regular adapter: `GPU_NO_PYTHONPATH_PASS 4.055310249328613 upstream-test_demo`.
- Real GPU persistent worker: `WORKER_NO_PYTHONPATH_PASS 4.055310249328613 persistent-worker`.
- Input: `/tmp/pixvqa_simplevqa_import_smoke_10s.mp4`, 12 seconds, stream-copy loop of checked-in `samples/dist_hevc.mp4`. This is a functional smoke input, **not** the Aerial golden fixture; numerical parity is established only between these two GPU modes here.
- Python: isolated CUDA 13 test env `/home/sergio/pixvqa-security-learned-env-20261110/.venv/bin/python`, with `PYTHONPATH` explicitly unset in the parent process.
- A shorter 2-second sample was rejected before inference by the existing minimum-spatial-sample validation; this was not the import defect.

## Remaining release gates

- RTX 4070 branch-fix smoke test with `PYTHONPATH` unset (i9 is not directly accessible from the RTX 5080 tunnel).
- CVVDP PQ nonsaturated comparison on RTX 4070; earlier same-machine PQ JOD 10.0 versus 10.0 is a compatibility check only.
- Full release/security-alert review; verify provenance metadata against the experimental stack.
- Do not automatically merge to `main`.

The full `scripts/install_learned_metrics.sh` already invokes `scripts/setup_hdr_chipqa_envs.sh`; the lower-level `scripts/setup_learned_env.sh` intentionally installs only the shared learned runtime. Running only the latter requires the separate HDR-ChipQA setup stage.
