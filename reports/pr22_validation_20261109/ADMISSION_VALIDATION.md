# Global admission — validation 2026-11-09

Branch: `perf/simplevqa-bounded-motion-20261109`  
Python: `/home/sergio/pixvqa-simplevqa-env/bin/python`  
Clip: `clips/BasketballDrive_1080p50_60s_loop6x_streamcopy.mp4`  
Reference SimpleVQA score: `3.8700499534606934`

## Unit tests

```
PASS: scripts/test_metric_scheduler.py (19 checks incl. cvqa-profile-tiers)
PASS: scripts/test_global_admission.py (8 checks; AF_UNIX long-path fallback, crash release)
PASS: scripts/test_simplevqa_shared_gpu_worker.py (mock)
PASS: scripts/test_simplevqa_reference.py
PASS: build/cpu/selftest (isolated copy /tmp/pixvqa-pr22-build-isolated/selftest)
```

## STEP 1 fixes (2026-11-09 post 8be0ab2)

1. **AF_UNIX path**: `ipc.py` auto-falls back to `/tmp/pixvqa-adm-<uid>-<hash12>/` when `PIXVQA_SCHEDULER_DIR/admission.sock` exceeds 107 bytes; `SocketPathError` for unfixable explicit paths.
2. **CVQA reserves**: default `PIXVQA_SCHEDULER_CVQA_PROFILE=1080p-batch` → 1650/1300 MiB RAM/VRAM per job (measured RTX 5080 + margin). `conservative` tier keeps 2500/2800 for UHD/unknown.

## CVQA admission ×5 post-fix (RTX 5080, 60 s BasketballDrive)

| Metric | Pre-fix admission_5 | Post-fix admission_5 |
|--------|---------------------|----------------------|
| Wall (s) | 50.71 | **46.86** |
| GPU peak (MiB) | 4549 (5th queued) | **6766** (all 5 concurrent) |
| job_4 wall spread | ~16 s vs peers | **1.62 s** |
| Score parity | yes | yes (8.395390956596998) |
| Socket | manual `/tmp/…` workaround | auto fallback (48 B path) |

## Real GPU — 2× SimpleVQA shared worker + global admission

Env: `PIXVQA_SCHEDULER_ADMISSION=1`, `--use-shared-gpu-worker`, CUDA.

| Job | Score | Parity (|Δ|≤1e-4) | Wall (combined) | GPU peak (MiB) |
|-----|-------|-------------------|-----------------|----------------|
| a | 3.8700499534606934 | yes | ~65 s | 3950 |
| b | 3.8700499534606934 | yes | (parallel) | (global) |

Matches prior 5-client shared-worker baseline (152.6 s for 5 jobs; 3950 MiB VRAM).

## Mixed SimpleVQA + CVQA-HDR-NR

**PASS** (2026-11-09): `deps/CompressedVQA-HDR` @ `76bff6d5`, weights symlinked from
`/home/sergio/models/compressedvqa-hdr/`, SHA256 matches `python/pixvqa_cvqahdr/__init__.py`.

Env: `PIXVQA_SCHEDULER_ADMISSION=1`; 1× SimpleVQA shared worker + 1× CVQA-HDR-NR on 60 s clip.

| Job | Score | Parity (|Δ|≤1e-4) | Wall (s) |
|-----|-------|-------------------|----------|
| CVQA baseline (sequential) | 8.395390956596998 | — | 17.78 |
| simplevqa-shared | 3.8700499534606934 | yes | 33.75 |
| cvqa-hdr-nr (mixed) | 8.395390956596998 | yes vs baseline | 21.98 |

Telemetry: GPU peak 5303 MiB (global), min MemAvailable 10.9 GiB, swap 0, combined wall 33.75 s.
No OOM. Artifact: `mixed_admission_benchmark.json`.

## pixvqa C++ CLI

`--scheduler-admission` added to `src/main.cpp` + `include/pixvqa/types.hpp`.  
Worktree `make pixvqa` fails at link (pre-existing undefined `resolve_cvvdp_parameters` / `transfer_family_token`) — CLI smoke not run here.

## Limitations

1. Global admission cannot account for foreign CUDA processes on the GPU.
2. Native Windows without `AF_UNIX`/`fcntl`: falls back with warning (no cross-process leases).
3. Mixed CVQA+SimpleVQA measured on looped BasketballDrive 60 s clip only.
