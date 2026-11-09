# Global admission — validation 2026-11-09

Branch: `perf/simplevqa-bounded-motion-20261109`  
Python: `/home/sergio/pixvqa-simplevqa-env/bin/python`  
Clip: `clips/BasketballDrive_1080p50_60s_loop6x_streamcopy.mp4`  
Reference SimpleVQA score: `3.8700499534606934`

## Unit tests

```
PASS: scripts/test_metric_scheduler.py (18 checks; pixvqa binary build skipped — link error in worktree)
PASS: scripts/test_global_admission.py (6 checks; AF_UNIX daemon, parallel clients, crash release)
PASS: scripts/test_simplevqa_shared_gpu_worker.py (mock)
```

## Real GPU — 2× SimpleVQA shared worker + global admission

Env: `PIXVQA_SCHEDULER_ADMISSION=1`, `--use-shared-gpu-worker`, CUDA.

| Job | Score | Parity (|Δ|≤1e-4) | Wall (combined) | GPU peak (MiB) |
|-----|-------|-------------------|-----------------|----------------|
| a | 3.8700499534606934 | yes | ~65 s | 3950 |
| b | 3.8700499534606934 | yes | (parallel) | (global) |

Matches prior 5-client shared-worker baseline (152.6 s for 5 jobs; 3950 MiB VRAM).

## Mixed SimpleVQA + CVQA-HDR-NR

**BLOCKED:** `CompressedVQA-HDR-NR` checkpoint/profile not present under authorized paths:

- Searched `/mnt/d/projects/pixvqa-public/deps/CompressedVQA-HDR/ckpts/NR_checkpoint.pth` — missing
- No `NR_checkpoint.pth` under `/mnt/d/projects/pixvqa-public`

Harness: `run_mixed_admission_benchmark.py` → `mixed_admission_benchmark.json` documents blocker.

## pixvqa C++ CLI

`--scheduler-admission` added to `src/main.cpp` + `include/pixvqa/types.hpp`.  
Worktree `make pixvqa` fails at link (pre-existing undefined `resolve_cvvdp_parameters` / `transfer_family_token`) — CLI smoke not run here.

## Limitations

1. Global admission cannot account for foreign CUDA processes on the GPU.
2. Native Windows without `AF_UNIX`/`fcntl`: falls back with warning (no cross-process leases).
3. CVQA mixed workload unmeasured until NR assets are installed locally.
