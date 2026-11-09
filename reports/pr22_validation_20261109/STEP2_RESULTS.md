# PR #22 STEP 1 + STEP 2 validation — 2026-11-09

Branch: `perf/simplevqa-bounded-motion-20261109` (post 8be0ab2 fixes)

## STEP 1 — production changes

| Area | Change |
|------|--------|
| AF_UNIX | `ipc.py`: auto short-dir fallback (`/tmp/pixvqa-adm-<uid>-<hash>`); `SocketPathError` on unfixable paths |
| CVQA profile | Default `1080p-batch`: 1650/1300 MiB RAM/VRAM; `PIXVQA_SCHEDULER_CVQA_PROFILE=conservative` → 2500/2800 |
| Tests | `test_long_scheduler_dir_socket_fallback`, `test_cvqa_1080p_batch_profile_defaults` |
| Docs | `METRIC_SCHEDULER.md` caveats + env knobs |

## STEP 2 — measured results (RTX 5080 WSL)

| Workload | Wall (s) | GPU peak (MiB) | Score parity | Notes |
|----------|----------|----------------|--------------|-------|
| CVQA admission ×5 (post-fix) | **46.86** | **6766** | 8.395391 | job spread **1.62 s** (no 5th-queue); socket auto-fallback 48 B |
| CVQA admission ×5 (pre-fix) | 50.71 | 4549 | yes | job_4 queued ~16 s |
| CVQA bare ×5 (prior) | 41.62 | 5614 | yes | baseline |
| SimpleVQA shared ×5 | **149.89** | **3950** | 3.870050 | admission on; single daemon |
| Mixed 1 CVQA + 1 shared SVQA | **36.44** combined | (see json) | both yes | `mixed_admission_benchmark.json` |
| Unit tests | — | — | — | scheduler 19, global admission 8, shared worker 3, SimpleVQA ref |
| Native selftest | — | — | — | `/tmp/pixvqa-pr22-build-isolated/selftest` PASS |

Safety: VRAM < 15500 MiB, MemAvailable > 2048 MiB, swap 0 throughout. GPU cleaned to **0 MiB**.

## Not run / limitations

- High-pressure 5 CVQA + 2–5 SimpleVQA (staged OOM guard) — not executed this pass; mixed 1+1 reconfirmed.
- **CI**: `gh` CLI unavailable; remote checks not verified from this host.
- WSL `nvidia-smi` compute-apps empty — GPU overlap not PID-proven.

## Merge readiness

**Ready for review** with documented caveats (1080p-batch tier, foreign CUDA processes, UHD → `conservative` profile). **Do not merge** without maintainer sign-off.
