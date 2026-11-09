# PixVQA metric resource scheduler

Branch: `perf/simplevqa-bounded-motion-20261109`  
Module: `python/pixvqa_scheduler/`

## Purpose

**Opt-in** Python resource admission for concurrent PyTorch metric jobs:

| Mode | Env / CLI | Scope |
|------|-----------|-------|
| Batch orchestrator | `PIXVQA_SCHEDULER=1` + `python -m pixvqa_scheduler …` | Single Python parent launches subprocesses |
| **Global admission** | `PIXVQA_SCHEDULER_ADMISSION=1` or `pixvqa --scheduler-admission` | Cross-terminal AF_UNIX daemon + leased tokens |

The scheduler/daemon:

- Does **not** load CUDA models (subprocess-only reservation).
- Applies per-metric concurrency caps and RAM/VRAM **budget reservations**.
- Polls host `MemAvailable` and **device-global** `nvidia-smi` memory (not PID-exclusive).
- Preserves existing `pixvqa` C++ CLI metric order and numerical outputs.

## Global admission daemon

Per-user socket under `$XDG_RUNTIME_DIR/pixvqa/` or `$PIXVQA_SCHEDULER_DIR` (WSL/Linux; requires `AF_UNIX` + `flock`). Linux `sun_path` is **107 bytes** (path + NUL). If `PIXVQA_SCHEDULER_DIR/.../admission.sock` would exceed that limit, the runtime dir **automatically falls back** to `/tmp/pixvqa-adm-<uid>-<hash12>/` (deterministic per long path — no namespace collision between distinct dirs). Set `PIXVQA_SCHEDULER_DAEMON_SOCKET` for a fixed short path, or choose a short `PIXVQA_SCHEDULER_DIR` yourself. Overlong explicit socket paths raise `SocketPathError`.

```bash
export PIXVQA_SCHEDULER_ADMISSION=1
export PYTHONPATH=python
pixvqa --scheduler-admission --metrics simplevqa -t clip.mp4 ...
# or direct Python backends:
python -m pixvqa_simplevqa --repo … --checkpoint … --test clip.mp4
```

The daemon auto-starts on first lease request. Manual foreground server:

```bash
python -m pixvqa_scheduler daemon
python -m pixvqa_scheduler daemon --status   # print socket path
```

Leases are tied to the client PID; stale PIDs and heartbeat timeouts release slots without manual cleanup.

### Batch + global

When both `PIXVQA_SCHEDULER=1` and `PIXVQA_SCHEDULER_ADMISSION=1` are set, the batch parent acquires the daemon lease and sets `PIXVQA_SCHEDULER_ADMISSION_SKIP=1` on children (no double-count).

## Verified process topology (audit)

### `pixvqa` C++ driver (`src/main.cpp`)

- Learned metrics launched via `fork` + `execvp`.
- **One pixvqa process runs at most one Python backend at a time** (sequential).
- Cross-metric parallelism happens when the **caller** launches multiple processes/terminals.
- `--scheduler-admission` sets `PIXVQA_SCHEDULER_ADMISSION=1` for Python backends.

### CompressedVQA-HDR-NR (`python/pixvqa_cvqahdr`)

| Item | Verified |
|------|----------|
| Invocation | `python -m pixvqa_cvqahdr` or `pixvqa --metrics cvqa-hdr-nr` |
| Model load | In-process per Python adapter |
| Shared worker | **None** |
| Admission | Opt-in via `PIXVQA_SCHEDULER_ADMISSION=1` |

### SimpleVQA (`python/pixvqa_simplevqa`)

| Mode | Notes |
|------|-------|
| Default streaming CUDA | Independent subprocess per job |
| `--use-shared-gpu-worker` | One daemon, one model; admission uses `shared_gpu` VRAM slot (4200 MiB once) |
| Global admission | Same env as CVQA; shared-GPU sets `shared_gpu=True` profile |

## Scheduler execution strategies

| Strategy | Metrics | Behavior |
|----------|---------|----------|
| `independent` | `cvqa-hdr-nr`, default `simplevqa` | Separate subprocess; each may load its own model |
| `shared_gpu` | `simplevqa` + `--use-shared-gpu-worker` | Single VRAM reservation; high client concurrency cap |

Default per-metric caps (override via env): see prior table in env knobs.

## Enabling (batch)

```bash
export PIXVQA_SCHEDULER=1
export PYTHONPATH=python
python -m pixvqa_scheduler simplevqa-batch \
  --repo deps/SimpleVQA --checkpoint deps/SimpleVQA/ckpts/UGC_BVQA_model.pth \
  --test clip.mp4 --count 3 --shared-gpu
```

### Environment knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `PIXVQA_SCHEDULER` | off | `1` enables batch orchestrator admission |
| `PIXVQA_SCHEDULER_ADMISSION` | off | `1` enables global daemon leases in backends / `--scheduler-admission` |
| `PIXVQA_SCHEDULER_ADMISSION_SKIP` | off | Internal: parent holds lease |
| `PIXVQA_SCHEDULER_DIR` | `$XDG_RUNTIME_DIR/pixvqa` | Daemon socket + lock directory (auto short fallback if path >107 B) |
| `PIXVQA_SCHEDULER_CVQA_PROFILE` | `1080p-batch` | `conservative` / `uhd` restores 2500/2800 MiB reserves for unknown/UHD |
| `PIXVQA_SCHEDULER_CVQA_RAM_MIB` | 1650 (`1080p-batch`) | Per-job host RAM reservation (scheduler accounting) |
| `PIXVQA_SCHEDULER_CVQA_VRAM_MIB` | 1300 (`1080p-batch`) | Per-job VRAM reservation (not PID-exclusive) |
| `PIXVQA_SCHEDULER_MAX_CONCURRENT_CVQA_HDR_NR` | 5 | Per-metric cap |
| `PIXVQA_SCHEDULER_MAX_CONCURRENT_SIMPLEVQA` | 2 | Independent SimpleVQA cap |
| `PIXVQA_SCHEDULER_MEM_FLOOR_MIB` | 2048 | Minimum MemAvailable |
| `PIXVQA_SCHEDULER_GPU_HEADROOM_MIB` | 512 | Device-global VRAM headroom |
| `PIXVQA_SCHEDULER_MAX_TOTAL` | 16 | Global in-flight cap |
| `PIXVQA_SCHEDULER_QUEUE_TIMEOUT_S` | 3600 | Bounded wait |
| `PIXVQA_SCHEDULER_LEASE_HEARTBEAT_S` | 60 | Stale lease cleanup |

## Reservation accounting

Admission compares **live telemetry** with **scheduler commitments** without double-counting resident memory (same rules as in-process batch scheduler). See commit `42c7419` audit.

### CVQA profile tiers (data-driven caveats)

| Tier | RAM / VRAM per job | Source |
|------|-------------------|--------|
| `1080p-batch` (default) | 1650 / 1300 MiB | PR #22 RTX 5080: 5× BasketballDrive 1080p60 60 s bare peak **5614 MiB global VRAM**, ~7.6 GiB host drop. Margins applied; **not** `5614÷5` as an exact per-process value. |
| `conservative` / `uhd` | 2500 / 2800 MiB | Prior defaults; use for 4K/UHD or unprofiled content. |

Five concurrent 1080p jobs may **queue unnecessarily** under `conservative` reserves even when telemetry shows headroom — switch tier or override env MiB knobs. Scheduler reservations are **not** hard RSS/VRAM caps; live `MemAvailable` / `nvidia-smi` guards still apply.

## Caveats

1. **External CUDA clients** are not tracked — `nvidia-smi` is device-global; admission cannot evict foreign GPU users.
2. **Native Windows** (no `AF_UNIX`/`fcntl`): global admission falls back with a stderr warning; batch in-process mode still works inside one Python parent on any OS.
3. **C++ pixvqa** metric order unchanged; only env injection for admission.
4. **CVQA concurrency**: scheduler does not serialize CVQA globally beyond budget caps (default 5).
5. **Score outputs**: admission wraps execution only — no numerical changes to metric results.

## Tests

```bash
PYTHONPATH=python python scripts/test_metric_scheduler.py
PYTHONPATH=python python scripts/test_global_admission.py
PYTHONPATH=python python scripts/test_simplevqa_shared_gpu_worker.py
```

Mixed workload benchmark (optional, needs GPU + assets):

```bash
/home/sergio/pixvqa-simplevqa-env/bin/python \
  reports/pr22_validation_20261109/run_mixed_admission_benchmark.py
```

## Measured evidence (PR #22, RTX 5080 WSL)

| Workload | Outcome | Global VRAM peak |
|----------|---------|------------------|
| 5× SimpleVQA streaming independent | Safety abort ~15.9 GiB | 15906 MiB |
| 5× SimpleVQA shared-GPU worker | Scores `3.8700499534606934` | 3950 MiB |
| Global admission daemon | Unit-tested cross-process leases | (mixed benchmark when CVQA installed) |

Artifacts: `reports/pr22_validation_20261109/`
