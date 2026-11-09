# PixVQA metric resource scheduler

Branch: `perf/simplevqa-bounded-motion-20261109`  
Module: `python/pixvqa_scheduler/`

## Purpose

Opt-in admission control for **concurrent Python/PyTorch metric subprocesses** within a
single Python orchestrator. The scheduler:

- Does **not** load CUDA models (subprocess-only).
- Applies per-metric concurrency caps and RAM/VRAM **budget reservations**.
- Polls host `MemAvailable` and **device-global** `nvidia-smi` memory (not PID-exclusive).
- Preserves existing `pixvqa` C++ CLI behavior (metrics still run sequentially inside one `pixvqa` invocation).

## Verified process topology (audit)

### `pixvqa` C++ driver (`src/main.cpp`)

- Learned metrics (`cvvdp`, `cvqa-hdr-nr`, `simplevqa`, `hdr-chipqa`) are launched via `fork` + `execvp` through `run_backend_process`.
- **One pixvqa process runs at most one Python backend at a time** (sequential run order).
- Cross-metric parallelism happens only when the **caller** launches multiple `pixvqa` or `python -m pixvqa_*` processes.

### CompressedVQA-HDR-NR (`python/pixvqa_cvqahdr`)

| Item | Verified |
|------|----------|
| Invocation | `python -m pixvqa_cvqahdr` or `pixvqa --metrics cvqa-hdr-nr` (spawns Python subprocess) |
| Model load site | In-process inside the Python adapter (`NR_model.SigLIP2_384_multi_dataset().to(device)`) |
| Shared worker | **None** — each process loads its own model |
| 5 concurrent calls | Reported to succeed in user testing on this branch |
| 5 independent CUDA model instances | **Not verified** (no per-process CUDA allocator instrumentation) |
| Internal changes | **None** on this PR |

GPU telemetry inside the adapter (`GpuSampler`) uses device-wide `nvidia-smi` rows, explicitly not PID-exclusive.

### SimpleVQA (`python/pixvqa_simplevqa`)

| Mode | Process model | Verified concurrency |
|------|---------------|---------------------|
| Default streaming CUDA | Independent subprocess per job; staged `test_demo.py` overlay | 2× 60s jobs OK; 5× independent hit ~15.9 GiB global VRAM abort on RTX 5080 16 GiB |
| `--use-shared-gpu-worker` | One daemon, one loaded model, serialized GPU lock | 5 clients, score `3.8700499534606934`, 152.6 s, global VRAM peak **3950 MiB** |
| `--use-worker` | Persistent child process per parent | Mock-tested; import amortization only |

## Scheduler execution strategies

| Strategy | Metrics | Behavior |
|----------|---------|----------|
| `independent` | `cvqa-hdr-nr`, default `simplevqa` | Separate subprocess; each may load its own model |
| `shared_gpu` | `simplevqa` + `--use-shared-gpu-worker` | Single VRAM reservation; high client concurrency cap |
| `cpu_only` | (reserved) | No VRAM budget |
| `serialized` | (reserved) | One in-flight job per metric |

Default per-metric caps (override via env):

| Metric | Default max concurrent | VRAM reserve (MiB) | Notes |
|--------|------------------------|--------------------|-------|
| `cvqa-hdr-nr` | 5 | 2800 | Does not serialize CVQA — matches observed concurrent behavior |
| `simplevqa` independent | 2 | 4000 | Conservative after 5-job OOM abort |
| `simplevqa` shared_gpu | 8 | 4200 (once) | Shared daemon slot |

## Enabling

```bash
export PIXVQA_SCHEDULER=1
export PYTHONPATH=python
python -m pixvqa_scheduler simplevqa-batch \
  --repo deps/SimpleVQA --checkpoint deps/SimpleVQA/ckpts/UGC_BVQA_model.pth \
  --test clip.mp4 --count 3 --shared-gpu
```

When `PIXVQA_SCHEDULER` is unset, `MetricScheduler.run_batch()` falls back to **sequential pass-through** (no admission change).

### Environment knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `PIXVQA_SCHEDULER` | off | `1` / `true` enables admission |
| `PIXVQA_SCHEDULER_MAX_CONCURRENT_CVQA_HDR_NR` | 5 | Per-metric cap |
| `PIXVQA_SCHEDULER_MAX_CONCURRENT_SIMPLEVQA` | 2 | Independent SimpleVQA cap |
| `PIXVQA_SCHEDULER_MEM_FLOOR_MIB` | 2048 | Minimum MemAvailable |
| `PIXVQA_SCHEDULER_GPU_HEADROOM_MIB` | 512 | Device-global VRAM headroom |
| `PIXVQA_SCHEDULER_MAX_TOTAL` | 16 | Global in-flight cap |
| `PIXVQA_SCHEDULER_STAGGER_S` | 0 | Optional start stagger |

## Caveats

1. **Cross-invocation**: Budgets apply only inside one `MetricScheduler` Python process. Separate shells, CI shards, or concurrent `pixvqa` binaries do not share state (warning emitted in reports).
2. **GPU telemetry is global**: `nvidia-smi memory.used` includes all GPU clients; do not infer per-job VRAM from these samples alone.
3. **C++ pixvqa unchanged**: Single-invocation metric order and worker flags are untouched.
4. **CVQA concurrency**: Scheduler does not add a global lock on CompressedVQA; caps are configurable and default to permissive (5).

## Tests

```bash
PYTHONPATH=python python scripts/test_metric_scheduler.py
PYTHONPATH=python python scripts/test_simplevqa_shared_gpu_worker.py
PYTHONPATH=python python scripts/test_simplevqa_worker.py
```

## Measured evidence (PR #22, RTX 5080 WSL)

| Workload | Outcome | Global VRAM peak |
|----------|---------|------------------|
| 5× SimpleVQA streaming independent | Safety abort ~15.9 GiB | 15906 MiB |
| 5× SimpleVQA shared-GPU worker | All scores match `3.8700499534606934` | 3950 MiB |
| 2× SimpleVQA streaming | Both OK, identical scores | 3950 MiB |

Artifacts: `reports/pr22_validation_20261109/` (untracked validation harness).
