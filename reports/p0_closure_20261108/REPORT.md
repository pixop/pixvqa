# P0 Closure Report — NR Metric Infrastructure

**Date:** 2026-11-08 (initial review); **final pre-commit verification:** 2026-11-08 12:04 UTC+1  
**Branch:** `feature/nr-metric-infrastructure` @ `abb12a7` (local working tree, unstaged)  
**Reviewer action:** code review, doc repair, fast regression tests — **no commit, no push**

## Verdict: **GO** (with documented follow-ups)

The working tree is merge-ready for the NR P0 critical path: standalone ChipQA removal, unified HDR-ChipQA in groups, SimpleVQA SDR guards, and the opt-in SimpleVQA CUDA worker pass **final verification** (GPU exact parity, worker cold/reuse, HDR-ChipQA reference parity with configured venvs, fresh C++ rebuild). One non-blocking gap remains: native `pixvqa` does not forward `--use-worker` and should not expose a misleading flag until multi-video worker persistence exists in the C++ dispatch path.

---

## Branch status

| Item | Value |
|---|---|
| Branch | `feature/nr-metric-infrastructure` |
| Tracking | `origin/feature/nr-metric-infrastructure` (up to date) |
| HEAD (committed) | `abb12a7` Integrate ChipQA and HDR-ChipQA NR metrics… |
| Working tree | 25 modified paths, 5 deleted (`pixvqa_chipqa`), 12 untracked (see exclusions) |
| Built binary | `build/cpu/pixvqa` — **fresh rebuild** 2026-11-08 (`rm build/cpu/main.o && make cpu`); `--help` reflects HDR-ChipQA group changes |

---

## CUDA environment (final verification)

| Check | Result |
|---|---|
| `nvidia-smi` | **Not in PATH** (`Command 'nvidia-smi' not found` — driver utils package absent; not a PyTorch failure) |
| `.venv` `torch.cuda.is_available()` | **True** — `torch 2.11.0+cu128`, 1× `NVIDIA GeForce RTX 5080` |
| Diagnosis | **PyTorch CUDA runtime OK**; host `nvidia-smi` CLI missing. GPU regression tests ran successfully via PyTorch. |

---

## Tests executed

### Initial review (2026-11-08)

| Test | Result | Notes |
|---|---|---|
| `python -m py_compile` (`pixvqa_simplevqa/*`, `pixvqa_hdr_chipqa/*`) | **PASS** | |
| `scripts/test_simplevqa_worker.py` | **PASS** (8/8) | Mock lifecycle: startup/response timeout, crash restart, request IDs, atexit |
| `scripts/test_simplevqa_reference.py` | **PASS** | Preflight, staging, PQ/HLG rejection |
| `scripts/test_learned_metrics_cli.py` | **PASS** | Groups, QCS, device, hdr-chipqa+pu21, simplevqa |
| `scripts/test_chipqa_reference.py` | **PASS** (parity **SKIP**) | `PIXVQA_HDR_CHIPQA_*_PYTHON` unset |
| GPU SimpleVQA parity (`reports/.../test_regression.py`) | **SKIP** | `nvidia-smi` unavailable in WSL session |
| C++ rebuild in-session | **SKIP** | Existing `build/cpu/pixvqa` matches working-tree help text |

### Final pre-commit verification (2026-11-08 12:04 UTC+1)

| Test | Result | Notes |
|---|---|---|
| `python -m py_compile` (`pixvqa_simplevqa/*`, `pixvqa_hdr_chipqa/*`) | **PASS** | |
| `reports/simplevqa_single_decode_worker_20261108/test_regression.py` | **PASS** | CUDA exact score `2.9507508277893066`; worker exact parity; overlay/parse guards |
| SimpleVQA worker cold/reuse (inline) | **PASS** | cold **17.853 s**, warm **2.504 s**; scores exact match on reuse |
| SimpleVQA CPU smoke | **PASS** | score `2.951167345046997` (CPU path; differs from CUDA ref as expected) |
| `scripts/test_simplevqa_worker.py` | **PASS** (8/8) | |
| `scripts/test_simplevqa_reference.py` | **PASS** | |
| `scripts/test_chipqa_reference.py` (with venvs) | **PASS** | SDR BT.709 + PQ HDR parity; HLG contract rows |
| `scripts/test_learned_metrics_cli.py` | **PASS** | 119 OK, 2 SKIP (`pq -v` decode fixture; bundled `deps/hdr_chipqa` group case) |
| C++ `make cpu` (forced `main.o` rebuild) | **PASS** | links OK; one pre-existing `-Wunused-parameter` on `hdr_chipqa` in `apply_processing_defaults` |
| `build/cpu/pixvqa --version` / `--help` | **PASS** | `0.3.0-dev build:cpu decode:none`; metrics list includes `simplevqa`, `hdr-chipqa` |

**Exact commands (final verification):**

```bash
git status

# CUDA probe
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
which nvidia-smi || true

# SimpleVQA GPU + worker
PYTHONPATH=python .venv/bin/python reports/simplevqa_single_decode_worker_20261108/test_regression.py
PYTHONPATH=python .venv/bin/python scripts/test_simplevqa_worker.py
PYTHONPATH=python .venv/bin/python scripts/test_simplevqa_reference.py
PYTHONPATH=python .venv/bin/python -c '...'  # CPU smoke + worker cold/reuse (see session log)

# HDR-ChipQA parity (existing home venvs; no install)
export PIXVQA_HDR_CHIPQA_FEATURE_PYTHON=/home/sergio/.venvs/pixvqa-chipqa/bin/python
export PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON=/home/sergio/.venvs/pixvqa-chipqa-ref/bin/python
PYTHONPATH=python .venv/bin/python scripts/test_chipqa_reference.py

# C++ rebuild + CLI conformance
rm -f build/cpu/main.o build/cpu/pixvqa && make cpu
.venv/bin/python scripts/test_learned_metrics_cli.py
./build/cpu/pixvqa --version
./build/cpu/pixvqa --help
```

---

## Code review — correctness / integration / security

### Consolidated HDR-ChipQA (`src/main.cpp`)

- **Removed** standalone `chipqa` metric, `ChipqaOptions`, and `pixvqa_chipqa` package.
- **Groups** `sdr`/`pq`/`hlg` now auto-select `hdr-chipqa` (replaces CVVDP in group default).
- **CVVDP** is opt-in via explicit `--metrics …,cvvdp` (documented + tested).
- **PU21 + hdr-chipqa** no longer rejected; `group-pq-includes-hdr-chipqa-with-pu21` and `hdr-chipqa-with-pu21-accepted` tests confirm intentional behavior.
- **QCS:** `qcs_eligible_selected()` correctly excludes hdr-chipqa-only runs.

### SimpleVQA worker (`python/pixvqa_simplevqa/worker.py`, `worker_process.py`)

| Concern | Assessment |
|---|---|
| Startup timeout | Bounded via `_read_line_bounded`; default follows caller `timeout` |
| Per-request timeout | `_wait_response` with monotonic deadline |
| stdout/stderr deadlock | Dedicated reader/drainer threads + `select`; stderr tail capped at 8 KiB |
| Thread safety | `_SESSION_LOCK` on singleton; per-session `_lock` on score path |
| Request IDs | UUID hex; child echoes `id`; parent filters mismatched lines |
| Crash retry | Marks `_broken`, kills child; **next** `score()` restarts via `_ensure_running` (no same-call retry — acceptable) |
| CUDA persistence | Model loaded once in child; single-decode overlay staged in temp dir |
| Symlink staging | `model/`, checkpoint, overlay `test_demo.py` under `TemporaryDirectory` — no repo mutation |
| Shutdown | Graceful `shutdown` JSON + `atexit.register(shutdown_worker)` |

### SimpleVQA adapter (`adapter.py`)

- Default CUDA subprocess: **upstream symlink** (`symlink-unmodified`).
- Single-decode overlay: **worker-only** (`cuda_single_decode_overlay_test_demo`).
- CPU overlay: DataParallel removal + `map_location` (unchanged policy).
- `torch.compile`: **not in production**; research report rejects it (`reports/simplevqa_torch_compile_warm_20261108/REPORT.md`).

### HDR-ChipQA adapter

- Unified SDR/PQ/HLG dispatch in `adapter.py`; experimental HLG warning.
- Phase-1 feature accel shim (`feature_accel_shim.py`) — parity-reviewed, upstream untouched.
- Phase-2 rejected (NaN parity) — documented in `LEARNED_METRIC_PERF_PLAN.md`.

---

## Issues found

### Native CLI `--use-worker` — deferred (not misleading)

| ID | Severity | Path:line | Description |
|---|---|---|---|
| W1 | **Medium (P1)** | `src/main.cpp:2069-2084`, `1706-1784` | `simplevqa_command()` spawns a **fresh** `python -m pixvqa_simplevqa` subprocess per `execute_one()` via `fork`/`execvp` (`run_backend_process`). Each native invocation exits when the child returns — **no multi-video worker lifecycle**. Forwarding `--use-worker` today would only help the single video in that one subprocess (same as default CUDA subprocess path with symlink); it would **not** amortize model load across a batch. **Do not add** `--simplevqa-use-worker` to `pixvqa` until C++ keeps one long-lived adapter session. |

**Recommended implementation path (P1, post-merge):**

1. Add a `LearnedSession` (or metric-scoped daemon) in C++ that outlives one video — stdin/stdout JSON line protocol to `worker_process.py`, or a documented supervisor script.
2. Teach `execute_learned()` to reuse the session for consecutive SimpleVQA scores in one `pixvqa` process (and optional config/TOML toggle).
3. Only then wire `--simplevqa-use-worker` / `[simplevqa] use_worker` through `simplevqa_command()` and update public docs.
4. Until then, batch CUDA users call `python -m pixvqa_simplevqa --use-worker` directly (documented in `LEARNED_METRICS.md`, `SIMPLEVQA_REFERENCE.md`).

### Other non-blocking (documented / P1)

| ID | Severity | Path:line | Description |
|---|---|---|---|
| W2 | **Low** | `python/pixvqa_simplevqa/worker.py:309` | Session cache key hardcodes device `"cuda"`; harmless today (`use_worker` requires CUDA). |
| W3 | **Info** | `docs/dev/CHIPQA_REFERENCE.md:43-44` | Example paths use developer-specific `~/.venvs/…`; dev doc only. |

### Fixed in this review (doc)

| File | Change |
|---|---|
| `docs/CLI_USAGE.md` | Added `simplevqa` to `--metrics` list |
| `docs/METRICS_AND_FORMATS.md` | Added `simplevqa` capability row + worker note |
| `docs/LEARNED_METRICS.md` | Clarified `--use-worker` is Python-module-only |
| `docs/dev/SIMPLEVQA_REFERENCE.md` | New “Persistent CUDA worker” section |
| `config/learned_metrics.example.toml` | Added commented `[hdr-chipqa]` section |

### No blocking defects found

Worker lifecycle, adapter parity policy, group semantics, and test coverage meet P0 bar. No security regressions identified beyond standard learned-metric subprocess trust model.

---

## Documentation alignment

| Doc | Status |
|---|---|
| `CLI_USAGE.md` | Aligned (groups, hdr-chipqa, simplevqa in options table) |
| `LEARNED_METRICS.md` | Aligned |
| `METRICS_AND_FORMATS.md` | Aligned (simplevqa row added) |
| `QCS.md` | Aligned (hdr-chipqa pooled-only) |
| `dev/SIMPLEVQA_REFERENCE.md` | Aligned (worker section added) |
| `dev/CHIPQA_REFERENCE.md` | Aligned with hdr-chipqa-only path (pre-existing dev paths) |
| `dev/NR_METRIC_INTEGRATION_PLAN.md` | Notes chipqa removal (pre-existing edits) |

**Coverage limit:** PU21 native behavior not re-audited beyond changed-file grep; QCS/PU21 interaction unchanged outside hdr-chipqa QCS exclusion.

---

## Proposed grouped staging (no `git add` performed)

### Commit 1 — Remove legacy ChipQA; unify native CLI

```
src/main.cpp
python/pixvqa_chipqa/__init__.py          (delete)
python/pixvqa_chipqa/__main__.py          (delete)
python/pixvqa_chipqa/adapter.py           (delete)
python/pixvqa_chipqa/inference.py         (delete)
python/pixvqa_chipqa/numpy_legacy_shim.py (delete)
python/pyproject.toml
```

**Message (draft):** Consolidate on hdr-chipqa: remove standalone chipqa metric and package; auto-select HDR-ChipQA in sdr/pq/hlg groups and make CVVDP group opt-in.

### Commit 2 — HDR-ChipQA adapter hardening + perf shim

```
python/pixvqa_hdr_chipqa/__init__.py
python/pixvqa_hdr_chipqa/__main__.py
python/pixvqa_hdr_chipqa/adapter.py
python/pixvqa_hdr_chipqa/inference.py
python/pixvqa_hdr_chipqa/upstream_env.py
python/pixvqa_hdr_chipqa/feature_accel_shim.py   (new)
python/pixvqa_hdr_chipqa/numpy_legacy_shim.py    (new)
scripts/test_chipqa_reference.py
```

### Commit 3 — SimpleVQA worker + adapter overlays

```
python/pixvqa_simplevqa/__main__.py
python/pixvqa_simplevqa/adapter.py
python/pixvqa_simplevqa/worker.py                (new)
python/pixvqa_simplevqa/worker_process.py        (new)
scripts/test_simplevqa_reference.py
scripts/test_simplevqa_worker.py                 (new)
```

### Commit 4 — CLI conformance tests

```
scripts/test_learned_metrics_cli.py
```

### Commit 5 — Public + dev documentation

```
docs/CLI_USAGE.md
docs/LEARNED_METRICS.md
docs/METRICS_AND_FORMATS.md
docs/QCS.md
docs/dev/CHIPQA_REFERENCE.md
docs/dev/NR_METRIC_INTEGRATION_PLAN.md
docs/dev/SIMPLEVQA_REFERENCE.md
docs/dev/HDR_CHIPQA_VALIDATION_SUMMARY.md        (new, optional)
docs/dev/LEARNED_METRIC_PERF_PLAN.md              (new, optional)
config/learned_metrics.example.toml
```

### Commit 6 (optional) — Research reports only

```
reports/simplevqa_single_decode_worker_20261108/
reports/simplevqa_torch_compile_warm_20261108/   (exclude inductor_cache/)
reports/p0_closure_20261108/REPORT.md
```

### Explicit exclusions (do not stage)

| Path | Reason |
|---|---|
| `.local/` | Local scratch |
| `.venv-chipqa/` | Deployment venv |
| `docs/papers/` | External papers |
| `reports/**/inductor_cache/` | Torch compile artifacts |
| `deps/*` | Upstream — no modification policy |

---

## Follow-ups (post-P0, not blocking merge)

1. **P1:** Implement reusable multi-video SimpleVQA worker session in C++ dispatch, then expose `--simplevqa-use-worker` (see W1 path above).
2. ~~Re-run GPU parity~~ — **done** final verification 2026-11-08 (`test_regression.py` PASS on RTX 5080).
3. ~~HDR-ChipQA parity~~ — **done** with `/home/sergio/.venvs/pixvqa-chipqa{,-ref}/bin/python`; CI/docs should document env vars for other machines.
4. Generalize dev-doc venv paths in `CHIPQA_REFERENCE.md` to `$PIXVQA_HDR_CHIPQA_FEATURE_PYTHON` placeholders only.
5. Optional: install `nvidia-utils-*` in WSL for `nvidia-smi` host visibility (PyTorch CUDA already functional without it).

---

## Summary

| Area | Result |
|---|---|
| Code | **PASS** — no P0 blockers; W1 deferred with explicit C++ lifecycle rationale |
| Tests | **PASS** — fast regressions + GPU exact parity + HDR-ChipQA reference parity + fresh CPU binary |
| Build | **PASS** — forced `make cpu` rebuild; `test_learned_metrics_cli.py` against new binary |
| Docs | **PASS** — targeted repairs applied; no new doc edits required this session |
| Staging plan | **Ready** — 5–6 grouped commits proposed above |

**GO** for commit prep on `feature/nr-metric-infrastructure`. **No blockers.** Recommended next action: stage grouped commits per plan above (user approval); defer native `--use-worker` to P1 session supervisor work.
