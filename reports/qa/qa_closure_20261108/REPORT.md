# QA Closure — Harness Fixes, Fresh Builds, Full Regression

**Date:** 2026-11-08 12:52 UTC+1  
**Branch:** `feature/nr-metric-infrastructure` @ `abb12a7` (unstaged working tree)  
**Prior session:** [full_local_20261108/REPORT.md](../full_local_20261108/REPORT.md)  
**Action:** fixed four stale conformance harness gaps, rebuilt `swdec`/`full`/`cuda`/`cpu` against current `main.cpp`, re-ran official QA and regressions; **no git add / commit / push**

## Verdict: **GO**

All previously failing ancillary harness scripts now **PASS**. Official `run_qa.py` full (236 PASS / 0 FAIL / 1 SKIP) and hardware (3 PASS) profiles pass against fresh decode and CUDA binaries. NR-path regressions (SimpleVQA, HDR-ChipQA, learned CLI) pass.

---

## Harness fixes (root cause vs product defect)

| Issue | Classification | Fix |
|---|---|---|
| `cvvdp-sdr-raw-yuv` expected reject | **Stale harness** — product accepts SDR CVVDP on raw YUV420p10le (`test_cvvdp_cli.py` positive path) | Changed case to **accept** with `interpretation=sdr` / `standard_fhd`; dropped obsolete `cvvdp_skip_reason` for SDR raw in numerical registry |
| `cvvdp-pq-hlg-stream-disagree` stderr mismatch | **Stale fixture** — `make_hlg_mp4` lacked in-stream VUI; probe reported “no transfer characteristic tag” instead of disagree | Added `setparams=…arib-std-b67…` to `make_hlg_mp4` (parity with PQ fixture) |
| `test_numerical_cli_invoke` `--cvqahdr-python` | **Stale expectation** — NR Python resolves from `learned_metrics` config only | Assert `--cvqahdr-python` and `PIXVQA_CVQAHDR_PYTHON` **absent** on cvqa-only invocations |
| `test_cli_help` `standard_hdr_pq` / examples 0&2 | **Stale needles + env** — `--help-hdr` text updated for HDR-ChipQA groups; examples need ffmpeg on PATH | Match current `--help-hdr` needles; check `standard_hdr_*` in `--help-advanced`; `shutil.which("ffmpeg")` fallback |
| `test_metric_numerical_conformance` with `PIXVQA_BIN=build/cpu` | **Wrong default binary** for compressed smoke matrix | Auto-fallback in `discover()` when selected binary reports `decode: none` (documented in script epilog) |

No product assertions were weakened; negative-input guarantees retained (e.g. `cvvdp-pq-hlg-stream-disagree` still requires explicit transfer).

---

## Build artifacts (fresh)

| Binary | mtime (local) | `--version` |
|---|---|---|
| `build/cpu/pixvqa` | 2026-11-08 12:47 | `build: cpu` / `decode: none` |
| `build/swdec/pixvqa` | 2026-11-08 12:48:57 | `build: cpu` / `decode: software` |
| `build/full/pixvqa` | 2026-11-08 12:48:24 | `build: cuda` / `decode: software nvdec` |
| `build/cuda/pixvqa` | 2026-11-08 12:46:46 | `build: cuda` / `decode: none` |

Rebuild notes:

- Targeted `main.o` refresh for `swdec`, `full`, `cuda`; full relink after `src/main.cpp` changes.
- **Build env:** `LIBVMAF_PREFIX` + `PKG_CONFIG_PATH` (libvmaf + zimg pc files) required so `HAS_VMAF=1` on cpu/swdec/full. Without them, `make` can silently produce vmaf-less binaries and QA vmaf cases fail.
- `FFMPEG_PREFIX=/home/sergio/ffmpeg-n8.1.3-lgpl-p5` for decode variants; `GCC13`/`CUDA_HOME=/home/sergio/cuda-toolkit` for `full`.

Prior QA JSON backed up under `prior/` before this session overwrote `reports/qa/full.json` and `hardware.json`.

---

## Official QA

### `make test-cpu` — **PASS**

```bash
LIBVMAF_PREFIX=... PKG_CONFIG_PATH=... make test-cpu
```

Log: [make_test_cpu.log](./make_test_cpu.log)

### `run_qa.py --profile full` — **PASS**

```bash
env -u LD_LIBRARY_PATH python3 scripts/run_qa.py --profile full --cpu build/cpu/pixvqa
```

| PASS | FAIL | SKIP |
|---:|---:|---:|
| 236 | 0 | 1 (optional valgrind) |

Logs: [run_qa_full_final2.log](./run_qa_full_final2.log), [full.json](./full.json)

### `run_qa.py --profile hardware` — **PASS**

```bash
env -u LD_LIBRARY_PATH PATH=/usr/lib/wsl/lib:$PATH \
  python3 scripts/run_qa.py --profile hardware --cpu build/cpu/pixvqa
```

| PASS | FAIL | SKIP |
|---:|---:|---:|
| 3 | 0 | 0 |

Log: [run_qa_hardware_final.log](./run_qa_hardware_final.log), [hardware.json](./hardware.json)

---

## Previously failing harness scripts (now fixed)

| Script | Was | Now | Log |
|---|---|---|---|
| `test_metric_cli_conformance.py` | 39 PASS / 2 FAIL | **41 PASS / 0 FAIL** | [regression/test_metric_cli_conformance.log](./regression/test_metric_cli_conformance.log) |
| `test_numerical_cli_invoke.py` | AssertionError `--cvqahdr-python` | **all OK** | [regression/test_numerical_cli_invoke.log](./regression/test_numerical_cli_invoke.log) |
| `test_cli_help.py` | 3 FAIL | **PASS** (after `cuda` help sync rebuild) | [regression/test_cli_help.log](./regression/test_cli_help.log) |
| `test_metric_numerical_conformance.py` | 9 FAIL with `PIXVQA_BIN=build/cpu` | **15 PASS** (explicit swdec + auto-fallback) | [regression/test_metric_numerical_conformance_*.log](./regression/) |

---

## NR / learned-metric & CUDA regressions

| Suite | Status |
|---|---|
| `py_compile` (simplevqa + hdr_chipqa) | **PASS** |
| `test_learned_metrics_cli.py` | **PASS** (129 OK, 1 SKIP: bundled hdr_chipqa repo) |
| `test_simplevqa_worker.py` | **PASS** |
| `test_simplevqa_reference.py` | **PASS** |
| `test_chipqa_reference.py` | **PASS** |
| `simplevqa_single_decode_worker` regression | **PASS** |
| `test_numerical_baselines.py` | **PASS** |
| `test_cuda_numeric.py` / `test_cuda_psnr_ssim_8bit.py` / `test_pu21_cuda.py` | **PASS** |
| `test_pu21_numeric.py` / `test_vmaf_numeric.py` | **PASS** |

Logs: [regression/](./regression/)

---

## Source changes (harness only this session)

| File | Purpose |
|---|---|
| `scripts/metric_cli_cases.py` | SDR raw CVVDP → positive case |
| `scripts/metric_test_fixtures.py` | HLG mp4 in-stream VUI via `setparams` |
| `scripts/numerical_conformance/registry.py` | Remove obsolete SDR-raw CVVDP skip |
| `scripts/numerical_conformance/environment.py` | Decode-capable binary auto-fallback |
| `scripts/test_metric_numerical_conformance.py` | Document `PIXVQA_BIN` decode requirement |
| `scripts/test_numerical_cli_invoke.py` | NR argv expectations |
| `scripts/test_numerical_baselines.py` | NA classification update |
| `scripts/test_cli_help.py` | Help needles + ffmpeg discovery |

Pre-existing branch edits (`src/main.cpp`, NR adapters, docs) preserved unchanged except via rebuild.

`git diff --check`: **clean** (CRLF warnings only on Windows-checked-out paths).

---

## Delta vs [full_local_20261108](../full_local_20261108/REPORT.md)

| full_local (prior) | qa_closure (this) |
|---|---|
| 4 harness FAIL suites (non-blocking) | All four **PASS** |
| `swdec`/`full` not rebuilt (Nov-7 mtimes) | Fresh rebuild Nov-8 with vmaf+ffmpeg |
| GO with harness caveats | **GO** without harness caveats |

---

## Merge decision

| Gate | Result |
|---|---|
| Harness discrepancy closure (4 scripts) | **PASS** |
| Fresh `swdec` + `full` linked to current `main.cpp` | **PASS** |
| Full QA profile | **PASS** |
| Hardware QA profile | **PASS** |
| NR metric regressions | **PASS** |

### **GO** for commit prep on `feature/nr-metric-infrastructure`

---

## Quick reproduce

```bash
# Preserve prior QA artifacts
mkdir -p reports/qa/qa_closure_20261108/prior
cp reports/qa/full.json reports/qa/hardware.json reports/qa/qa_closure_20261108/prior/

export LIBVMAF_PREFIX=~/pixvqa-deps/libvmaf-74f930d
export PKG_CONFIG_PATH="$LIBVMAF_PREFIX/lib/x86_64-linux-gnu/pkgconfig:deps/zimg-3.0.6/lib/pkgconfig"
export FFMPEG_PREFIX=~/ffmpeg-n8.1.3-lgpl-p5
export GCC13=~/cuda-toolkit CUDA_HOME=~/cuda-toolkit

rm -f build/swdec/main.o build/full/main.o build/cuda/main.o
make swdec full cuda
make test-cpu

.venv/bin/python scripts/test_metric_cli_conformance.py
.venv/bin/python scripts/test_numerical_cli_invoke.py
.venv/bin/python scripts/test_cli_help.py
PIXVQA_BIN=build/swdec/pixvqa .venv/bin/python scripts/test_metric_numerical_conformance.py

env -u LD_LIBRARY_PATH python3 scripts/run_qa.py --profile full --cpu build/cpu/pixvqa
env -u LD_LIBRARY_PATH PATH=/usr/lib/wsl/lib:$PATH python3 scripts/run_qa.py --profile hardware --cpu build/cpu/pixvqa
```

Machine-readable rollup: [results.json](./results.json)
