# Learned-metric performance plan (baseline → optimization)

**Status:** baseline captured 2026-11-07; **HDR-ChipQA adapter shim optimization (Phase 1 only)** measured 2026-11-07 (~3.0 % controlled A/B wall-clock, exact feature parity) — see [`reports/hdr_chipqa_perf_opt_20261107/REPORT.md`](../../reports/hdr_chipqa_perf_opt_20261107/REPORT.md). Phase 2 candidates (`aggd_features` table lookup, `Y_compute_lnl` 2-D scipy filters) **rejected** 2026-11-08 for NaN/non-finite features and SHA-256 parity failure — see [`reports/hdr_chipqa_perf_opt_20261107/phase2/REPORT.md`](../../reports/hdr_chipqa_perf_opt_20261107/phase2/REPORT.md). **Optimization sprint 2026-11-08** (Numba AGGD, GPU filters, SimpleVQA CUDA/CPU overlays) — **no new production changes** — see [`reports/learned_metric_optimization_20261108/REPORT.md`](../../reports/learned_metric_optimization_20261108/REPORT.md). Baseline: [`reports/learned_metric_perf_baseline_20261107/REPORT.md`](../../reports/learned_metric_perf_baseline_20261107/REPORT.md).

## Goals

1. Establish reproducible wall-clock and RSS before P0 integration sign-off.
2. Identify parity-safe optimization targets (no score drift).
3. Separate **environment blockers** from **code-path bottlenecks**.

## Measurement protocol

| Item | Policy |
|------|--------|
| Clip | 60-frame SDR HEVC (`reports/chipqa-validation/sdr/cityalley_crf24_60f.mp4`) unless metric requires PQ |
| Runs | 1 cold + 1 warm per metric (`/usr/bin/time -v`) |
| Staging | Adapter-level timers in `profile_learned_metrics.py` (no upstream edits) |
| Score check | Compare staged vs `adapter.run()` JSON `quality_score` |
| Host note | WSL `/mnt/d` adds I/O latency; prefer ext4 for regression timing |

## HDR-ChipQA — confirmed bottleneck

**~98% of time in `hdr_chipqa.py` feature extraction** (21.7 s of ~22 s on 960×540×60).

Optimization phases (each requires golden re-check):

1. **Profile upstream** — `cProfile` / `py-spy` on feature env, pinned commit `916157d`. ✅
2. **Adapter shim (Phase 1)** — `paired_product`, `estimateggdparam`, `yuv_read` bulk; ~3.0 % controlled A/B, SHA-256 parity. ✅ shipped
3. **Adapter shim (Phase 2)** — `aggd_features` gamma tables + 2-D scipy filters for `Y_compute_lnl`. ❌ **rejected** (NaN/inf, parity fail; no env flag retained)
4. **Decode cache** — reuse temp YUV when pixvqa runs multiple metrics on one test path.
5. **Interpreter consolidation** — evaluate merging feature + npy bridge if sklearn pin allows (currently split 3.12 / 3.10).
6. **Batch / scale** — document linear scaling vs resolution×frames before any subsampling.

## SimpleVQA — environment dependency

Upstream `test_demo.py` is **CUDA-first**:

- unconditional `DataParallel` (lines 191–194)
- `torch.load` without `map_location` (CUDA checkpoint keys)

CPU-only hosts fail before a valid score with the stock upstream demo. **GPU baseline recorded 2026-11-08** on WSL + RTX 5080 — see [`reports/learned_metric_optimization_20261108/REPORT.md`](../../reports/learned_metric_optimization_20261108/REPORT.md).

| Item | Official 8 s UGC clip (`2999049224_…_8s.mp4`) |
|------|-----------------------------------------------|
| Score | `2.9507508277893066` |
| Wall | ~22.3 s (warm), peak RSS ~1.42 GiB |
| Clip for regression | Prefer official 8 s sample; cityalley 60f/180f hit upstream spatial/motion mismatch |

Low-risk CUDA overlay (no DataParallel + `inference_mode`) achieved **~1.7%** median gain with exact score parity — **below 5% ship bar, not integrated**. Research-only CPU overlay reaches parity (Δ ≈ 4×10⁻⁴) but is not shipped.

```bash
PYTHONPATH=python .venv/bin/python -m pixvqa_simplevqa \
  --repo deps/SimpleVQA --checkpoint deps/SimpleVQA/ckpts/UGC_BVQA_model.pth \
  --test deps/SimpleVQA/videos/2999049224_original_centercrop_960x540_8s.mp4 \
  --device cuda --transient-output
```

Expected dominant stages: SlowFast pretrained load (cold), per-clip motion encoding, ResNet spatial branch.

## Integration instrumentation (future, opt-in)

Proposed env-gated hooks (not implemented):

- `PIXVQA_PROFILE=1` → JSON stage timings on stderr from adapters
- No change to subprocess argv or upstream CWD semantics

## Exit criteria for “performance P0”

- [x] HDR-ChipQA: documented profile + **~3.0 %** controlled A/B improvement (Phase 1 shim only, exact SHA-256 parity); Phase 2 + Numba AGGD sprint experiments rejected — remaining hotspot is upstream `aggd_features` / `Y_compute_lnl`
- [x] SimpleVQA: GPU baseline on official 8 s clip — score `2.9507508277893066`, ~22.3 s wall, ~1.42 GiB RSS; CUDA overlay <5% → not shipped
- [x] Golden scores unchanged on validated clip; cityalley short clips documented as invalid for upstream single-scale forward
- [ ] `run_baseline.sh` passes in CI-like ext4 checkout (optional nightly)
