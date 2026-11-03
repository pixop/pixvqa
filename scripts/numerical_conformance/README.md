# Numerical conformance (SDR/PQ/HLG UHD fixtures and baselines)

Synthetic **3840×2160**, **5 frames**, **60 fps**, **YUV420 10-bit limited** fixtures live under
`.cache/pixvqa-numerical-fixtures/` (gitignored). They are generated automatically; do not commit
`*.yuv` / `*.mp4` / `*.mkv`.

## Scene and color science

Module `scene.py` builds five frames of deterministic scene-linear RGB (gradient, shadow patch,
highlight, stripes, moving edge). Module `color_science.py` maps linear RGB to:

| Family | Display / encoding path |
|--------|-------------------------|
| SDR | BT.709 gamma (1/2.4) on linear, then BT.709 YCbCr limited 10-bit |
| PQ | ST 2084 PQ OETF on linear 0–10000 nits, BT.2020 YCbCr limited |
| HLG | BT.2100 HLG OETF, BT.2020 YCbCr limited |

PQ/HLG are **not** relabelled SDR samples; each family uses its own non-linear RGB before matrixing.

## Containers

| Asset | Container | Notes |
|-------|-----------|--------|
| `reference_lossless.*` | **MKV** (HEVC lossless Main10) | MP4 lossless HEVC is unreliable; MKV documented substitute |
| `distorted_lossy.mp4` | **MP4** (HEVC lossy) | Exercises MP4 + VUI |

Verified: decode(lossless MKV) == `reference.yuv`; decode(`distorted_lossy.mp4`) == `distorted.yuv`.

## Metric registry

See `registry.py` (mirrors `kMetricCaps` in `src/main.cpp` and `docs/METRICS_AND_FORMATS.md`).

For CVQA-HDR-NR and metric groups that include it, the suite also generates **60-frame** compressed
(and matching reference) variants (`*_60f.*`) because the official NR sampler requires at least
one second at 60 fps. The core 5-frame matrix is unchanged for other metrics.

After upgrading the harness, regenerate cached fixtures once:

```bash
# Smoke (5-frame core assets only)
python3 scripts/test_metric_numerical_conformance.py --profile smoke --force-fixtures

# Comprehensive (adds 60-frame CVQA-HDR-NR companions)
python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --force-fixtures
```

Generation streams **one frame at a time** (no full 60-frame RGB batch). Smoke does **not** build 60-frame assets.
Legacy manifests without `reference_yuv_60` remain valid for smoke; missing NR files are built only for comprehensive.

Peak memory: run `python3 scripts/test_numerical_fixture_cache.py` and read the
`INFO single-frame UHD encode` line (single-frame tracemalloc peak, typically under 650 MB).

```bash
export PIXVQA_LEARNED_PYTHON=/home/sergio/codeccomp/metrics/pixvqa/deps/learned-env/bin/python
# optional if not using config/learned_metrics.local.toml beside the binary
```

Optional VMAF override: `PIXVQA_VMAF_MODEL=/path/to/model.json`

## Commands (RTX 4070 WSL)

```bash
pip install numpy   # if missing
make -j8 full cpu swdec

export PIXVQA_BIN=build/full/pixvqa
export PIXVQA_CLI_CONFORMANCE_STRICT=1
export PIXVQA_NUMERICAL_STRICT=1

# Phase 1 only (42 CLI cases)
python3 scripts/test_metric_parameters.py

# Phase 2 smoke (fast numerical)
python3 scripts/test_metric_numerical_conformance.py --profile smoke --strict

# Phase 2 comprehensive
python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --strict

# Both phases
python3 scripts/run_pixvqa_validation.py --profile comprehensive --strict
```

## Baselines

Candidate results: `baselines/numerical_conformance/<profile>.candidate.json`  
Approve after review (regenerate candidates after parser or provenance changes; do not approve pre-fix `.candidate.json` files):

```bash
python3 scripts/test_metric_numerical_conformance.py --profile smoke --approve-baseline
```

Candidates include only metric fields selected per case, plus `learned_provenance` (CVQA checkpoint/profile SHA-256, CompressedVQA-HDR commit when available, CVVDP package versions). Approval fails if mandatory provenance is incomplete.

### Verify before approval (`--compare-candidate`)

Run the comprehensive suite read-only against a verified `.candidate.json` without approving it:

```bash
python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --strict \
  --compare-candidate baselines/numerical_conformance/comprehensive.candidate.json \
  --json reports/numerical_conformance_comprehensive.json
```

This adds `candidate_reproducibility: PASS|FAIL` (or `NOT_RUN` without the flag). It compares the **95 numerical metric cases** only (not the six internal-consistency rows), checks strict artifact provenance (fixtures, learned-model hashes, dependency versions, tolerance profile), and leaves `numerical_regression: NOT_EVALUATED` until you `--approve-baseline`. The compare path is never overwritten; if it equals the default auto-save path, the suite skips regenerating that file during the compare run.

When the candidate’s recorded `pixvqa_commit` differs from the current HEAD, reproducibility can still **PASS** only if a verified `git diff` shows that every changed path is on an explicit infrastructure allowlist (`candidate_commit_compatibility.py`, policy `numerical_conformance_infrastructure_allowlist_v1`). The JSON report records both commits, the changed-file list, and `compatibility_exception` — commit mismatch is never silent. Unresolved commits, diff failures, or any change outside the allowlist (metric code, decoders, fixtures, learned wrappers, unknown paths) **FAIL** closed. Extend the allowlist only with auditable harness changes; do not use broad `scripts/` prefixes.

Approved baselines are **never** overwritten by normal runs.

## Validation levels (do not conflate)

| Level | Meaning | Verdict when not applicable |
|-------|---------|------------------------------|
| **Execution conformance** | PixVQA exits 0; applicable mandatory cases yield finite parsed metrics. | Fails on mandatory `FAIL`; `NA` for unsupported product combinations. |
| **Numerical regression** | Observed values vs an **approved** baseline (`tolerances.py`). | `NOT_EVALUATED` until a baseline is approved. |
| **Internal numerical consistency** | Same-build PixVQA cross-path checks: HDR inferred vs explicit PSNR, raw vs compressed PSNR, CPU vs GPU PSNR, software vs NVDEC decode. | `NOT_RUN` on smoke / when checks are not executed. |
| **Candidate reproducibility** | Read-only `--compare-candidate` vs verified unapproved JSON (95 metric cases + provenance). | `NOT_RUN` without the flag; `PASS`/`FAIL` when comparing. |

JSON reports expose all four under `validation_levels`. Internal consistency **PASS** does **not** imply independent reference parity.

### Optional BerlinSeqs CVVDP check

Not part of portable fixtures. Enabled with `PIXVQA_BERLIN_REF` and `PIXVQA_BERLIN_TEST`. Compares PixVQA CVVDP JOD to **pinned constants** in `berlin.py` (5f=`8.788558959960938`, 60f=`9.17333984375`). This is an optional **reproducibility** check scoped to **CVVDP only**; constants are **not** documented as independently verified in the harness. Does not imply parity for PSNR, VMAF, CVQA, or other metrics.
