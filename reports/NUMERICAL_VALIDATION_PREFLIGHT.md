# PixVQA numerical validation preflight (`feature/unified-metric-parameters`)

## RTX 4070 results (execution phase — prior run)

| Suite | PASS | FAIL | SKIP | NA |
|-------|-----:|-----:|-----:|---:|
| CLI conformance (`test_metric_parameters.py`) | 42 | 0 | 0 | 0 |
| Numerical comprehensive | 101 | 0 | 7 | — |

Native metrics, CVVDP, CVQA-HDR-NR, and mixed metric groups **execute** successfully on RTX 4070.

## Three validation levels (report separately)

### 1. Execution conformance

**Question:** Did PixVQA complete successfully with finite parsed metrics for every applicable mandatory case?

- **Verdict (RTX 4070 comprehensive):** **PASS** (101 PASS, 0 FAIL on executed matrix).
- **Strict note:** Mandatory `SKIP` (missing deps / optional GPU paths) fails `--strict`; mandatory **`NA`** (product-unsupported combinations) does **not** fail strict.

### 2. Numerical regression conformance

**Question:** Do observed metric values match an **approved** baseline within explicit per-field tolerances?

- **Verdict:** **NOT_EVALUATED** until `baselines/numerical_conformance/comprehensive.json` is explicitly approved (`--approve-baseline` after independent review).
- Candidate baselines are written to `*.candidate.json` only; never auto-promoted.
- When approved: missing mandatory cases, missing fields, non-finite values, or tolerance violations → **FAIL** (no silent bypass).

### 3. Independent reference parity

**Question:** Do cross-path checks and optional external references agree within parity tolerances?

- Includes: raw vs comp PSNR consistency, HDR inferred vs explicit, CPU vs GPU PSNR, SW vs NVDEC decode, optional BerlinSeqs CVVDP JOD (`berlin.py`, `reference_parity` tolerance).
- **Separate** from baseline regression; Berlin optional when assets unset.

## N/A vs SKIP (seven comprehensive skips)

| Classification | Meaning |
|----------------|---------|
| **NA** | PixVQA product/design unsupported (e.g. SDR raw CVVDP, CVQA on `raw_raw`, metric groups on SDR/PQ `raw_raw` with CVQA). |
| **SKIP** | Environment optional path (CVQA deps missing, optional GPU decode/backend not run). Not counted as numerical regression pass. |

Do not reclassify checkpoint failures, execution errors, or baseline mismatches as N/A.

## RTX 4070 commands (WSL)

```bash
pip install numpy
make -j8 full cpu swdec

export PIXVQA_BIN=build/full/pixvqa
export PIXVQA_CLI_CONFORMANCE_STRICT=1
export PIXVQA_NUMERICAL_STRICT=1
export PIXVQA_LEARNED_PYTHON=/path/to/learned-env/bin/python

python3 scripts/test_metric_parameters.py
python3 scripts/test_numerical_baselines.py
python3 scripts/test_metric_numerical_conformance.py --profile smoke --strict
python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --strict
python3 scripts/run_pixvqa_validation.py --profile comprehensive --strict
```

JSON report: `reports/numerical_conformance.json` → read `validation_levels`.

## Baseline approval (explicit — not done in this change)

```bash
python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --approve-baseline
```

## GO / NO-GO for baseline approval campaign

| Gate | Status |
|------|--------|
| Execution conformance on RTX 4070 | **GO** |
| Harness strict numerical regression | **GO** (after this commit) |
| Approved baseline on disk | **NO-GO** (awaiting review) |
| Full numerical regression verdict | **NOT_EVALUATED** until approval |
