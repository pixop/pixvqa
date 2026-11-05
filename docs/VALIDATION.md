# Metric validation and regression testing

This guide is the entry point for PixVQA validation beyond basic installation QA. Run commands from the repository root. For fixture generation, tolerances, provenance schema and advanced workflows, see the [numerical conformance reference](../scripts/numerical_conformance/README.md).

## Which validation should I run?

| Goal | Command | Notes |
|---|---|---|
| Check an installation | `python3 scripts/run_qa.py --profile smoke` | Basic smoke QA, not full metric validation |
| Check CLI contracts | `python3 scripts/test_metric_cli_conformance.py` | 42-case CLI matrix; requires appropriate built binaries |
| Test parameter resolution | `python3 scripts/test_metric_parameters.py` | Separate unit/integration tests, not the 42-case matrix |
| Check numerical conformance | `python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --strict` | UHD SDR/PQ/HLG, learned metrics if installed, approved numerical baseline |
| Orchestrate Phase 1 + Phase 2 | `python3 scripts/run_pixvqa_validation.py --profile comprehensive --strict` | Phase 1 runs the 42-case CLI matrix, then parameter resolver tests (Phase 1b), then numerical conformance. Use `--skip-parameter-tests` to omit Phase 1b. |

## Prerequisites

For comprehensive GPU validation, build the required variants and ensure FFmpeg and the optional learned-metric backends are available:

```bash
make -j8 cpu swdec full
export PIXVQA_BIN="$PWD/build/full/pixvqa"
export PIXVQA_LEARNED_PYTHON="/path/to/learned-env/bin/python"
python3 scripts/test_metric_cli_conformance.py
python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --strict \
  --json reports/numerical_conformance_comprehensive.json
```

The CLI conformance runner uses CPU and software-decoder binaries as configured by the harness; `PIXVQA_BIN` and `PIXVQA_DECODE_BIN` may be used to override its defaults. `PIXVQA_BIN` should point to a binary with the capabilities required by the numerical test (normally `build/full/pixvqa`). The guided installer instead puts a selected executable at `bin/pixvqa`. These paths are not interchangeable unless their build capabilities match.

Learned metric installation and local model paths are documented in [Learned metrics](LEARNED_METRICS.md). See [learned prerequisites for numerical validation](../scripts/numerical_conformance/LEARNED_DEPS.md) for missing-dependency behavior.

## Five validation levels

The comprehensive JSON report contains five entries under `validation_levels`:

| Key | What it establishes | Expected when not requested |
|---|---|---|
| `execution_conformance` | Applicable metric cases execute and return parsed results | Mandatory execution failures fail the suite |
| `numerical_regression` | Results match an **approved** baseline using per-metric tolerances | `NOT_EVALUATED` before approval |
| `internal_numerical_consistency` | Same-implementation cross-path checks (HDR inferred/explicit, raw/compressed, CPU/GPU, SW/NVDEC) | `NOT_RUN` where no checks execute |
| `independent_reference_parity` | Scoped optional external-reference checks; BerlinSeqs currently checks CVVDP JOD against pinned constants **not independently certified in this harness** | `NOT_EVALUATED` without optional Berlin inputs |
| `candidate_reproducibility` | Read-only comparison with a supplied unapproved baseline candidate | `NOT_RUN` without `--compare-candidate` |

**PASS for numerical regression means the implementation reproduces an approved snapshot, not that scores match an independent authoritative implementation.** Internal numerical consistency is likewise not independent parity.

## Approved comprehensive baseline

The tracked comprehensive baseline is [`baselines/numerical_conformance/comprehensive.json`](../baselines/numerical_conformance/comprehensive.json) with 95 numerical cases and learned-model provenance. It was approved after RTX 4070 execution and candidate-repeatability checks in the unified-parameters development campaign. The original baseline's Git provenance is intentionally preserved; subsequent infrastructure-only revisions do not imply that metrics were recalculated.

For normal validation, **do not** use `--approve-baseline` or `--compare-candidate`:

```bash
python3 scripts/test_metric_numerical_conformance.py --profile comprehensive --strict \
  --json reports/numerical_conformance_comprehensive.json
```

To inspect a new **unapproved** candidate without overwriting it, see the [candidate comparison instructions](../scripts/numerical_conformance/README.md#verify-before-approval---compare-candidate). Approved baseline changes require a separate review and explicit approval command. Do not regenerate or approve baselines as part of routine validation.

## PASS, FAIL, SKIP and NA

- **PASS:** The check completed and met its applicable condition.
- **FAIL:** Execution, numerical comparison or mandatory validation failed.
- **SKIP:** A check did not execute because of missing dependencies or environmental conditions; mandatory skips fail `--strict`.
- **NA:** A metric/input combination is unsupported by the product contract; not an infrastructure failure.

Seven combinations are NA in the comprehensive matrix:

| Case ID | Reason |
|---|---|
| `sdr.raw_raw.cvvdp` | CVVDP does not support SDR raw YUV in this interface |
| `sdr.raw_comp.cvvdp` | CVVDP does not support SDR raw reference input |
| `sdr.raw_raw.cvqa-hdr-nr` | CVQA-HDR-NR requires compressed test input |
| `pq.raw_raw.cvqa-hdr-nr` | CVQA-HDR-NR requires compressed test input |
| `sdr.raw_raw.group_sdr` | The SDR group includes CVQA-HDR-NR, which requires compressed test input |
| `sdr.raw_comp.group_sdr` | The SDR group includes CVVDP, which does not support SDR raw reference input |
| `pq.raw_raw.group_pq` | The PQ group includes CVQA-HDR-NR, which requires compressed test input |

These are **not** seven skipped checks. Missing model files or backend dependencies must not be recategorized as NA.

## Relationship to release validation

`make release-check` runs the [release qualification stages](RELEASE_VALIDATION.md). It is **not documented as a substitute** for the two-phase CLI/numerical campaign. Consult both workflows when planning a release or merge gate.
