# CVQA-HDR-NR MOS calibration: audit dispositions and implementation verification

Date: 2026-11-14
Branch: `feature/cvqa-hdr-nr-mos-calibration`
Status: **selected research model integrated into this working branch; not production-subjective-quality certified.**

## Selected model (subsequent to the independent audit)

For HDR PQ, the selected **domain-aware logistic MOS calibration** is the Original-U-trained full-392 fit with **high subjective MOS-target weighting, lambda = 1.0**. For SDR it is the U-v2-trained full-392 fit with the same weighting. These are **not** the previous unweighted fitted coefficients. Lambda 1.0 doubles the base training weight for synthetic target MOS >= 4.5, with linear increase between 3.5 and 4.5:

`w' = w * [1 + clip((MOS_target - 3.5)/1.0, 0, 1)]`

Both models were taken from the **`original_ids` / `target` / `lambda_weight=1.0`** rows of:

`perceptual_optimization/calibration/production_readiness_20261113/mos_target_reweight_20261113/artifacts/endpoints.csv`.

The separate `nighttraffic_merge_sensitivity` fit yields *exactly identical full-392 model parameters* for each selected weighted route; source-family grouping changes held-out cross-validation predictions, not the stored final fit.

The actual 4-parameter M1 sigmoid is:

```text
x_c = clip(x, 4.1344840888613135, 10.051497077192835)
z = (x_c - 8.382293636852387)/1.1647912353499874
b_eff = max(0,b)
s = max(0.001, exp(log_scale))
base = clip(1 + 4*sigmoid(a + b_eff*(z - c)/s), 1, 5)
MOS = clip(base + gamma_PQ*(PQ) + gamma_2160*(h==2160), 1, 5)
```

The canonical CVQA `quality_score` is not modified. MOS is an additional field when stream metadata unambiguously supports SDR or PQ and height 1080/2160.

## Reviewer issues: disposition

| Item | Disposition in this branch |
|---|---|
| A01: incorrect two-parameter port | **Fixed and checked.** Uses all four serialized M1 parameters, golden endpoint values for all 4 routes and 68 separately saved AVT predictions. Never use the defective uploaded implementation. |
| A02: Python/C++ quoted status protocol mismatch | **Fixed.** Python emits unquoted status tokens; C++ parser accepts current and legacy quoted tokens. Both actual C++ summary branches were exercised in a compiled harness. |
| A03: potential NightTraffic cross-fold leakage | **Addressed in research sensitivity, not a runtime change.** Conservative co-grouping of 6 HDR + 4 SDR observations gives 55 source-family groups. The paired 5-outer/4-inner rerun shows minor out-of-fold metric changes. Similar content identity is not hash-proven. Do not assert universal content-level independence. |
| A04: cross-study subjective-scale bridge | **Disclosed scientific limitation.** 120 CVQA-score-matched interpolated pseudo-pairs from only three families are not 120 identical human-rated video pairs. |
| A05: reused AVT model-selection evidence | **Disclosed.** AVT influenced U-v2 and routing. Apparent AVT test metrics are exploratory, not blind independent confirmation. |
| A06: AVT stimulus and metadata discrepancies | **Not resolved by code.** HDR: 8 rated observations from two 2160p source families. SDR primary: 54 ratings from 5 families, 51 unique encoded files, with 4:2:2 and some 1080p-to-4K subjective presentation differences. VD HEVC: all six scored distributed files are **physically 2160p** even when three filenames encode an earlier 1080p source condition. Missing transfer tags cause automatic MOS to be unavailable; filenames are never used to bypass the metadata gate. |
| A07: metadata probe and optional-stage robustness | **Fixed.** ffprobe timeout, typed payload checks, geometry validation and explicit failure reasons. An exception caught only around optional MOS calibration preserves canonical CVQA and writes an unavailable reason. |
| A08: provenance, enablement, reproducibility | **Fixed.** Versioned model metadata, SHA-256 digest of both M1 coefficients, gammas, normalization, support, ridge and weighting; automatic MOS behavior documented (not opt-in). Working-tree code must be committed separately to give an immutable implementation revision. |
| A09: misleading initial finalist description | **Corrected in documentation.** The initial holdout validation chose plain M7 by source-balanced MAE; domain-aware M7-Bias was subsequently retained as a research candidate. Labelled **domain-aware logistic MOS calibration** for readers. Bias values refer to *signed* predicted minus subjective MOS. |

### Weighting trade-off

The lambda=1.0 HDR option was explicitly selected as an **upper-quality-range policy preference**, not because it wins overall HDR native-MOS error. On reused AVT ratings, HDR8 MAE changes **0.5167 -> 0.5439** and bias **+0.3803 -> +0.4504**. SDR54 improves from **0.5107 -> 0.4147** MAE and bias **-0.5025 -> -0.3968**. The source-disjoint development OOF MAE slightly worsens under target weighting, and the highly rated training-target tail is sparse. These remain research evidence.

### Unresolved upper-range policy

**No new extrapolation or MOS 4.5 cap was silently introduced.** The current implementation retains the existing input-support clamp and output clamp [1,5]. Under these decisions, the maximum produced MOS values at input 10.051497 are HDR1080 **3.9392**, HDR2160 **4.0938**, SDR1080 **4.2155**, SDR2160 **4.3834**. The user discussed allowing monotonic extrapolation above support and/or capping at MOS 4.5, but no final distinct implementation policy was authorized. This matters especially for HDR1080: the selected +100% fit alone does not yield MOS > 4 inside support.

## Verified checks

- **42** dedicated Python MOS tests passed, including exact endpoints and all five native AVT metrics (MAE, RMSE, signed bias, PLCC, SRCC) reconstructed from the frozen 68-case fixture.
- Independent read-only parity of **68/68** selected-route saved research predictions against live `predict_mos`: **maximum absolute difference 0.0**.
- **4/4** chunking tests passed in the configured PyTorch Conda environment.
- `make cpu` and `make test-cpu` passed; golden-log and production-parser compatibility scripts passed.
- A compiled harness directly exercised C++ `parse_backend_report` -> `print_learned_summary`: unquoted available, legacy quoted available, and unavailable yielded the expected output.
- Python modules compile successfully; `git diff --check` passes.

A fresh real-video CVQA GPU inference and native subjective-quality blind study were **not** performed as part of this integration. Neither the passing software tests nor the chosen extrapolation policy establishes independent perceptual validation.

## Reviewed S1 opt-in (2026-11-14 follow-up)

Independently reviewed policy `reviewed_s1_hdr0_sdr1` (HDR Original_U baseline λ=0, SDR U_v2 target λ=1) is integrated as an **opt-in** via `PIXVQA_CVQA_MOS_POLICY`. Default remains `legacy_target_lambda1_both` (both routes λ=1). Sealed golden 68-row fixture and dual-policy regression tests added. **Hardening (2026-11-14):** reviewed policy fail-closed upstream pin gate in adapter (`gate_reviewed_mos_outcome`), `embedded_coefficients_sha256`, invalid-score type handling. Details: [CVQA_MOS_REVIEWED_S1_INTEGRATION.md](CVQA_MOS_REVIEWED_S1_INTEGRATION.md).

## Implementation paths

- `python/pixvqa_cvqahdr/mos_calibration.py`: dual policies, equations, metadata gate, provenance digest
- `python/pixvqa_cvqahdr/adapter.py`: retain canonical score on optional MOS failure
- `python/pixvqa_cvqahdr/__main__.py`: stable protocol tokens and record output
- `src/main.cpp`: compatible C++ status parsing and CLI summary
- `scripts/test_cvqahdr_mos_calibration.py`: repeatable golden, metadata and protocol tests
- `tests/fixtures/mos_calibration/avt_target_lambda1_selected_route.json`: self-contained 68 selected-route AVT predictions/labels; correct physical VD geometry
- `docs/LEARNED_METRICS.md`: user-facing automatic calibrated MOS behavior and limitations

**Workflow:** This file documents modifications in the existing, uncommitted branch working tree. No unrelated user changes, branches, commits, merges or remote pushes were made.
