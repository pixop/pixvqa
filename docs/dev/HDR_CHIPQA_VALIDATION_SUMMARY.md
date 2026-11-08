# HDR-ChipQA validation summary (development)

**Authoritative primary-pass record** for PixVQA HDR-ChipQA integration validation. User-facing docs stay concise; methodological detail lives here and in the development reports.

## Score semantics (upstream + PixVQA)

HDR-ChipQA is trained to predict **dark-scene MOS** (`dark_mos` in `deps/hdr_chipqa/sureal_dark_mos_and_dmos.csv`, n=310, range ≈**4.73–89.83**). Serving uses linear SVR `predict()` with **no clipping or rescaling**:

| Stage | Path | Lines | Behavior |
|-------|------|-------|----------|
| Training labels | `deps/hdr_chipqa/hdrchipqa_svr.py` | 46–48 | `scores = scores_df['dark_mos']` |
| Train target | same | 91, 112, 124 | `clf.fit(X_train, train_scores)` — MOS units |
| Inference (vendor) | `deps/hdr_chipqa/testing.py` | 17–21 | `score = grid_svr.predict(X_train)` — raw float |
| Inference (PixVQA) | `python/pixvqa_hdr_chipqa/inference.py` | 108–110 | `float(svr.predict(scaler.transform(X))[0])` |
| Logistic map | `deps/hdr_chipqa/hdrchipqa_svr.py` | 34–36 | **Evaluation only** (`results()` for SROCC/LCC/RMSE); **not** applied at serving |

**Conclusion:** Scores are **supervised in MOS label units** but **unbounded** at inference (linear SVR extrapolates). Values above ~90 or below ~5 are outside training support — **not guaranteed calibrated MOS**. Monotonicity under a codec ladder is a **sanity check only**, not MOS validity.

## Primary pristine validation (included)

All ladders: x265 **medium**, 960×540 Lanczos, PQ BT.2020 tags, 60 frames, PSNR/SSIM vs **aligned pristine downscaled reference**, decoded raw YUV scored with bundled HDR-ChipQA (algorithm unchanged).

### PQ BT.2020 — three pristine-source ladders

| # | Content | Campaign dir | Rate modes | Notes |
|---|---------|--------------|------------|-------|
| 1 | **BerlinSeqs** frames 0–59 | `reports/hdr_chipqa_fixed_qp_20261107` | CQP 18–36 | Pristine 4K YUV; QP ladder monotonic |
| 1b | **BerlinSeqs** frames 0–59 | `reports/hdr_chipqa_crf_vs_qp_pristine_20261107` | CRF 18–36 | CRF from same pristine ref; **monotonic** |
| 2 | **BerlinSeqs** frames 1800–1859 | `reports/hdr_chipqa_source_audit_20261107` | CRF 18–36 | Corrected 30 s segment; pristine raw input; **monotonic** |
| 3 | **DinnerScene** frames 0–59 | `reports/hdr_chipqa_pq_final_single_validation_20261107` | CQP + CRF | QP reused from (1) campaign; CRF fresh; QP **monotonic**, CRF minor CRF30>CRF24 step |

Berlin (0–59): pristine CRF and CQP both monotonic on HDR-ChipQA. DinnerScene: CQP monotonic; pristine CRF has small adjacent reversal at 24→30.

### SDR BT.709

| Content | Campaign | Result |
|---------|----------|--------|
| CityAlley | `reports/hdr_chipqa_monotonicity_20261107` (SDR rows) + prior `reports/chipqa-validation` | CRF ladder **monotonic** on 60/180/300 f |

### HLG (experimental)

| Content | Campaign | Result |
|---------|----------|--------|
| OLED_Wonders | `reports/hdr_chipqa_monotonicity_20261107` (HLG rows) | CRF ladder monotonic; **not paper-validated** — same BT2020 feature path, no EOTF |

The source-provenance investigation is archived separately in `reports/hdr_chipqa_source_audit_20261107/REPORT.md`. It is not part of the primary validation dataset or the pass/fail accounting.

## Integration checks (static)

```bash
export PIXVQA_HDR_CHIPQA_FEATURE_PYTHON=/home/sergio/.venvs/pixvqa-chipqa/bin/python
export PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON=/home/sergio/.venvs/pixvqa-chipqa-ref/bin/python
$PIXVQA_HDR_CHIPQA_FEATURE_PYTHON scripts/test_chipqa_reference.py
python3 scripts/test_learned_metrics_cli.py
```

See also [`CHIPQA_REFERENCE.md`](CHIPQA_REFERENCE.md) for environments, goldens, and CLI examples.

## Production posture

HDR-ChipQA in PixVQA is an **optional diagnostic NR metric**. Pristine-source CQP ladders are the most reliable monotonic proxy tested; CRF ordering is **not universally reliable** even on pristine input (DinnerScene 60 f). Do **not** treat scores as bounded 0–100 MOS or as a production BD-rate axis without external subjective calibration.
