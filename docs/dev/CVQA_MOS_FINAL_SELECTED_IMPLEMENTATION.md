# CVQA-to-MOS — transfer-separated selected calibration (current runtime)

**Updated:** 16 November 2026
**Status:** Automatically applied when supported video metadata is available. Provisional estimated quality, **not** a production-certified subjective MOS predictor.

The current selected calibration uses the original M1 monotone sigmoid and M7 metadata correction, but **separate SDR and HDR PQ training datasets**. Earlier fitted curves trained both routes on a mixed 392-observation dataset; that model is preserved in the historical v6 scientific report but is no longer the active coefficient set.

The historical protocol policy ID `selected_s1_sdr_p1_hdr_c0` is retained for compatibility because the architecture remains S1 + SDR post-offset + HDR without post-offset. **Do not identify a coefficient set from the policy ID alone.** The runtime `calibration_version`, source SHA-256 and embedded-coefficient digest distinguish the 2026-11-16 transfer-separated calibration from the older 2026-11-14 fit.

> **Feature branch update (19 November 2026):** The automatic mapping on [feature/cvqa-mos-v7-local-tail-006](https://github.com/pixop/pixvqa/tree/feature/cvqa-mos-v7-local-tail-006) uses the frozen sigmoid defined below **only at and above the score joins** (SDR 5.65, HDR PQ 6.268450). Below those joins, it uses the separately versioned monotone, convex C² SDR low-tail adapter with **MOS(0)=1**, and the two SDR CVQA 4.134 anchors are lowered by **0.080 MOS**. See the [complete localized v7 specification and figures](CVQA_MOS_V7_LOCAL_TAIL_CONVEX.md). This is not merged to main or production-certified.

## Frozen v7 fitted formula and local branch extension

```text
z_T = (canonical_CVQA - mean_T) / std_T
estimated_MOS = clip(1 + 4*sigmoid(a_T + b_T*z_T) + d_(T,height), 1, 5)
```

| Transfer | Training rows | Score mean | Score standard deviation | Subjective bridge | Ridge | Post-offset |
|---|---:|---:|---:|---|---:|---:|
| SDR | 200 SDR only | 8.010332079849347 | 1.241145006178454 | Revised U-v2 | 0.1 | +0.5107732905216783 |
| HDR PQ | 192 PQ only | 8.769753592063894 | 0.9333470039280067 | Original U | 0.1 | 0 |

Both models use baseline source/study-balanced weights without additional high-quality weighting (λ=0).

| Route | Sigmoid intercept | Sigmoid slope | Combined correction |
|---|---:|---:|---:|
| SDR 1080p | −0.49117628849094824 | 1.0977838841771388 | +0.5107732905216783 |
| SDR 2160p | −0.49117628849094824 | 1.0977838841771388 | +0.5962084585844877 |
| HDR PQ 1080p | +0.07227163628268213 | 0.8293304839931255 | −0.131036744556774 |
| HDR PQ 2160p | +0.07227163628268213 | 0.8293304839931255 | +0.0978636394362893 |

The input CompressedVQA score is never clamped. Out-of-training-support flags are calculated **per transfer**: SDR [4.1344840888613135, 10.038337481465987] and PQ [6.268450107554486, 10.051497077192835]. Output MOS is clipped to [1,5].

## Scientific source and integrity

- Scientific report: `perceptual_optimization/calibration/production_readiness_20261113/CVQA_TO_MOS_TRANSFER_SEPARATED_DELIVERABLE_20261116/PIXVQA_CVQA_to_MOS_Scientific_Technical_Report_v7.md`
- Full-precision research coefficient source: `tests/fixtures/mos_calibration/cvqa_mos_v7_source_coefficients.json`
- Source SHA-256: `e817ed479079c0e208e8ec0d84aad3b5836606eeb329585d27e2c461c24b8e73`
- Golden fixture: `tests/fixtures/mos_calibration/avt_transfer_separated_v7_selected.json` (68 historical AVT clips, independent four-tuple oracle).
- The older `avt_selected_s1_sdr_p1_hdr_c0.json` fixture is intentionally preserved unchanged.

For detailed development evidence and known limitations, see [Transfer-Separated v7 Integration](CVQA_MOS_TRANSFER_SEPARATED_V7.md).

## Runtime behavior

The calibration is applied automatically after canonical CompressedVQA scoring only when ffprobe identifies supported SDR/PQ metadata at 1080p/2160p. HLG, ambiguous transfer metadata, unsupported heights, invalid scores or mismatched pinned upstream assets suppress MOS but never change canonical CompressedVQA.

The environment selector `PIXVQA_CVQA_MOS_POLICY` is obsolete: unset or the selected policy ID is accepted; other non-empty values fail closed.

## Verification

```bash
PYTHONPATH=python .venv/bin/python -B scripts/test_cvqahdr_mos_calibration.py
PYTHONPATH=python .venv/bin/python -O -B scripts/test_cvqahdr_mos_calibration.py
```

These tests compare runtime values with an independent scientific reference, including the four CompressedVQA=10 routes, 68 native-rated AVT fixture cases, extra per-clip SDR54/HDR35 and VD6 comparisons, score support, monotonicity, digest integrity, and fail-closed metadata routing.

**Limitations:** HDR35 MAE 0.6042 is worse than the former mixed-training HDR curve's 0.5552. No high-quality native HDR observations with MOS≥4.5 were available. This update integrates the research-selected curves; prospective subjective validation and production release qualification remain separate.
