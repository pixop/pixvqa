# CVQA-to-estimated-MOS calibration: transfer-separated models (16 November 2026)

> **Historical record:** Superseded for runtime on 27 November 2026 by [canonical B08](CVQA_MOS_A2_TV123_INTEGRATION.md). Earlier commands, selectors and module paths below describe the archived implementation, not current PixVQA.

> **Feature-branch extension (19 November 2026):** The original fitted v7 coefficients and historical evidence described here remain frozen. A selected **−0.080 MOS convex C² SDR low-anchor correction**, together with the previously selected HDR PQ C² lower tails, is implemented **only** on feature branch [feature/cvqa-mos-v7-local-tail-006](https://github.com/pixop/pixvqa/tree/feature/cvqa-mos-v7-local-tail-006). For the newly effective below-support mapping, comparison figures and limitations, see [v7 localized C² tail](CVQA_MOS_V7_LOCAL_TAIL_CONVEX.md). Nothing was merged to main.

## Purpose and change

This change updates the **selected automatic CVQA-to-estimated-MOS calibration** in PixVQA to the new transfer-separated research fits. The canonical CompressedVQA clip score remains untouched. The calibration still uses the same sigmoid (M1), resolution/transfer correction (M7), final MOS clipping to [1,5], metadata gating and upstream checkpoint/profile pinning.

The previous selected SDR and PQ curves were fitted separately but each used the same **mixed 392-observation development cohort**. The new models are fitted with exclusively own-transfer rows:

| Route | Development sample | Subjective target | Extra HQ weighting | M7 ridge | Native-MOS post-offset |
|---|---|---|---|---:|---:|
| SDR | 200 SDR only, 30 recorded source groups | Revised U-v2 | Baseline, λ=0 | 0.1 | +0.5107732905216783 |
| HDR PQ | 192 HDR PQ only, 35 recorded source groups | Original U | Baseline, λ=0 | 0.1 | 0 |

The independent research source is the JSON file:

`perceptual_optimization/calibration/production_readiness_20261113/CVQA_TO_MOS_TRANSFER_SEPARATED_DELIVERABLE_20261116/data/PIXVQA_CVQA_to_MOS_Selected_Coefficients_v7_Research.json`

Its SHA-256 is `e817ed479079c0e208e8ec0d84aad3b5836606eeb329585d27e2c461c24b8e73`. A byte-identical copy is retained as `tests/fixtures/mos_calibration/cvqa_mos_v7_source_coefficients.json`, so independent regression tests do not need the separate research repository at run time.

## Model equation and values

For transfer `T` and encoded video height `h`:

`z_T = (canonical_CVQA - mean_T) / std_T`

`estimated_MOS = clip(1 + 4 * sigmoid(a_T + b_T * z_T) + d_{T,h}, 1, 5)`

This concise form is algebraically equivalent to the original two-stage M1/M7 clipping, with the selected positive SDR post-offset applied after the metadata stage. Input CVQA is **never score-support-clamped**; only output MOS is clipped.

| Transfer | Score mean | Score standard deviation | Score support |
|---|---:|---:|---|
| SDR | 8.010332079849347 | 1.241145006178454 | [4.1344840888613135, 10.038337481465987] |
| PQ | 8.769753592063894 | 0.9333470039280067 | [6.268450107554486, 10.051497077192835] |

| Route | Effective sigmoid offset `a_T` | Effective slope `b_T` | Correction `d_{T,h}` |
|---|---:|---:|---:|
| SDR 1080 | −0.49117628849094824 | 1.0977838841771388 | +0.5107732905216783 |
| SDR 2160 | −0.49117628849094824 | 1.0977838841771388 | +0.5962084585844877 |
| PQ 1080 | +0.07227163628268213 | 0.8293304839931255 | −0.131036744556774 |
| PQ 2160 | +0.07227163628268213 | 0.8293304839931255 | +0.0978636394362893 |

At canonical CompressedVQA score 10, predicted ratings are **SDR 4.6328337676073605/4.71826893567017** and **PQ 3.9182233198793637/4.147123703872427** (1080p/2160p).

## Runtime policy and backward traceability

The identifier `selected_s1_sdr_p1_hdr_c0` remains unchanged for protocol compatibility. It describes the same S1 architecture with SDR post-offset and no HDR offset; it **does not imply the old coefficients**. The calibration version now reads `selected-s1-sdr200-pq192-transfer-separated-20261116-v3`, with regenerated coefficient fingerprints and transfer-specific normalization and score-support metadata.

The older fixture `tests/fixtures/mos_calibration/avt_selected_s1_sdr_p1_hdr_c0.json` is **preserved** with its historical predictions. The new selected 68-case fixture is `avt_transfer_separated_v7_selected.json`, computed independently from frozen v7 four-tuple coefficients rather than from the PixVQA runtime. The additional 89 selected native-rating per-clip rows and six VD clips are separately saved as CSV fixtures.

The change **does not** introduce new transfer-function detection, change HLG rejection, modify the NR score or relax upstream checkpoint/profile compatibility. When metadata is supported, the selected calibration is applied automatically as it was before. It is **not** a new opt-in feature.

## Validation, limitations and deployment

Retrospective native-MOS comparisons after correction:

| AVT cohort | Selected new model | MAE | RMSE |
|---|---|---:|---:|
| SDR HEVC54 (five source families, family-LOO native correction) | SDR U-v2 | 0.2780 | 0.3360 |
| SDR VD6 (six clips, fixed SDR54 native correction) | SDR U-v2 | 0.3079 | 0.3959 |
| HDR HEVC35 (five source families, no native correction) | PQ original U | 0.6042 | 0.8069 |

The HDR model **regresses** against the prior mixed-data model on HDR35 (previous MAE 0.5552). There are no native HDR scores >=4.5 to validate the high-quality tail. This code update implements the user's selected transfer-separated models; it does **not** establish their prospective subjective validity or authorize a production release.

Run regression checks from the PixVQA repository:

```bash
PYTHONPATH=python .venv/bin/python scripts/test_cvqahdr_mos_calibration.py
PYTHONPATH=python .venv/bin/python -m unittest discover -s scripts -p 'test_cvqahdr*' -v
```

The MOS test checks all four supported routes, numerical parity with the independently frozen JSON, 68 golden cases, 89 native per-clip cases, six VD examples, metadata and route gating, output monotonicity and clipping, per-transfer out-of-support flags, provenance hash checks, fail-closed behavior and unchanged canonical score.

The original calibration v5 and original scientific report v6 remain unchanged in the research repository. The original software integration was committed on the CVQA feature branch, without merging to main or deploying a production release.

**Independent software review (17 November 2026):** [CVQA-to-MOS v7 audit and CLI regression resolution](CVQA_MOS_INDEPENDENT_SOFTWARE_AUDIT_20261117.md). The review corrected stale CLI assertions and hardened the direct prediction integrity checks and public provenance metadata. The original coefficients and calibration outputs remain unchanged.
