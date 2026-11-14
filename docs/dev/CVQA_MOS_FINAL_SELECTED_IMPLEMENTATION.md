# CVQA-HDR-NR selected composite MOS policy (PixVQA staging)

Date: 2026-11-14
Branch: `feature/cvqa-hdr-nr-mos-calibration`
Status: **integrated in working tree; shadow-not-prod**

## Policy ID

`selected_s1_sdr_p1_hdr_c0` (`SELECTED_MOS_POLICY` in `mos_calibration.py`)

**Not the default.** Opt-in via `PIXVQA_CVQA_MOS_POLICY=selected_s1_sdr_p1_hdr_c0`.

## Inference formula

Frozen S1 reviewed M1+M7 (identical coefficients to `reviewed_s1_hdr0_sdr1`), then route-specific additive post-output offset, then final `[1, 5]` clamp:

| Route | S1 base | Post-offset | Notes |
|---|---|---|---|
| SDR 1080/2160 | `old392\|U_v2\|target\|1.0` M1+M7 | **+0.4105017796126768** | Full-fit deployed delta (P1) |
| HDR PQ 1080/2160 | `old392\|Original_U\|baseline\|0.0` M1+M7 | **0.0** | C0 — no S2 warp |

U and U-v2 target mappings are **training-only**; not applied at inference.

**v2 (2026-11-14):** Input support clamp **removed** for selected policy only. Historical training support `[4.1344840888613135, 10.051497077192835]` is reported for OOD warnings but does not cap input. Final MOS output still clamped to `[1, 5]`. Legacy and reviewed policies retain bilateral input clamping.

## Provenance locks

| Artifact | SHA-256 |
|---|---|
| Frozen S1 coefficient bundle | `67d39be487fccee6b0a08ef33a4dd7e8adf1745578eff9060ab48e394069a02c` |
| SDR selection lock | `15dbde98a70c3a4ac7e2f7036a084d589e4154f320ebf4dba657f9bb4c894581` |
| HDR selection lock | `5a4749f6fae7d3a7a9c6736910d83c02d41e4d7630e09bba4476ebfed0ee68cd` |
| Composite coefficients digest | `463b641347d68585ea9f99fc6d8c58b3ea274896458f6b1e58cbf58592b43cd3` |
| Embedded runtime digest (offset + locks + input_support_clamp) | `b855164620e66b208826cc68d516fb89297f240ff3c4f019288790d3b1d39916` |

The composite digest **does not** claim the native MOS offset is inside the S1 manifest. `REVIEWED_EMBEDDED_COEFFICIENTS_SHA256` is unchanged.

## Research evidence (retrospective, not blind)

| Cohort | Metric | S1 reviewed | Selected (P1 offset) |
|---|---|---|---|
| SDR HEVC54 | bias | −0.3968 | +0.0172 (source LOO) |
| SDR HEVC54 | MAE | 0.4147 | 0.2810 (source LOO) |
| SDR HEVC54 | RMSE | 0.5101 | 0.3426 (source LOO) |
| VD6 full-fit | MAE | 0.4096 | 0.3192 |
| VD6 full-fit | RMSE | 0.5167 | 0.4135 |
| HDR35 C0 | bias | +0.3429 | (no offset) |
| HDR35 C0 | MAE | 0.5548 | — |
| HDR35 C0 | RMSE | 0.7566 | — |

The +0.4105018 offset is a **full-fit deployed delta**, not a single fixed offset validated on 54 held-out source-LOO rows. AVT ratings previously informed U-v2 routing.

## Usage

```bash
export PIXVQA_CVQA_MOS_POLICY=selected_s1_sdr_p1_hdr_c0
./bin/pixvqa --metrics cvqa-hdr-nr -t distorted.mp4
```

```python
from pixvqa_cvqahdr.mos_calibration import predict_mos, SELECTED_MOS_POLICY
estimate = predict_mos(8.5, transfer="sdr", encoded_height=1080, policy=SELECTED_MOS_POLICY)
```

Run record fields: `research_status=shadow-not-prod`, `post_mos_offset`, `post_output_offset_by_transfer`, selection lock SHA-256s.

## Upstream gate

Same pin gate as reviewed S1 (`gate_reviewed_mos_outcome`). Legacy default not gated.

## Verified checks

- Golden fixture: `tests/fixtures/mos_calibration/avt_selected_s1_sdr_p1_hdr_c0.json` (68 rows)
- 60 SDR = reviewed S1 + offset (output clamped [1,5]); 8 HDR = reviewed S1 exact (within support)
- v2: unclamped input extrapolation; 68 oracle rows unchanged (within support)
- 86+ unit tests including extrapolation, OOD flags, digest pins
- Legacy and reviewed fixtures unchanged
- `scripts/test_cvqahdr_mos_calibration.py`

## Production blockers

| Blocker | Notes |
|---|---|
| Not default | Requires explicit env; no flag flip without subjective study |
| Shadow-not-prod | `research_status` in JSON record |
| Retrospective evidence | AVT-informed routing; source LOO ≠ prospective validation |
| HDR thin n=8 | C0 chosen for headroom, not lowest MAE |
| No deployment | Implementation only; no commits/pushes in this pass |

## Changed paths (v2 unclamped input, 2026-11-14)

- `python/pixvqa_cvqahdr/mos_calibration.py` — `input_support_clamp=False` for selected; digests bumped
- `scripts/test_cvqahdr_mos_calibration.py` — 10 extrapolation/regression tests added
- `tests/fixtures/mos_calibration/avt_selected_s1_sdr_p1_hdr_c0.json` — digest metadata v2
- `docs/LEARNED_METRICS.md`, `docs/dev/CVQA_MOS_REVIEWED_S1_INTEGRATION.md`

Calibration version: `selected-s1-sdr-p1-hdr-c0-unclamped-input-20261114-v2`. Frozen S1 bundle SHA and selection locks unchanged.