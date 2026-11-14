# CVQA-to-MOS — single selected formula (runtime)

**Status:** Automatic default since 2026-11-14 simplification. Policy ID fixed to `selected_s1_sdr_p1_hdr_c0`.

## Formula

```
z = (CVQA − 8.382293636852387) / 1.1647912353499874
MOS = clip(1 + 4·σ(a_T + b_T·z) + d_{T,h}, 1, 5)
```

Effective route parameters are embedded in `python/pixvqa_cvqahdr/mos_calibration.py`. Source JSON:

`perceptual_optimization/.../CVQA_TO_MOS_SCIENTIFIC_REVISION_20261114/data/PIXVQA_CVQA_to_MOS_Selected_Coefficients_v5.json`

SHA-256: `ba60d7c0d6154c7d66f16e2e453b69a936e7b96e3053c3b54747377537557718`

## Runtime behavior

- Applied automatically on every successful CVQA-HDR-NR run when ffprobe identifies SDR or HDR PQ at 1080p/2160p.
- No `PIXVQA_CVQA_MOS_POLICY` required. Legacy policy env values **fail closed** (`obsolete_policy_env:…`).
- CVQA input is **not** clamped; final MOS clipped to [1, 5].
- Upstream pin gate (commit, checkpoint SHA-256, profile SHA-256) required before MOS is published.

## Tests

```bash
PYTHONPATH=python python3 scripts/test_cvqahdr_mos_calibration.py
```

Golden fixture: `tests/fixtures/mos_calibration/avt_selected_s1_sdr_p1_hdr_c0.json` (68 rows).

Independent deliverable parity: `verification.py` in the scientific revision folder (max runtime difference ≤ 8.88×10⁻¹⁶).