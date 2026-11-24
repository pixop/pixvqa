# H1 seven-observer HDR 2160p ordinal ACR overlay — research candidate

**Date:** 2026-11-24
**Status:** Experimental study-specific candidate. NOT a selected or production PixVQA calibration.

## What was added

Executable module: python/pixvqa_cvqahdr/mos_calibration_h1_research.py

It provides predict_mos_h1_research(score, transfer="pq", encoded_height=2160). The frozen PixVQA A2 coefficients, selected v7 default, CLI algorithm selector, and every existing production MOS output remain unchanged.

### Human evidence and screening

- Seven H1 observers provided 196 individual ordinal ACR 1–5 responses on 28 HDR PQ 2160p clips (14 source families × 2 encodings).
- B01–B06: 168 user-verified scores with six original JPEG photographs and six PNG scans.
- B07: 28 directly supplied scores, with source-text preservation; no photo or scan exists.
- Sequential per-PVS P.910(2023) correlation screening excludes B02. The primary fit includes B01, B03–B07: 168 ordinal individual ratings.
- Seven-observer scores, robust outlier weighting and two individual-score training-only exclusions were evaluated as sensitivity controls. No original ratings were deleted.

### Experimental model and weighting

Five-category ordinal logistic likelihood applied to every retained human score, with centered observer severity intercepts and historical-curve shrinkage lambda=0.25.

For HDR2160, the new latent response is the fully fitted A2 value plus:

f(x) = -0.1387460215555705 + 0.6449941229765033*(x-9.15) - 0.43841398782780305*max(0,x-9.25).

Expected ACR is computed from category probabilities averaged over the six screened observer intercepts, not by fitting a curve to per-video mean MOS.

Experimental what-if output: 0.70*A2(x) + 0.30*H1_EACR(x).

**The 30% fraction is an assumed cross-study scale-transport hypothesis, not measured MOS equivalence.** Historical continuous 0–100 MOS mapped to 1–5, new categorical ACR and AI contact-sheet scores are not interchangeable observations. Original A2 25% AI weight controlled amplitude selection; lambda=0.25 here is a separate prior penalty, and neither equals this hypothetical 30% H1 output blend.

## Strict domain handling

- Candidate supports only PQ HDR, height 2160, and CVQA in [7.599227864935756,10.098637140060532].
- Other A2 routes are exactly unchanged.
- HDR2160 CVQA scores 5, 6, 7, or outside the H1 observed range cause an exception rather than inventing an extrapolation or an abrupt fall-back discontinuity.
- CVQA scores above 10 are valid; 10.05 is supported by this candidate, 10.2 simply lies outside the empirical H1 range.
- This module is NOT registered in the production algorithm selector, ensuring it can be exercised only by explicit Python import.

## Numerical output

| CVQA | Current A2 HDR2160 | Seven-observer H1 candidate | Candidate minus A2 |
|---:|---:|---:|---:|
| 8.0 | 3.661643 | 3.406362 | -0.255280 |
| 8.5 | 3.847170 | 3.669229 | -0.177940 |
| 9.0 | 4.048835 | 3.942895 | -0.105940 |
| 9.5 | 4.234381 | 4.164850 | -0.069531 |
| 10.0 | 4.371806 | 4.311260 | -0.060546 |

## Scientific validation status

Fixed-form, 14-source-family LOFO:
- Observer-adjusted ordinal model with no curve correction: NLL 1.413720, MAE 0.913201.
- With the exploratory anchored hinge correction: NLL 1.377886, MAE 0.887275.
- Robust 10%-contamination variant: NLL 1.382650, MAE 0.898785 (worse than standard).
- Nested 14-outer/13-inner source-family model/weight selection: best-inner-selected NLL 1.455697 versus no-correction 1.413720. Conservative one-SE rule selects **no correction in every outer fold**.

Thus the H1 candidate fails the production-change gate. A2 already used historical ratings of overlapping content, and neither the TV nor rating-scale mapping is common across studies. The H1 results alone do not justify changes to SDR and 1080p, or extrapolation to CVQA below 7.599.

## Reproducibility and next ballot

Archive master on G: drive:
G:\Videos\Perceptual_Optimization\LIVE_HDR_vs_SDR\extracted\public_release\subjective_test_packages_20261119\H1_28_HDR2160_human\human_ballots\H1_28clip_ACR_MASTER.xlsx

Seven-observer research:
D:\projects\perceptual_optimization\calibration\production_readiness_20261113\AI_EXTENSION_TV2_20261119\analysis\H1_HUMAN7_TV123_RECALIBRATION_20261124

Key files:
- H1_7OBS_PROVISIONAL_CURVE_SPEC.json
- H1_7OBS_EXPERIMENTAL_A2_CURVES_20261124.csv
- H1_7OBS_UPDATED_A2_HDR2160_EXPERIMENTAL.png
- H1_7OBS_STUDY_WEIGHT_SENSITIVITY.json
- H1_7OBS_ROBUST_RESULTS.json
- H1_7OBS_NESTED_SELECTION.json

When B08 arrives, preserve the new observer scores with provenance, rerun source-aware screening / individual ordinal calibration and robustness, and version the next curve separately.
