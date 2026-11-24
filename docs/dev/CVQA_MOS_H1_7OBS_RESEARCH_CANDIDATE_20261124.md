# H1 seven-observer ordinal ACR overlay — revised 75 / 15 / 10 weights

**Date:** 2026-11-24. **Status:** RESEARCH ONLY; not a validated or production-selectable MOS calibration.

## Intended evidence weights

| Cohort | Weight |
|---|---:|
| Original TV1–TV3 human scores | 75% |
| New H1 human ACR study (seven collected ballots) | 15% |
| Original 40 AI contact-sheet assessments | 10% |

The preceding research candidate used a 30% H1 output blend while retaining the A2 AI25 boost. It did NOT implement the intended new historical/AI split. The revised model corrects both stages.

## Historical-human 75 / AI 10 amplitude re-selection

The original A2 research source run_a2_tv123_nested.py was independently replayed with 145 historical video clips (15 content families), all 435 TV1/2/3 human scores, and the original 40 AI-scored clips. Historical human sigmoid coefficients are unchanged.

The boost amplitude was selected with the original source-balanced MAE grid and the reweighted historical-human/AI-only objective:

    (75/85) * old_human_source_balanced_MAE + (10/85) * AI40_source_balanced_MAE

This reproduces original A2 AI25 amplitude 0.240 and yields revised AI10 amplitude **0.235**, with the exact same human sigmoid coefficients (0.09446591142061583, 0.6046767170474217, 0.6167771933889601, 0.11819874714947097). The human TV1–TV3 scores were not averaged before fitting.

The amplitude change applies to every A2 route, by up to -0.005 MOS in high-CVQA regions, and is zero below CVQA 7.75.

## New H1 15% ordinal human-study component

The seven H1 ballots contain 196 individual 1–5 ACR responses, 28 HDR PQ 2160p clips, 14 content source families, and seven observers. B01–B06 are photograph-verified. B07 was supplied directly in chat, without a scan.

Following the previous sequential P.910 per-video correlation screening, B02 is excluded from the primary fit; B01 and B03–B07 contribute **168 individual ACR observations**, with all seven retained as a sensitivity comparison. Verified individual scores are not deleted.

A five-category ordered-logistic observation model fits individual responses, per-observer severity intercepts, and a quality-dependent hinge correction relative to the revised AI10 A2 baseline:

    latent_f(x) = -0.1371520921243924 + 0.6461944945101277 * (x - 9.15)
                  - 0.4382042255038394 * max(0, x - 9.25)

The fitted ordinal scale is 0.6975259574833641 and category thresholds are 1.5, 2.5, 3.5, and 4.5.

The output-level sensitivity scenario is:

    H1_751510_whatif(x) = 0.85 * A2_human75_AI10(x) + 0.15 * E[H1_ACR | x]

**This is a staged two-objective sensitivity scenario, not a single statistically identified three-cohort calibration.** Historical continuous 0–100 TV scores converted to 1–5 and new discrete categorical ACR scores obtained on a different TV are not on a proven common perceptual scale. A 15% output blend assumes partial transportability. The separate historic-curve shrinkage lambda=0.25 is NOT a cohort importance weight. It is also distinct from the original A2 AI25 amplitude weight.

## Numeric effect at HDR PQ 2160p

| CVQA | Original A2 (AI25) | Reweighted historical AI10 | H1 75/15/10 what-if | Change from A2 |
|---:|---:|---:|---:|---:|
| 8.0 | 3.661643 | 3.661471 | 3.533859 | -0.127784 |
| 8.5 | 3.847170 | 3.845874 | 3.757040 | -0.090130 |
| 9.0 | 4.048835 | 4.045921 | 3.993220 | -0.055616 |
| 9.5 | 4.234381 | 4.230012 | 4.195671 | -0.038710 |
| 10.0 | 4.371806 | 4.366806 | 4.337084 | -0.034722 |

The new H1 data provide empirical CVQA support only within [7.5992278649, 10.0986371401] for HDR PQ 2160p. CVQA scores above 10 are valid. The research API refuses to produce an H1 candidate outside this interval, rather than invent low-end extrapolation or a discontinuous fallback. At CVQA 5–7 the historical-human/AI10 baseline is unchanged because the boost begins at 7.75.

Other transfer/resolution routes have no H1 correction; in the separate research method they only reflect the small 0.240-to-0.235 shared AI amplitude change. Standard PixVQA A2 still uses AI25 amplitude 0.240.

## Observer outliers, robustness and selection caveats

- Source-family-held-out 14-fold fit at fixed hinge / lambda0.25: ordinal NLL 1.377878, individual ACR MAE 0.887396.
- Fixed no-curve-correction alternative: NLL 1.413618, MAE 0.913427.
- Robust epsilon0.10 contamination sensitivity: NLL 1.382685, MAE 0.898937; robust downweighting is not supported.
- Nested 14-outer/13-inner model-class and shrinkage selection: chosen best-inner model held-out NLL 1.455711 vs no correction NLL 1.413618. Conservative one-standard-error selection chooses no correction in all 14 folds.
- Historical A2 saw prior scores on overlapping content sources; even source-family-held-out H1 refits cannot certify independent cross-study generalization.

**Decision:** keep production A2 and default v7 unchanged. The new weights are retained as an independently callable research scenario, not promoted to production.

## Code and reproducibility

Research module: python/pixvqa_cvqahdr/mos_calibration_h1_research.py. Its public research entrypoint remains predict_mos_h1_research. It is not part of the production algorithm selector. Companion regression tests are scripts/test_cvqahdr_mos_calibration_h1_research.py.

Original 7-ballot H1 master remains on G: drive under the H1_28_HDR2160_human/human_ballots directory. All 7 ballots are preserved with provenance, including B07 direct-only.

Reweighted analysis directory on D: drive:

    D:\projects\perceptual_optimization\calibration\production_readiness_20261113\
    AI_EXTENSION_TV2_20261119\analysis\H1_HUMAN7_751510_REWEIGHT_20261124

- A2_HUMAN75_AI10_REPLAY.json — source-balanced original-model amplitude replay and input hashes.
- H1_7OBS_STUDY_WEIGHT_SENSITIVITY.json — individual ordinal fits and source-held-out metrics.
- H1_7OBS_ROBUST_RESULTS.json — robustness and observer sensitivity.
- H1_7OBS_NESTED_SELECTION.json — nested model selection.
- H1_751510_PROVISIONAL_CALIBRATION.json — candidate formula, precise parameters and samples.
- H1_A2_751510_7OBS_EXPERIMENTAL_FOUR_CURVES.csv — old/reweighted/current scenarios by route.
- H1_A2_751510_HDR2160_COMPARISON.png — illustrative curves.
- H1_A2_751510_FOUR_ROUTES.png — all four route effects.

When B08 arrives, archive its scores with provenance, rerun screening and re-fit the ordinal component using the same original75/AI10 historical anchor, and create a new versioned scenario.
