# Canonical CVQA-to-MOS calibration: A2-TV123 + H1 B08 (5%)

**Revision:** 27 November 2026
**Algorithm:** `a2_tv123_ai25_h1_b08_w005`
**Calibration version:** `a2-tv123-ai25-h1-b08-w005-20261127-v1`
**Status:** Sole active PixVQA mapping on this feature branch; provisional research estimate, not production subjectively certified.

This specification supersedes the earlier A2-only, v7+C2 and B07 H1 proposals. The canonical clip-level CompressedVQA-HDR-NR score is unchanged. The estimator maps it to a separate bounded 1–5 quality estimate; it does not replace CVQA, rerun a neural network, or produce measured MOS.

## 1. Subjective evidence and selection

The original A2 calibration was fitted to the individual TV1, TV2 and TV3 scores: 145 clips, 15 source families, 435 human ratings. These are repeated labels on 145 clips, not 435 independent videos. The unified sigmoid was fitted to human evidence. Its smoothstep amplitude (0.240 MOS units) was selected using 75% historical human / 25% AI source-balanced MAE; the AI40 evidence came from 40 contact-sheet-rated clips, not calibrated full-motion HDR viewing. These coefficients and amplitude are frozen. The human-only C1 control had slightly lower pooled historical human MAE; no AI accuracy improvement is claimed.

H1 supplies complementary direct five-category Absolute Category Rating (ACR) evidence from 28 HDR PQ 2160p clips, 14 source families and eight paper ballots (224 ratings). The ordinal categories are 1 Bad, 2 Poor, 3 Fair, 4 Good, 5 Excellent. H1 families overlap the historical study; H1 is not an independent source-family corpus. Historical TV scores were continuous 0–100 targets, converted to 1–5 for calibration; H1 uses discrete 1–5 ratings. Display and protocol differ.

The fixed observer rule iteratively removes the lowest Pearson correlation with the inclusive clip-mean panel score when it falls below 0.75, recalculating the panel mean after removal. B02 was lowest (r=0.620625) and was excluded. B04 initially fell below 0.75 but passed after B02 removal (r=0.760553); B08 passed (r=0.803901). Final fit retains B01, B03, B04, B05, B06, B07, B08 (196 ratings). All eight collected ballots were screened; the all-eight fit is retained only as a sensitivity experiment. The B08 rating sequence was checked against the supplied transcription, not against an original paper image. No raw ballot or clip-level subjective data are included in this repository.

The H1 ordinal model uses a fixed A2 reference, three shared latent correction parameters and centered observer offsets. For rating y in {1,2,3,4,5}, the category probabilities are differences of cumulative logistic probabilities at y−0.5 and y+0.5, with open lower and upper endpoints. Its loss is mean negative log likelihood plus 0.25 times the mean squared latent correction over the historical HDR2160 reference scores, 0.035 times the mean squared centered observer offsets, and 0.015 times the sum of squared slope/hinge coefficients. The sigmoid scale is fitted with log-scale bounds corresponding to [0.12,2.2]. This is a regularized ordinal fit, not a weighted pooling of historical and H1 ratings.

## 2. Frozen mathematical specification

Define S(z) = 1/(1+exp(−z)), C(v)=min(5,max(1,v)), x = canonical clip-level CVQA, I2160 = 1 for 2160p else 0, and IPQ = 1 for PQ else 0. Only encoded heights 1080/2160 and recognized SDR/PQ transfers are runtime routes.

### 2.1 Original A2 baseline (unchanged)

```text
z_r(x) = 0.09446591142061583
       + 0.6046767170474217 * ((x - 8.5) / 2)
       + 0.6167771933889601 * I2160
       + 0.11819874714947097 * IPQ
t(x)   = clamp((x - 7.75) / 2.25, 0, 1)
A_r(x) = C(1 + 4*S(z_r(x)) + 0.240*t(x)^2*(3 - 2*t(x)))
```

The same unified coefficients yield four route-specific curves, not four unrelated fits.

### 2.2 H1 direct-ACR ordinal expectation (HDR PQ 2160p reference)

Let A_H(x)=A_PQ,2160(x). The latent location and expected category rating are:

```text
mu(x) = A_H(x) + theta0 + theta1*(x - 9.15) + theta2*max(0, x - 9.25)
theta = (-0.17481187557998495,
          0.674162834557142,
         -0.40763810029946124)
bias  = (-0.2260022704806734,
         -0.08047014521274108,
         -0.8714780208480978,
          0.7709868980706249,
          0.33747765885675934,
          0.2992962096844712,
         -0.2298103300703432)
sigma = 0.6921736536804622
cuts  = (1.5, 2.5, 3.5, 4.5)
E(x)  = 1 + sum_over_j_and_k(S((mu(x) + bias[j] - cuts[k])/sigma))/7
```

Offsets are ordered by the retained anonymous ballot IDs above. The expected rating follows from the sum of the four category exceedance probabilities, averaged over the seven retained observers. Runtime does not select individual observers.

### 2.3 Anchored, relative, full-range correction

Observed H1 HDR2160 score interval: L=7.599227864935756 to H=10.098637140060532. Use the relative correction normalized by the A2 amount above MOS 1. Its residual is anchored at CVQA=10, so each route preserves its original A2 value there.

```text
D10 = E(10) - A_H(10) = -0.18777110419520504

rho(x) = (E(x) - A_H(x) - D10)/(A_H(x) - 1),   x >= L
rho(x) = -0.3696660530203457
         + 0.24353569558155375 * expm1(x - L),  x < L
rho(10) = 0 exactly

M_r(x) = C(1 + (A_r(x) - 1)*(1 + 0.05*rho(x)))
```

The exponential extension below L uses a one-CVQA-unit scale and matches the fitted correction and numerical derivative at L. Above H the same ordinal expression is extrapolated; there is no taper or second high-side clamp. The H1 correction is shared across all four routes, including outside the observed interval. Only MOS output is bounded; scores above 10 remain valid inputs. The H1 correction is not a fixed additive MOS shift, a 5% mixture of subjective databases, or a 95/5 mixture of A2 and raw H1 MOS. It is a 5% scaling of the anchored relative residual. No v7 M7 correction, SDR offset, C2 low-tail or per-transfer normalization survives.

The sole canonical runtime implementation is [cvqa_mos_b08.py](../../python/pixvqa_cvqahdr/cvqa_mos_b08.py). The full-precision coefficient payload, including model identity and weight, is pinned by SHA-256:

```text
bf90bfac809f9e95ceba1cbf0ca5dc55135007d572d1fe5b888e4d5777741fc2
```

## 3. Why uniform w=0.05

The original A2 methodology was evaluated jointly with the ordinal fit using 15 outer source-family holdouts: within each fold, the held-out family was excluded from both historical and H1 model fits. The A2 baseline coefficients were refitted and the previously selected fold-specific AI25 amplitude was used; the historical reference out-of-fold predictions were reproduced to within 8.94e-9 MOS. The H1 observer screen and model form were fixed globally before these folds. This is retrospective joint source-held-out evaluation, not fully nested hyperparameter selection or independent prospective validation.

Historical MAE below uses original TV1/TV2/TV3 scores individually, on their 0–100 scale (predicted score = 25*(M−1)). The overall row covers the 116 supported-route clips (348 ratings), with the 29 historical 720p clips excluded from the runtime-route aggregate. H1 MAE is on its native 1–5 scale. Each metric first averages within source family, then across families. Do not compare the absolute MAEs of different scales.

| Evidence / route | A2 (w=0) | B08 (w=0.05) | Relative MAE change |
|---|---:|---:|---:|
| Historical HDR PQ 2160p (42 clips) | 7.854655 | 7.829222 | −0.3238% |
| Historical HDR PQ 1080p (14 clips) | 9.229310 | 9.219835 | −0.1027% |
| Historical SDR 2160p (45 clips) | 7.557202 | 7.574175 | +0.2246% |
| Historical SDR 1080p (15 clips) | 9.519424 | 9.648675 | +1.3578% |
| Historical four-route overall (116 clips) | 8.194512 | 8.201957 | +0.0909% |
| H1 individual ACR | 0.999453 | 0.996949 | −0.2506% |
| H1 clip-mean ACR | 0.683504 | 0.680407 | −0.4531% |

The selected 5% is a conservative engineering compromise, not a claim of optimality or statistical superiority. It adds modest direct-ACR influence, with a negligible aggregate historical penalty and slight HDR/H1 MAE reductions. SDR1080 exceeds the approximate +1% per-route guardrail (+1.36%); this exception was explicitly accepted rather than concealed. The all-eight sensitivity at 5% shows the same qualitative trade-off (+0.04% historical overall; −0.34% individual and −0.87% clip-mean H1 MAE, relative to the all-eight baseline). The full sweep is in [aggregate validation summary](cvqa_mos_b08/validation_summary.json). No significance test establishes improvement.

At w=0.05 the maximum absolute difference from frozen A2 over CVQA 0–15 is approximately 0.05921 MOS (PQ2160), 0.04299 (PQ1080), 0.05605 (SDR2160), 0.04005 (SDR1080). The sampled four curves remain bounded, nondecreasing, ordered and exactly anchored at 10. Dense numerical tests additionally cover CVQA 0–20.

## 4. Plots and interactive viewer

- [Four canonical curves (PNG)](cvqa_mos_b08/B08_four_curves.png) · [SVG](cvqa_mos_b08/B08_four_curves.svg)
- [A2/B07/B08 absolute curves (PNG)](cvqa_mos_b08/A2_B07_B08_absolute.png) · [SVG](cvqa_mos_b08/A2_B07_B08_absolute.svg)
- [A2/B07/B08 correction comparison (PNG)](cvqa_mos_b08/A2_B07_B08_comparison.png) · [SVG](cvqa_mos_b08/A2_B07_B08_comparison.svg)
- [Weight trade-off (PNG)](cvqa_mos_b08/B08_weight_tradeoff.png) · [SVG](cvqa_mos_b08/B08_weight_tradeoff.svg)
- [Offline curve explorer (HTML)](cvqa_mos_b08/B08_curve_explorer.html) · [Numerical samples (CSV)](cvqa_mos_b08/B08_curve_samples.csv)

Open the HTML file directly in a modern browser. It embeds CVQA 0–15 samples at 0.01 spacing from the frozen mapper and supports an exploratory weight slider from 0 to 0.30, default 0.05, and a CVQA slider. No server, dependencies or runtime configuration are needed. Moving the slider never modifies PixVQA. The B07 comparison is the superseded seven-collected/six-retained full-range scenario at the same 5% weight (not the older localized 15% proposal). Its historical coefficients exist only in the plot generation script, not in the installed inference modules.

Regenerate the assets using Python with NumPy and Matplotlib installed:

```bash
PYTHONPATH=python python scripts/plot_cvqa_mos_b08.py
```

## 5. Usage and integration

For ordinary PixVQA scoring the mapping is automatic. No MOS algorithm, policy, or weight option is accepted. Previously defined MOS selector environment variables are no longer read; the old CLI flag is rejected as an unrecognized argument. The algorithm ID is retained in output records solely for provenance.

```bash
./bin/pixvqa --metrics cvqa-hdr-nr -t clip.mp4
```

Standalone mapper (only Python standard library; copy the file by itself):

```bash
python python/pixvqa_cvqahdr/cvqa_mos_b08.py 6 10 --transfer pq --height 2160
```

```python
from cvqa_mos_b08 import predict_mos
for transfer, height in [("pq", 2160), ("pq", 1080), ("sdr", 2160), ("sdr", 1080)]:
    print(transfer, height, predict_mos(6.0, transfer=transfer, encoded_height=height))
```

| Route | CVQA 6 | CVQA 10 (A2 anchor) |
|---|---:|---:|
| HDR PQ 2160p | 3.015088829 | 4.371805638 |
| HDR PQ 1080p | 2.428343691 | 3.882582083 |
| SDR 2160p | 2.900262704 | 4.288779438 |
| SDR 1080p | 2.323318629 | 3.774661152 |

PixVQA checks ffprobe geometry/transfer and the pinned upstream commit, checkpoint SHA-256 and profile SHA-256 before emitting the optional MOS output. Unsupported height, unknown/ambiguous transfer, HLG, probe failure, coefficient corruption or unverified upstream suppress MOS without altering canonical CVQA. Protocol/JSON retain `mos_calibration_algorithm`, `mos_calibration_status` and, when available, `mos_calibrated_cvqa`; detailed run JSON contains the coefficient digest, evidence caveat, `ai_loss_fraction=0.25`, `h1_shape_weight=0.05`, and separate H1 support/extrapolation flags. The historical `route_weight_lambda` field denotes the original AI amplitude-loss fraction (0.25), not H1 weight.

## 6. Verification and limitations

Run the standard-library regression suite and the independent temporal/chunking tests (the latter requires PyTorch):

```bash
PYTHONPATH=python python3 scripts/test_cvqahdr_mos_calibration.py
PYTHONPATH=python python3 -O scripts/test_cvqahdr_mos_calibration.py
python tests/test_cvqa_chunking.py
```

The suite independently reconstructs the frozen formula, tests 80,004 route/score combinations, exact anchor, bounds, monotonicity, ordering, continuity, standalone CLI parity, all metadata and probe paths, pinned upstream gating, corruption, removed selectors and optional-stage fault handling. No learned-model checkpoint or GPU is needed for the MOS regression suite.

The H1 measured support is only PQ2160 and the stated score interval. Full-range extension and transfer to the other three routes are engineering assumptions, not new subjective measurements. The original paper ballots, clip-level records, and full refit scripts remain in the separate authorized research project. Prospectively validate on independent source families, viewing conditions and contribution-like content before claiming production-grade subjective quality. Do not use the estimates as per-frame MOS or assume averaging nonlinear MOS-mapped scores equals mapping the canonical pooled score.

Historical documents and fixture files from earlier calibration iterations are retained as evidence, not as active runtime specifications. Consult Git history for their removed implementations.
