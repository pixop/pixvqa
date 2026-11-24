"""Study-limited experimental H1 seven-ballot ACR overlay, revised 75/15/10.

RESEARCH ONLY. Not selectable by PixVQA's normal MOS algorithm selector.
Neither the default v7 calibration nor the frozen A2-TV123 (AI25) is modified.

The *historical-human 75 / AI 10* stage replays the original 145-clip / 40-AI
source-family balanced A2 amplitude selector, normalized within this stage
(75/85 human and 10/85 AI); the selected amplitude is 0.235 instead of 0.240.
The original human-only sigmoid remains identical.

The *new-human 15* stage fits all individual five-category ACR responses from
the six P.910-screened H1 observers (seven collected), with observer intercepts,
then illustrates a 15% output blend of expected ACR and historical AI10 A2.
The old TV1-3 MOS and new H1 ACR are measured on different scales and TVs; the
second stage is an explicit sensitivity hypothesis, NOT a statistically
identified joint three-cohort calibration or MOS-scale equivalence.

For HDR PQ 2160p this research overlay refuses to extrapolate beyond H1's
observed CVQA support. In other routes it returns the reweighted A2 baseline
(the small AI-amplitude change applies to all routes).
"""

from __future__ import annotations

import math
from typing import Any

from pixvqa_cvqahdr import mos_calibration_a2


RESEARCH_VERSION = "h1_hdr2160_75original_15human7_10ai_20261124_v2"
RESEARCH_STATUS = "experimental_unvalidated"
ORIGINAL_TV123_FRACTION = 0.75
H1_STUDY_OUTPUT_FRACTION = 0.15
AI_FRACTION = 0.10
A2_AI10_REWEIGHTED_AMPLITUDE = 0.235
HISTORICAL_OLD_AI25_AMPLITUDE = 0.240
H1_HISTORICAL_CURVE_ANCHOR = 0.25  # a penalty weight, NOT an evidence fraction
CVQA_H1_MIN = 7.599227864935756
CVQA_H1_MAX = 10.098637140060532
H1_SOURCE_CLIPS = 28
H1_SOURCE_FAMILIES = 14
H1_OBSERVERS_COLLECTED = 7
H1_OBSERVERS_SCREENED = ("B01", "B03", "B04", "B05", "B06", "B07")

# Fitted with the AI10-reweighted A2 baseline.
# Source: H1_HUMAN7_751510_REWEIGHT_20261124/
# H1_7OBS_STUDY_WEIGHT_SENSITIVITY.json, screened6/hinge/0.25/full_fit.
LATENT_THETA = (
    -0.1371520921243924,
    0.6461944945101277,
    -0.4382042255038394,
)
OBSERVER_INTERCEPTS = (
    -0.2620574768226746,
    -0.11581825007946885,
    -0.909397933187157,
    0.7283216511706806,
    0.2989401315025798,
    0.26001187741604,
)
ORDINAL_SCALE = 0.6975259574833641
CATEGORY_THRESHOLDS = (1.5, 2.5, 3.5, 4.5)


def _sigmoid(value: float) -> float:
    v = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-v))


def _validate_score(value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CVQA score must be finite") from exc
    if not math.isfinite(x):
        raise ValueError("CVQA score must be finite")
    return x


def _reweighted_historical_a2(
    score: float, *, transfer: str, encoded_height: int
) -> float:
    """Human-trained original A2 sigmoid + replayed AI10 amplitude 0.235."""
    # Validate via the existing A2 contract without modifying its prediction.
    mos_calibration_a2.predict_mos(
        score, transfer=transfer, encoded_height=encoded_height
    )
    z = (
        mos_calibration_a2.A2_INTERCEPT
        + mos_calibration_a2.A2_CVQA_COEF
        * ((score - mos_calibration_a2.A2_CVQA_CENTER)
           / mos_calibration_a2.A2_CVQA_SCALE)
        + mos_calibration_a2.A2_I2160_COEF
        * (1.0 if encoded_height == 2160 else 0.0)
        + mos_calibration_a2.A2_I_PQ_COEF * (1.0 if transfer == "pq" else 0.0)
    )
    span = mos_calibration_a2.A2_BOOST_END - mos_calibration_a2.A2_BOOST_START
    t = max(0.0, min(1.0, (score - mos_calibration_a2.A2_BOOST_START) / span))
    boost_basis = t * t * (3.0 - 2.0 * t)
    prediction = 1.0 + 4.0 * _sigmoid(z) + A2_AI10_REWEIGHTED_AMPLITUDE * boost_basis
    return max(1.0, min(5.0, prediction))


def _latent_correction(score: float) -> float:
    a, b, c = LATENT_THETA
    return a + b * (score - 9.15) + c * max(0.0, score - 9.25)


def _h1_expected_acr(score: float, historical_baseline: float) -> float:
    mu = historical_baseline + _latent_correction(score)
    return sum(
        1.0 + sum(
            _sigmoid((mu + offset - threshold) / ORDINAL_SCALE)
            for threshold in CATEGORY_THRESHOLDS
        )
        for offset in OBSERVER_INTERCEPTS
    ) / len(OBSERVER_INTERCEPTS)


def predict_mos_h1_research(
    canonical_cvqa_score: float,
    *,
    transfer: str,
    encoded_height: int,
) -> float:
    """Evaluate a bounded, unvalidated 75/15/10 study-aware research scenario.

    Returns the AI10 historical baseline for SDR and HDR1080.
    For HDR2160, requires score inside the observed H1 score interval; outside
    this interval raises rather than drawing an unsupported H1 extrapolation.
    """
    x = _validate_score(canonical_cvqa_score)
    t = str(transfer).strip().lower()
    h = int(encoded_height)
    historical = _reweighted_historical_a2(x, transfer=t, encoded_height=h)

    if t != "pq" or h != 2160:
        return historical
    if not CVQA_H1_MIN <= x <= CVQA_H1_MAX:
        raise ValueError(
            f"H1 study unsupported at CVQA={x:g}; observed human study "
            f"support is [{CVQA_H1_MIN:.6f}, {CVQA_H1_MAX:.6f}]. "
            "CVQA >10 is valid even when the H1 overlay is unsupported."
        )
    new_acr = _h1_expected_acr(x, historical)
    estimate = ((1.0 - H1_STUDY_OUTPUT_FRACTION) * historical
                + H1_STUDY_OUTPUT_FRACTION * new_acr)
    return max(1.0, min(5.0, estimate))


def h1_research_metadata() -> dict[str, Any]:
    return {
        "research_version": RESEARCH_VERSION,
        "research_status": RESEARCH_STATUS,
        "cohort_weights": {
            "original_tv123_human": ORIGINAL_TV123_FRACTION,
            "H1_human_ACR": H1_STUDY_OUTPUT_FRACTION,
            "AI_contact_sheets": AI_FRACTION,
        },
        "AI10_amplitude": A2_AI10_REWEIGHTED_AMPLITUDE,
        "original_A2_AI25_amplitude": HISTORICAL_OLD_AI25_AMPLITUDE,
        "historical_amplitude_loss_ratio": {
            "old_TV123_human": ORIGINAL_TV123_FRACTION / (ORIGINAL_TV123_FRACTION + AI_FRACTION),
            "AI40": AI_FRACTION / (ORIGINAL_TV123_FRACTION + AI_FRACTION),
        },
        "new_H1_output_blend_fraction": H1_STUDY_OUTPUT_FRACTION,
        "ordinal_historical_anchor_lambda": H1_HISTORICAL_CURVE_ANCHOR,
        "calibration_method": "staged two-objective sensitivity scenario, not a joint loss",
        "reference_A2_algorithm": mos_calibration_a2.MOS_CALIBRATION_ALGORITHM,
        "source_clips": H1_SOURCE_CLIPS,
        "source_families": H1_SOURCE_FAMILIES,
        "observers_collected": H1_OBSERVERS_COLLECTED,
        "observers_screened": list(H1_OBSERVERS_SCREENED),
        "H1_ratings_collected": 196,
        "H1_ratings_screened": 168,
        "H1_support": [CVQA_H1_MIN, CVQA_H1_MAX],
        "supported_H1_transfer": "pq",
        "supported_H1_encoded_height": 2160,
        "non_H1_routes": "AI10 historical A2, no new ACR correction",
        "outside_H1_support": "reject HDR2160 rather than extrapolate",
        "warning": (
            "The H1 ACR scale and TV differ from historical TV1-TV3. The "
            "15% transfer is an unvalidated assumption, not independently "
            "identified MOS calibration. The H1 piecewise model failed the "
            "nested model-selection production gate. Not selectable in "
            "PixVQA production or CLI."
        ),
    }
