"""Opt-in, study-limited H1 seven-ballot CVQA-to-ACR research overlay.

NOT an algorithm selection, production calibration, or validated cross-TV MOS.
The currently selected A2 (and default v7) are untouched.

For HDR PQ 2160p, evaluation is deliberately restricted to the CVQA support
covered by the 28 new H1 clips. Below or above this interval the provisional
mapping refuses to extrapolate, including the otherwise valid CVQA > 10 range.

The 30% *output* blend is an explicit transport hypothesis from ordinal ACR to
historic MOS scale; it is not the original 25%-AI amplitude weight, nor a
validated equivalence between the studies or their televisions.
"""

from __future__ import annotations

import math
from typing import Any

from pixvqa_cvqahdr import mos_calibration_a2


RESEARCH_VERSION = "h1_hdr2160_a2_seven_observers_20261124_v1"
RESEARCH_STATUS = "experimental_unvalidated"
CVQA_H1_MIN = 7.599227864935756
CVQA_H1_MAX = 10.098637140060532
H1_STUDY_OUTPUT_FRACTION = 0.30
H1_HISTORICAL_CURVE_ANCHOR = 0.25
H1_SOURCE_CLIPS = 28
H1_SOURCE_FAMILIES = 14
H1_OBSERVERS_COLLECTED = 7
H1_OBSERVERS_SCREENED = ("B01", "B03", "B04", "B05", "B06", "B07")

# Source: H1_7OBS_STUDY_WEIGHT_SENSITIVITY.json, case screened6,
# hinge model, historical-curve regularization 0.25. Each ACR response was
# included individually in a 5-category ordered-logistic likelihood.
LATENT_THETA = (-0.1387460215555705, 0.6449941229765033, -0.43841398782780305)
OBSERVER_INTERCEPTS = (
    -0.2622676545260202,
    -0.11600537071745345,
    -0.909486807431632,
    0.7286214904189257,
    0.2991035559648164,
    0.26003478629136356,
)
ORDINAL_SCALE = 0.6977632247313065
CATEGORY_THRESHOLDS = (1.5, 2.5, 3.5, 4.5)


def _sigmoid(z: float) -> float:
    if z < -40.0:
        return 0.0
    if z > 40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _validate_score(value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CVQA score must be a finite number") from exc
    if not math.isfinite(x):
        raise ValueError("CVQA score must be a finite number")
    return x


def _latent_correction(x: float) -> float:
    a, b, c = LATENT_THETA
    return a + b * (x - 9.15) + c * max(0.0, x - 9.25)


def _h1_expected_acr(x: float, original_a2: float) -> float:
    # Expectation of ordinal categories, averaged across screened observer
    # profiles. NOT an arithmetic average of the input ballot scores.
    mu = original_a2 + _latent_correction(x)
    return sum(
        1.0 + sum(
            _sigmoid((mu + bias - cut) / ORDINAL_SCALE)
            for cut in CATEGORY_THRESHOLDS
        )
        for bias in OBSERVER_INTERCEPTS
    ) / len(OBSERVER_INTERCEPTS)


def predict_mos_h1_research(
    canonical_cvqa_score: float,
    *,
    transfer: str,
    encoded_height: int,
) -> float:
    """Evaluate the seven-observer H1 research candidate.

    Non-PQ-2160 routes equal unmodified A2 exactly. For PQ-2160 this function
    raises ValueError outside the H1 observed score range. This is intentional:
    a global fallback could create a discontinuity; extrapolation from new
    six-observer ACR is unsupported by these 28 clips.
    """
    x = _validate_score(canonical_cvqa_score)
    transfer_value = str(transfer).lower()
    height = int(encoded_height)
    prior = mos_calibration_a2.predict_mos(
        x, transfer=transfer_value, encoded_height=height
    )
    if transfer_value != "pq" or height != 2160:
        return prior
    if not CVQA_H1_MIN <= x <= CVQA_H1_MAX:
        raise ValueError(
            f"H1 seven-observer candidate unsupported at CVQA={x:g}; "
            f"new human evidence supports only {CVQA_H1_MIN:.6f}"
            f"–{CVQA_H1_MAX:.6f}. CVQA scores above 10 remain valid."
        )
    expected_acr = _h1_expected_acr(x, prior)
    candidate = ((1.0 - H1_STUDY_OUTPUT_FRACTION) * prior
                 + H1_STUDY_OUTPUT_FRACTION * expected_acr)
    return max(1.0, min(5.0, candidate))


def h1_research_metadata() -> dict[str, Any]:
    return {
        "research_version": RESEARCH_VERSION,
        "status": RESEARCH_STATUS,
        "reference_calibrator": mos_calibration_a2.MOS_CALIBRATION_ALGORITHM,
        "source_clips": H1_SOURCE_CLIPS,
        "source_families": H1_SOURCE_FAMILIES,
        "observers_collected": H1_OBSERVERS_COLLECTED,
        "observers_screened": list(H1_OBSERVERS_SCREENED),
        "human_ACR_ratings_screened": 168,
        "human_ACR_ratings_all": 196,
        "supported_transfer": "pq",
        "supported_encoded_height": 2160,
        "human_evidence_cvqa_range": [CVQA_H1_MIN, CVQA_H1_MAX],
        "output_study_weight": H1_STUDY_OUTPUT_FRACTION,
        "historical_anchor_lambda": H1_HISTORICAL_CURVE_ANCHOR,
        "formula": "0.70*A2(x)+0.30*E[ordered-ACR( A2(x)+f(x)+observer_bias )]",
        "outside_H1_support": "error for HDR2160 (no unsupported extrapolation)",
        "other_routes": "original A2, unchanged",
        "warning": (
            "Experimental within-study ordinal response. Different TV and rating "
            "protocol from historical TV1-TV3; no certified MOS equivalence. "
            "Nested source-family model selection did not confirm improvement. "
            "Not selectable via production calibration selector."
        ),
    }
