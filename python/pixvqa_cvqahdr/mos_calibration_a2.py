"""A2-TV123 experimental CVQA-to-estimated-MOS calibration (25% AI amplitude loss).

Jointly calibrated to individual TV1/TV2/TV3 subjective scores (435 ratings) with a
shared smoothstep amplitude tuned using 75% human / 25% AI source-balanced MAE.
Mutually exclusive with frozen v7+M7+C2: no SDR native offset, no per-transfer
normalization, no low-tail adapter, no AI inference at runtime.

Scientific selection: perceptual_optimization/.../SELECTED_A2_TV123_IMPLEMENTATION_DECISION_20261120.md
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

MOS_CALIBRATION_ALGORITHM = "a2_tv123_ai25"
CALIBRATION_VERSION = "a2-tv123-ai25-individual-ratings-20261120-v2"
CALIBRATION_LABEL = (
    "Experimental A2-TV123 unified sigmoid + smoothstep amplitude (25% AI tuning)"
)
RESEARCH_STATUS = "experimental_integration_candidate"
OUTPUT_SCALE_LABEL = "estimated perceptual quality 1-5"
WEIGHT_KIND = "tv123_individual_human_sigmoid_ai25_amplitude"
COEFFICIENT_COHORT = "TV123_145clips_435_human_40_ai"
COEFFICIENT_SOURCE = (
    "perceptual_optimization/calibration/production_readiness_20261113/"
    "AI_EXTENSION_TV2_20261119/analysis/AI_INFORMED_POSTCALIBRATION_20261120/"
    "A2_TV123_INDIVIDUAL_RATINGS_20261120/"
    "SELECTED_A2_TV123_IMPLEMENTATION_DECISION_20261120.md"
)
EXPERIMENT_SUMMARY = (
    "perceptual_optimization/calibration/production_readiness_20261113/"
    "AI_EXTENSION_TV2_20261119/analysis/AI_INFORMED_POSTCALIBRATION_20261120/"
    "A2_TV123_INDIVIDUAL_RATINGS_20261120/results/experiment_summary.json"
)
MANIFEST_ID = "pixvqa_calib_a2_tv123_ai25_20261120_v2"

A2_INTERCEPT = 0.09446591142061583
A2_CVQA_COEF = 0.6046767170474217
A2_I2160_COEF = 0.6167771933889601
A2_I_PQ_COEF = 0.11819874714947097
A2_CVQA_CENTER = 8.5
A2_CVQA_SCALE = 2.0
A2_BOOST_START = 7.75
A2_BOOST_END = 10.0
A2_AMPLITUDE = 0.24
A2_AI_LOSS_FRACTION = 0.25

EVIDENCE_CAVEAT = (
    "Experimental A2-TV123: 145 clips, 15 source families, 435 correlated TV-score "
    "observations (TV1+TV2+TV3 retained individually). Amplitude selected with 75% "
    "human / 25% AI source-balanced MAE; AI scores from contact sheets, not full-motion "
    "HDR viewing. Human-only C1 had slightly lower pooled MAE; do not claim AI improved "
    "subjective accuracy. Research LOW cohort includes 720p; PixVQA runtime accepts only "
    "1080p and 2160p. Not subjectively certified for production."
)

# CVQA scores above 10 are valid; no artificial upper limit applies to A2.
_SUPPORT = {"score_min": 0.0}
_SUPPORTED_HEIGHTS = frozenset({1080, 2160})

_CALIBRATORS: dict[str, dict[str, Any]] = {
    "A2_TV123_Unified": {
        "route": "unified",
        "research_arm": "New_TV123_A2",
        "weight_kind": WEIGHT_KIND,
        "weight_lambda": A2_AI_LOSS_FRACTION,
        "model_key": "TV123|A2|human_sigmoid|ai25_amplitude",
        "support": dict(_SUPPORT),
        "sigmoid_coefficients": {
            "intercept": A2_INTERCEPT,
            "cvqa_coef": A2_CVQA_COEF,
            "i_2160_coef": A2_I2160_COEF,
            "i_pq_coef": A2_I_PQ_COEF,
            "cvqa_center": A2_CVQA_CENTER,
            "cvqa_scale": A2_CVQA_SCALE,
        },
        "amplitude": {
            "start_cvqa": A2_BOOST_START,
            "end_cvqa": A2_BOOST_END,
            "amplitude_mos": A2_AMPLITUDE,
        },
    },
}

_POLICY_SPEC: dict[str, Any] = {
    "algorithm_id": MOS_CALIBRATION_ALGORITHM,
    "input_support_clamp": False,
    "input_support_policy": "unbounded_extrapolation",
    "calibration_version": CALIBRATION_VERSION,
    "calibration_label": CALIBRATION_LABEL,
    "weight_kind": WEIGHT_KIND,
    "coefficient_cohort": COEFFICIENT_COHORT,
    "coefficient_source": COEFFICIENT_SOURCE,
    "experiment_summary": EXPERIMENT_SUMMARY,
    "manifest_id": MANIFEST_ID,
    "evidence_caveat": EVIDENCE_CAVEAT,
    "calibrators": _CALIBRATORS,
    "formula": "unified_sigmoid_plus_smoothstep_amplitude",
    "v7_legacy_components": False,
    "post_output_offset_by_transfer": {"pq": 0.0, "sdr": 0.0},
    "ai_loss_fraction": A2_AI_LOSS_FRACTION,
}


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _embedded_coefficients_payload() -> dict[str, Any]:
    return {
        "algorithm_id": MOS_CALIBRATION_ALGORITHM,
        "sigmoid_coefficients": [
            A2_INTERCEPT,
            A2_CVQA_COEF,
            A2_I2160_COEF,
            A2_I_PQ_COEF,
        ],
        "boost_start": A2_BOOST_START,
        "boost_end": A2_BOOST_END,
        "amplitude": A2_AMPLITUDE,
        "ai_loss_fraction": A2_AI_LOSS_FRACTION,
        "formula": "unified_sigmoid_plus_smoothstep_amplitude",
        "post_output_offset": 0.0,
        "low_tail": None,
        "v7_legacy_components": False,
    }


def _coefficients_digest_payload() -> dict[str, Any]:
    return {
        "algorithm_id": MOS_CALIBRATION_ALGORITHM,
        "decision_artifact": "SELECTED_A2_TV123_IMPLEMENTATION_DECISION_20261120.md",
        "experiment_summary": "A2_TV123_INDIVIDUAL_RATINGS_20261120/results/experiment_summary.json",
        "sigmoid_coefficients": [
            A2_INTERCEPT,
            A2_CVQA_COEF,
            A2_I2160_COEF,
            A2_I_PQ_COEF,
        ],
        "boost_start": A2_BOOST_START,
        "boost_end": A2_BOOST_END,
        "amplitude": A2_AMPLITUDE,
        "ai_loss_fraction": A2_AI_LOSS_FRACTION,
        "support": dict(_SUPPORT),
    }


EMBEDDED_COEFFICIENTS_SHA256 = "5b432596109f24ae4a1050ea25f0be4b6aafeab3366d70ef6f5fd45641639e72"
SELECTED_COEFFICIENTS_DIGEST_SHA256 = "05497a15ce8f5fee5dc378c4a1961418f8b55e7e4f734b90227d432894673f2c"


def _enforce_digest_match(computed: str, expected: str, *, label: str) -> str:
    if computed != expected:
        raise ValueError(f"{label} digest mismatch: expected {expected}, got {computed}")
    return computed


def policy_spec() -> dict[str, Any]:
    return copy.deepcopy(_POLICY_SPEC)


def calibrators_for_policy() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(_CALIBRATORS)


def coefficients_digest() -> str:
    digest = hashlib.sha256(
        json.dumps(_coefficients_digest_payload(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _enforce_digest_match(
        digest, SELECTED_COEFFICIENTS_DIGEST_SHA256, label="A2 coefficients"
    )


def embedded_coefficients_sha256() -> str:
    computed = hashlib.sha256(
        json.dumps(_embedded_coefficients_payload(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _enforce_digest_match(
        computed, EMBEDDED_COEFFICIENTS_SHA256, label="A2 embedded_coefficients"
    )


def calibrator_for_transfer(transfer: str) -> str:
    if transfer in {"pq", "sdr"}:
        return "A2_TV123_Unified"
    raise ValueError(f"unsupported transfer {transfer!r}")


def _evaluate(score: float, transfer: str, encoded_height: int) -> float:
    if transfer not in {"pq", "sdr"}:
        raise ValueError(f"unsupported transfer {transfer!r}")
    if encoded_height not in _SUPPORTED_HEIGHTS:
        raise ValueError(f"unsupported encoded height {encoded_height}")
    i_2160 = 1.0 if encoded_height == 2160 else 0.0
    i_pq = 1.0 if transfer == "pq" else 0.0
    z = (
        A2_INTERCEPT
        + A2_CVQA_COEF * ((score - A2_CVQA_CENTER) / A2_CVQA_SCALE)
        + A2_I2160_COEF * i_2160
        + A2_I_PQ_COEF * i_pq
    )
    baseline = 1.0 + 4.0 * _sigmoid(z)
    span = A2_BOOST_END - A2_BOOST_START
    t = _clamp((score - A2_BOOST_START) / span, 0.0, 1.0)
    smoothstep = t * t * (3.0 - 2.0 * t)
    return _clamp(baseline + A2_AMPLITUDE * smoothstep, 1.0, 5.0)


@dataclass(frozen=True)
class PredictMosResult:
    estimated_perceptual_quality: float
    score_requested: float
    score_effective: float
    ood_below_support: bool
    policy: str
    calibrator: str
    route_weight_lambda: float
    model_key: str
    post_mos_offset: float = 0.0
    input_support_clamp: bool = False
    support_extrapolation: bool = False
    route_sigmoid_a: float = 0.0
    route_sigmoid_b: float = 0.0
    route_correction_d: float = 0.0
    calibration_algorithm: str = MOS_CALIBRATION_ALGORITHM


def predict_mos_detailed(
    canonical_cvqa_score: float,
    *,
    transfer: str,
    encoded_height: int,
    calibrator: str | None = None,
) -> PredictMosResult:
    try:
        requested = float(canonical_cvqa_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical_cvqa_score must be a finite number") from exc
    if not math.isfinite(requested):
        raise ValueError("canonical_cvqa_score must be finite")
    embedded_coefficients_sha256()
    coefficients_digest()
    key = calibrator or calibrator_for_transfer(transfer)
    spec = _CALIBRATORS.get(key)
    if spec is None:
        raise ValueError(f"unknown calibrator {key!r}")
    lo = float(_SUPPORT["score_min"])
    ood_below = requested < lo
    estimate = _evaluate(requested, transfer, encoded_height)
    return PredictMosResult(
        estimated_perceptual_quality=estimate,
        score_requested=requested,
        score_effective=requested,
        ood_below_support=ood_below,
        policy=MOS_CALIBRATION_ALGORITHM,
        calibrator=key,
        route_weight_lambda=float(spec["weight_lambda"]),
        model_key=str(spec["model_key"]),
        post_mos_offset=0.0,
        input_support_clamp=False,
        support_extrapolation=ood_below,
        route_sigmoid_a=A2_INTERCEPT,
        route_sigmoid_b=A2_CVQA_COEF,
        route_correction_d=A2_AMPLITUDE,
        calibration_algorithm=MOS_CALIBRATION_ALGORITHM,
    )


def predict_mos(
    canonical_cvqa_score: float,
    *,
    transfer: str,
    encoded_height: int,
    calibrator: str | None = None,
) -> float:
    return predict_mos_detailed(
        canonical_cvqa_score,
        transfer=transfer,
        encoded_height=encoded_height,
        calibrator=calibrator,
    ).estimated_perceptual_quality


def mos_calibration_record(
    outcome: Any,
    *,
    cvqa_upstream: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from pixvqa_cvqahdr.mos_calibration import (  # noqa: WPS433 — narrow cycle
        _json_safe_number,
        cvqa_upstream_compatibility,
    )

    pspec = policy_spec()
    record: dict[str, Any] = {
        "status": "available" if outcome.available else "unavailable",
        "research_status": RESEARCH_STATUS,
        "mos_calibration_algorithm": MOS_CALIBRATION_ALGORITHM,
        "mos_policy": MOS_CALIBRATION_ALGORITHM,
        "calibration_version": pspec["calibration_version"],
        "calibration_label": pspec["calibration_label"],
        "weight_kind": pspec["weight_kind"],
        "coefficient_cohort": pspec["coefficient_cohort"],
        "coefficient_source": pspec["coefficient_source"],
        "experiment_summary": pspec["experiment_summary"],
        "coefficients_digest_sha256": coefficients_digest(),
        "m1_implementation": "unified_sigmoid_height_pq_indicators_plus_smoothstep",
        "formula": pspec["formula"],
        "canonical_score_scale": "canonical CVQA (values >10 allowed)",
        "output_scale": OUTPUT_SCALE_LABEL,
        "canonical_score_input": _json_safe_number(outcome.canonical_score_input),
        "evidence_caveat": pspec["evidence_caveat"],
        "cvqa_upstream_compatibility": cvqa_upstream_compatibility(cvqa_upstream),
        "temporal_aggregation_note": (
            "E[f(score)] != f(E[score]) in general. MOS estimate applies to the "
            "canonical clip/video-level CVQA score only; per-frame scores are not "
            "individually MOS-mapped unless a separate aggregation policy is approved."
        ),
        "input_support_clamp": False,
        "input_support_policy": "unbounded_extrapolation",
        "unresolved_policy_note": (
            "A2-TV123 does not use v7 transfer-separated normalization, SDR native-MOS "
            "offset, M7 corrections or C2 low-tail. Input CVQA is not support-clamped; "
            "scores above 10 are valid, the smoothstep saturates at 10, and only "
            "the MOS output is bounded to [1, 5]."
        ),
        "manifest_id": pspec["manifest_id"],
        "coefficients_digest_note": (
            "coefficients_digest_sha256 pins the frozen A2-TV123 sigmoid tuple, "
            "smoothstep amplitude and decision-artifact provenance. This digest is "
            "independent of v7 fitted-model hashes and must not be interpreted as v7."
        ),
        "embedded_coefficients_sha256": embedded_coefficients_sha256(),
        "post_output_offset_by_transfer": dict(pspec["post_output_offset_by_transfer"]),
        "ai_loss_fraction": A2_AI_LOSS_FRACTION,
        "v7_legacy_components": False,
        "algorithm_selector_env": "PIXVQA_MOS_CALIBRATION_ALGORITHM",
    }
    if outcome.reason == "cvqa_upstream_incompatible":
        record["cvqa_upstream_gate_status"] = "blocked"
        record["status"] = "unavailable"
    if outcome.reason is not None:
        record["reason"] = outcome.reason
    score_requested = _json_safe_number(outcome.score_requested)
    if score_requested is not None:
        record["score_requested"] = score_requested
        record["score_effective"] = _json_safe_number(outcome.score_effective)
        record["ood_below_support"] = outcome.ood_below_support
        record["input_support_clamp"] = False
        if outcome.support_extrapolation:
            record["support_extrapolation"] = True
    if outcome.calibrator is not None:
        record["calibrator"] = outcome.calibrator
        spec = _CALIBRATORS[outcome.calibrator]
        record["research_arm"] = spec["research_arm"]
        record["route_weight_lambda"] = outcome.route_weight_lambda or spec["weight_lambda"]
        record["model_key"] = outcome.model_key or spec["model_key"]
        record["support"] = dict(spec["support"])
        record["unified_sigmoid"] = dict(spec["sigmoid_coefficients"])
        record["smoothstep_amplitude"] = dict(spec["amplitude"])
        if outcome.post_mos_offset is not None:
            record["post_mos_offset"] = outcome.post_mos_offset
    if outcome.available and outcome.mos_calibrated_cvqa is not None:
        record["mos_calibrated_cvqa"] = outcome.mos_calibrated_cvqa
        record["estimated_perceptual_quality"] = outcome.mos_calibrated_cvqa
    if outcome.domain is not None:
        record["transfer"] = outcome.domain.transfer
        record["encoded_height"] = outcome.domain.encoded_height
        record["encoded_width"] = outcome.domain.encoded_width
        record["color_transfer_tag"] = outcome.domain.color_transfer_tag
    record["note"] = (
        f"{OUTPUT_SCALE_LABEL} is an experimental research mapping from canonical "
        "CVQA-HDR-NR using A2-TV123; it is not validated for production subjective-quality "
        "prediction and must not be confused with frozen v7+C2 calibration."
    )
    return record