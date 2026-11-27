"""Canonical CVQA-to-estimated-MOS adapter (B08 H1, w=0.05).

The canonical clip-level CVQA score is never changed. Only supported SDR/PQ
1080p/2160p routes are mapped, with existing upstream identity gates.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, replace
from typing import Any

from pixvqa_cvqahdr import PINNED_CHECKPOINT_SHA256, PINNED_COMMIT, PINNED_PROFILE_SHA256
from pixvqa_video.ffmpeg_bins import resolve_ffprobe_legacy
from pixvqa_cvqahdr import cvqa_mos_b08 as mapping

MOS_CALIBRATION_ALGORITHM = mapping.ALGORITHM
MOS_POLICY = MOS_CALIBRATION_ALGORITHM
CALIBRATION_VERSION = mapping.VERSION
CALIBRATOR = "A2_TV123_H1_B08"
OUTPUT_SCALE_LABEL = "estimated perceptual quality 1-5"
RESEARCH_STATUS = "provisional"
EVIDENCE_CAVEAT = (
    "TV123: 145 clips, 15 source families, 435 human ratings; amplitude selected "
    "with 75% human / 25% AI40 loss. H1: 28 HDR2160 clips, 14 families, 8 ballots "
    "collected, 7 retained (196 ratings). Uniform correction strength 0.05. "
    "Cross-route and out-of-H1-support extrapolation are engineering assumptions. "
    "Not subjectively certified for production."
)

FFPROBE_TIMEOUT_SEC = 30.0
_PROBE_ERRORS = frozenset({
    "ffprobe_timeout",
    "ffprobe_failed",
    "invalid_ffprobe_json",
    "invalid_ffprobe_payload",
    "no_video_stream",
    "invalid_geometry",
    "non_positive_geometry",
})

_SUPPORTED_HEIGHTS = frozenset({1080, 2160})

_PQ_TRANSFER_OK = frozenset({"smpte2084", "pq", "st2084"})
_SDR_TRANSFER_OK = frozenset({
    "bt709",
    "smpte170m",
    "bt470bg",
    "bt470m",
    "iec61966-2-1",
    "gamma22",
    "gamma28",
    "bt2020-10",
    "bt2020-12",
})
_BT2020_AMBIGUOUS = frozenset({"bt2020"})
_HLG_TRANSFER = frozenset({"arib-std-b67", "hlg", "std-b67", "bt2020-hlg"})

def _norm_tag(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip().lower()


@dataclass(frozen=True)
class MosDomain:
    transfer: str  # "pq" or "sdr"
    encoded_height: int
    color_transfer_tag: str
    encoded_width: int


@dataclass(frozen=True)
class PredictMosResult:
    estimated_perceptual_quality: float
    score_requested: float
    score_effective: float
    ood_below_support: bool
    ood_above_support: bool
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


@dataclass(frozen=True)
class MosCalibrationOutcome:
    available: bool
    mos_calibrated_cvqa: float | None
    reason: str | None
    domain: MosDomain | None
    calibrator: str | None
    canonical_score_input: float | None
    policy: str | None = None
    score_requested: float | None = None
    score_effective: float | None = None
    ood_below_support: bool = False
    ood_above_support: bool = False
    route_weight_lambda: float | None = None
    model_key: str | None = None
    post_mos_offset: float | None = None
    input_support_clamp: bool | None = None
    support_extrapolation: bool = False
    route_sigmoid_a: float | None = None
    route_sigmoid_b: float | None = None
    route_correction_d: float | None = None
    calibration_algorithm: str = MOS_CALIBRATION_ALGORITHM


def _probe_error_reason(third: str | None) -> str | None:
    if third is None:
        return None
    if third in _PROBE_ERRORS:
        return third
    if third.startswith("ffprobe_failed:"):
        return "ffprobe_failed"
    return None


def classify_transfer_tag(tag: str) -> tuple[str | None, str | None]:
    """Return (transfer, reject_reason) for ffprobe color_transfer."""
    tr = _norm_tag(tag)
    if not tr or tr in {"unknown", "unspecified", "reserved"}:
        return None, "transfer_missing"
    if tr in _HLG_TRANSFER or "hlg" in tr or "b67" in tr:
        return None, "transfer_hlg"
    if tr in _BT2020_AMBIGUOUS:
        return None, "transfer_unsupported"
    if tr in _PQ_TRANSFER_OK:
        return "pq", None
    if tr in _SDR_TRANSFER_OK:
        return "sdr", None
    return None, "transfer_unsupported"


def probe_encoded_geometry(path: str) -> tuple[int, int, str | None]:
    ffprobe = resolve_ffprobe_legacy("PIXVQA_CVQAHDR_FFPROBE")
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,color_transfer",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=FFPROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return 0, 0, "ffprobe_timeout"
    except OSError as exc:
        return 0, 0, f"ffprobe_failed:{exc}"
    except subprocess.CalledProcessError as exc:
        return 0, 0, f"ffprobe_failed:{exc}"
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        return 0, 0, "invalid_ffprobe_json"
    if not isinstance(payload, dict):
        return 0, 0, "invalid_ffprobe_payload"
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return 0, 0, "invalid_ffprobe_payload"
    if not streams:
        return 0, 0, "no_video_stream"
    stream = streams[0]
    if not isinstance(stream, dict):
        return 0, 0, "invalid_ffprobe_payload"
    try:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except (TypeError, ValueError):
        return 0, 0, "invalid_geometry"
    if width <= 0 or height <= 0:
        return 0, 0, "non_positive_geometry"
    transfer_tag = _norm_tag(stream.get("color_transfer"))
    return width, height, transfer_tag or None


def resolve_mos_domain(path: str) -> tuple[MosDomain | None, str | None]:
    width, height, transfer_tag = probe_encoded_geometry(path)
    probe_err = _probe_error_reason(transfer_tag)
    if probe_err is not None:
        return None, probe_err
    if height not in _SUPPORTED_HEIGHTS:
        return None, "height_unsupported"
    if not transfer_tag:
        return None, "transfer_missing"
    transfer, reject = classify_transfer_tag(transfer_tag)
    if transfer is None:
        return None, reject
    return MosDomain(
        transfer=transfer,
        encoded_height=height,
        color_transfer_tag=transfer_tag,
        encoded_width=width,
    ), None


def _json_safe_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def coefficients_digest() -> str:
    return mapping.verified_digest()


def predict_mos_detailed(
    canonical_cvqa_score: float, *, transfer: str, encoded_height: int
) -> PredictMosResult:
    requested = _json_safe_number(canonical_cvqa_score)
    if requested is None:
        raise ValueError("canonical_cvqa_score must be finite")
    estimate = mapping.predict_mos(requested, transfer=transfer, encoded_height=encoded_height)
    return PredictMosResult(
        estimated_perceptual_quality=estimate,
        score_requested=requested, score_effective=requested,
        ood_below_support=requested < 0, ood_above_support=False,
        policy=MOS_POLICY, calibrator=CALIBRATOR,
        route_weight_lambda=0.25, model_key=MOS_CALIBRATION_ALGORITHM,
        support_extrapolation=requested < 0,
        route_sigmoid_a=mapping.A2[0], route_sigmoid_b=mapping.A2[1],
        route_correction_d=mapping.AMPLITUDE,
    )


def predict_mos(canonical_cvqa_score: float, *, transfer: str, encoded_height: int) -> float:
    return predict_mos_detailed(
        canonical_cvqa_score, transfer=transfer, encoded_height=encoded_height
    ).estimated_perceptual_quality


def apply_mos_calibration(canonical_cvqa_score: float, path: str) -> MosCalibrationOutcome:
    base = MosCalibrationOutcome(
        available=False, mos_calibrated_cvqa=None, reason=None,
        domain=None, calibrator=None, canonical_score_input=canonical_cvqa_score,
        policy=MOS_POLICY,
    )
    requested = _json_safe_number(canonical_cvqa_score)
    if requested is None:
        return replace(base, reason="calibration_error:canonical_cvqa_score must be finite")
    domain, reason = resolve_mos_domain(path)
    if domain is None:
        return replace(base, reason=reason)
    base = replace(base, domain=domain, calibrator=CALIBRATOR)
    try:
        detail = predict_mos_detailed(requested, transfer=domain.transfer,
                                      encoded_height=domain.encoded_height)
    except ValueError as exc:
        return replace(base, reason=f"calibration_error:{exc}")
    return replace(
        base, available=True, mos_calibrated_cvqa=detail.estimated_perceptual_quality,
        score_requested=detail.score_requested, score_effective=detail.score_effective,
        ood_below_support=detail.ood_below_support, ood_above_support=False,
        route_weight_lambda=detail.route_weight_lambda, model_key=detail.model_key,
        post_mos_offset=0.0, input_support_clamp=False,
        support_extrapolation=detail.support_extrapolation,
        route_sigmoid_a=detail.route_sigmoid_a, route_sigmoid_b=detail.route_sigmoid_b,
        route_correction_d=detail.route_correction_d,
    )


def minimal_unavailable_mos_record(canonical_score_input: object, reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable", "reason": reason,
        "mos_calibration_algorithm": MOS_CALIBRATION_ALGORITHM,
        "mos_policy": MOS_POLICY, "calibration_version": CALIBRATION_VERSION,
        "research_status": RESEARCH_STATUS, "h1_shape_weight": mapping.WEIGHT,
        "canonical_score_input": _json_safe_number(canonical_score_input),
        "output_scale": OUTPUT_SCALE_LABEL,
        "note": "MOS unavailable; canonical CVQA unchanged.",
    }


def cvqa_upstream_from_run_record(record: dict[str, Any]) -> dict[str, Any]:
    upstream_block = record.get("upstream")
    if isinstance(upstream_block, dict) and "commit_verified" not in upstream_block:
        upstream_block = upstream_block.get("upstream") or upstream_block
    checkpoint = record.get("checkpoint") or {}
    profile = record.get("calibration_profile") or {}
    return {
        "upstream": upstream_block if isinstance(upstream_block, dict) else {},
        "checkpoint_sha256": checkpoint.get("sha256"),
        "profile_sha256": profile.get("sha256"),
    }


def cvqa_upstream_compatibility(cvqa_upstream: dict[str, Any] | None) -> dict[str, Any]:
    if not cvqa_upstream:
        return {
            "compatible": False,
            "integration_gate": "unverified",
            "missing_fields": ["upstream", "checkpoint_sha256", "profile_sha256"],
            "incompatible_fields": [],
            "note": "no upstream metadata supplied",
        }
    upstream = cvqa_upstream.get("upstream") or {}
    actual_commit = upstream.get("actual_commit")
    checkpoint_sha = cvqa_upstream.get("checkpoint_sha256")
    profile_sha = cvqa_upstream.get("profile_sha256")
    missing: list[str] = []
    incompatible: list[str] = []
    if not actual_commit:
        missing.append("actual_commit")
    if not checkpoint_sha:
        missing.append("checkpoint_sha256")
    if not profile_sha:
        missing.append("profile_sha256")
    commit_match = actual_commit == PINNED_COMMIT if actual_commit else False
    checkpoint_match = checkpoint_sha == PINNED_CHECKPOINT_SHA256 if checkpoint_sha else False
    profile_match = profile_sha == PINNED_PROFILE_SHA256 if profile_sha else False
    if actual_commit and actual_commit != PINNED_COMMIT:
        incompatible.append("actual_commit")
    if checkpoint_sha and not checkpoint_match:
        incompatible.append("checkpoint_sha256")
    if profile_sha and not profile_match:
        incompatible.append("profile_sha256")
    compatible = commit_match and checkpoint_match and profile_match and not missing
    return {
        "compatible": compatible,
        "pinned_commit": PINNED_COMMIT,
        "pinned_checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "pinned_profile_sha256": PINNED_PROFILE_SHA256,
        "actual_commit": actual_commit,
        "commit_verified": commit_match,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha256_verified": checkpoint_match,
        "profile_sha256": profile_sha,
        "profile_sha256_verified": profile_match,
        "missing_fields": missing,
        "incompatible_fields": incompatible,
        "integration_gate": "verified" if compatible else "unverified",
        "note": (
            "Explicit SHA comparison against pixvqa_cvqahdr.__init__ pins. "
            "preflight verify_upstream() checks git HEAD only; checkpoint and profile "
            "digests are recorded per run but are not hard preflight failures for "
            "canonical CVQA scoring."
        ),
    }


def cvqa_upstream_gate(cvqa_upstream: dict[str, Any] | None) -> dict[str, Any]:
    return cvqa_upstream_compatibility(cvqa_upstream)


def gate_reviewed_mos_outcome(
    outcome: MosCalibrationOutcome,
    cvqa_upstream: dict[str, Any] | None,
) -> MosCalibrationOutcome:
    """Block MOS output when CVQA upstream pins are missing or incompatible."""
    if not outcome.available:
        return outcome
    compat = cvqa_upstream_compatibility(cvqa_upstream)
    if compat["compatible"]:
        return outcome
    return MosCalibrationOutcome(
        available=False,
        mos_calibrated_cvqa=None,
        reason="cvqa_upstream_incompatible",
        domain=outcome.domain,
        calibrator=outcome.calibrator,
        canonical_score_input=outcome.canonical_score_input,
        policy=outcome.policy,
        score_requested=outcome.score_requested,
        score_effective=outcome.score_effective,
        ood_below_support=outcome.ood_below_support,
        ood_above_support=outcome.ood_above_support,
        route_weight_lambda=outcome.route_weight_lambda,
        model_key=outcome.model_key,
        post_mos_offset=outcome.post_mos_offset,
        input_support_clamp=outcome.input_support_clamp,
        support_extrapolation=outcome.support_extrapolation,
        route_sigmoid_a=outcome.route_sigmoid_a,
        route_sigmoid_b=outcome.route_sigmoid_b,
        route_correction_d=outcome.route_correction_d,
        calibration_algorithm=MOS_CALIBRATION_ALGORITHM,
    )


def mos_calibration_record(
    outcome: MosCalibrationOutcome, *, cvqa_upstream: dict[str, Any] | None = None,
) -> dict[str, Any]:
    digest = mapping.verified_digest()
    record: dict[str, Any] = {
        "status": "available" if outcome.available else "unavailable",
        "research_status": RESEARCH_STATUS,
        "mos_calibration_algorithm": MOS_CALIBRATION_ALGORITHM,
        "mos_policy": MOS_POLICY,
        "calibration_version": CALIBRATION_VERSION,
        "calibration_label": "A2-TV123 with uniform B08 H1 5% correction",
        "weight_kind": "historical_AI25_amplitude_and_H1_shape_weight",
        "coefficient_cohort": "TV123_145clips_435_human_40_ai_H1_8ballots_7retained",
        "coefficient_source": "docs/dev/CVQA_MOS_A2_TV123_INTEGRATION.md",
        "manifest_id": CALIBRATION_VERSION,
        "coefficients_digest_sha256": digest,
        "embedded_coefficients_sha256": digest,
        "formula": "a2_plus_anchored_fullrange_h1_relative_headroom",
        "canonical_score_scale": "canonical CVQA (values >10 allowed)",
        "canonical_score_input": _json_safe_number(outcome.canonical_score_input),
        "output_scale": OUTPUT_SCALE_LABEL,
        "evidence_caveat": EVIDENCE_CAVEAT,
        "cvqa_upstream_compatibility": cvqa_upstream_compatibility(cvqa_upstream),
        "input_support_clamp": False,
        "input_support_policy": "unbounded_extrapolation",
        "post_output_offset_by_transfer": {"pq": 0.0, "sdr": 0.0},
        "ai_loss_fraction": 0.25,
        "h1_shape_weight": mapping.WEIGHT,
        "h1_correction": mapping.coefficient_payload(),
        "v7_legacy_components": False,
        "h1_observed_score_range": [mapping.L, mapping.H],
        "h1_observed_route": {"transfer": "pq", "encoded_height": 2160},
        "temporal_aggregation_note": "Maps canonical clip-level CVQA; not a per-frame MOS pooling policy.",
        "note": "Canonical B08 estimated quality; not production subjectively certified.",
    }
    if outcome.reason:
        record["reason"] = outcome.reason
    if outcome.reason == "cvqa_upstream_incompatible":
        record["cvqa_upstream_gate_status"] = "blocked"
    if outcome.score_requested is not None:
        record.update(score_requested=outcome.score_requested, score_effective=outcome.score_effective,
                      ood_below_support=outcome.ood_below_support)
        if outcome.support_extrapolation:
            record["support_extrapolation"] = True
    if outcome.calibrator is not None:
        record.update(
            calibrator=CALIBRATOR, model_key=MOS_CALIBRATION_ALGORITHM,
            research_arm="B08_uniform_w005", route_weight_lambda=0.25,
            support={"score_min": 0.0}, post_mos_offset=0.0,
            unified_sigmoid=dict(zip(("intercept","cvqa_coef","i_2160_coef","i_pq_coef"), mapping.A2)),
            smoothstep_amplitude={"start_cvqa":mapping.BOOST_START,"end_cvqa":mapping.BOOST_END,
                                  "amplitude_mos":mapping.AMPLITUDE},
        )
    if outcome.domain is not None:
        record.update(transfer=outcome.domain.transfer, encoded_height=outcome.domain.encoded_height,
                      encoded_width=outcome.domain.encoded_width, color_transfer_tag=outcome.domain.color_transfer_tag)
        if outcome.score_requested is not None:
            record["h1_score_extrapolation"] = not mapping.L <= outcome.score_requested <= mapping.H
            record["h1_cross_route_extrapolation"] = (outcome.domain.transfer, outcome.domain.encoded_height) != ("pq", 2160)
    if outcome.available and outcome.mos_calibrated_cvqa is not None:
        record["mos_calibrated_cvqa"] = outcome.mos_calibrated_cvqa
        record["estimated_perceptual_quality"] = outcome.mos_calibrated_cvqa
    return record
