"""Estimated perceptual quality 1–5 calibration for CVQA-HDR-NR canonical 0–10 scores.

Single scientifically selected transfer (policy ``selected_s1_sdr_p1_hdr_c0``), applied
automatically when ffprobe metadata is supported. No policy selection or environment
switch is required.

Normalized score::

    z = (CVQA − 8.382293636852387) / 1.1647912353499874

Estimated MOS (final clamp only; CVQA is never clamped)::

    MOS = clip(1 + 4·σ(a_T + b_T·z) + d_{T,h}, 1, 5)

Supported routes: SDR or HDR PQ at encoded heights 1080 and 2160. HLG, ambiguous
``bt2020``, missing/unknown transfer, and unsupported heights fail closed. Out-of-support
CVQA scores extrapolate monotonically with OOD flags only.

Source coefficients: ``PIXVQA_CVQA_to_MOS_Selected_Coefficients_v5.json`` (frozen S1
composite). ``PIXVQA_CVQA_MOS_POLICY`` is **obsolete** — non-empty values other than
``selected_s1_sdr_p1_hdr_c0`` fail closed so legacy formulas cannot be activated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from pixvqa_cvqahdr import PINNED_CHECKPOINT_SHA256, PINNED_COMMIT, PINNED_PROFILE_SHA256
from pixvqa_video.ffmpeg_bins import resolve_ffprobe_legacy

# Obsolete: ignored when unset or when set to MOS_POLICY; any other non-empty value fails closed.
ENV_MOS_POLICY = "PIXVQA_CVQA_MOS_POLICY"
MOS_POLICY = "selected_s1_sdr_p1_hdr_c0"
DEFAULT_MOS_POLICY = MOS_POLICY
SELECTED_MOS_POLICY = MOS_POLICY

FROZEN_S1_COEFFICIENTS_BUNDLE_SHA256 = (
    "67d39be487fccee6b0a08ef33a4dd7e8adf1745578eff9060ab48e394069a02c"
)
SDR_SELECTION_LOCK_SHA256 = (
    "15dbde98a70c3a4ac7e2f7036a084d589e4154f320ebf4dba657f9bb4c894581"
)
HDR_SELECTION_LOCK_SHA256 = (
    "5a4749f6fae7d3a7a9c6736910d83c02d41e4d7630e09bba4476ebfed0ee68cd"
)
SELECTED_COEFFICIENTS_SOURCE_JSON_SHA256 = (
    "ba60d7c0d6154c7d66f16e2e453b69a936e7b96e3053c3b54747377537557718"
)
OUTPUT_SCALE_LABEL = "estimated perceptual quality 1-5"

CALIBRATION_VERSION = "selected-s1-sdr-p1-hdr-c0-unclamped-input-20261114-v2"
CALIBRATION_LABEL = "Selected S1 composite CVQA-to-MOS calibration"
WEIGHT_KIND = "route_mixed"
RESEARCH_STATUS = "provisional"
COEFFICIENT_COHORT = "old392"
COEFFICIENT_SOURCE = (
    "perceptual_optimization/calibration/production_readiness_20261113/"
    "CVQA_TO_MOS_SCIENTIFIC_REVISION_20261114/data/"
    "PIXVQA_CVQA_to_MOS_Selected_Coefficients_v5.json"
)
MANIFEST_ID = "pixvqa_calib_s1_f1_392_20261114_v1"
HDR_WEIGHT_LAMBDA = 0.0
SDR_WEIGHT_LAMBDA = 1.0
SDR_POST_MOS_OFFSET = 0.4105017796126768
PQ_GAMMA_1080 = -0.10199018084158194
GAMMA_2160_SDR = 0.16798189863001703
GAMMA_2160_PQ = 0.13694653969694998

EVIDENCE_CAVEAT = (
    "User-selected S1 composite: effective two-parameter logistic per transfer plus "
    "combined route correction d_{T,h} (algebraically equivalent to frozen S1 M1+M7 "
    "with SDR post-offset). Input scores are NOT clamped to historical training support "
    "[4.1344840888613135, 10.051497077192835]; OOD flags warn only. Extrapolation "
    "uncertainty accepted. AVT informed U-v2 routing; retrospective metrics only. "
    "Not production subjective-quality certified."
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

_NORMALIZATION = {"mean": 8.382293636852387, "std": 1.1647912353499874}
_SUPPORT = {
    "score_min": 4.1344840888613135,
    "score_max": 10.051497077192835,
}
_SHARED_SUPPORT = dict(_SUPPORT)
_SHARED_NORMALIZATION = dict(_NORMALIZATION)

# Effective presentation parameters (full precision; equivalent to frozen S1 composite).
_ROUTE_PARAMS: dict[tuple[str, int], dict[str, float]] = {
    ("sdr", 1080): {
        "a": -0.10602048052744088,
        "b": 1.0583289781579686,
        "d": 0.4105017796126768,
    },
    ("sdr", 2160): {
        "a": -0.10602048052744088,
        "b": 1.0583289781579686,
        "d": 0.4105017796126768 + GAMMA_2160_SDR,
    },
    ("pq", 1080): {
        "a": -0.3136256719154792,
        "b": 0.9742692529668485,
        "d": PQ_GAMMA_1080,
    },
    ("pq", 2160): {
        "a": -0.3136256719154792,
        "b": 0.9742692529668485,
        "d": PQ_GAMMA_1080 + GAMMA_2160_PQ,
    },
}

# Stable calibrator identifiers and frozen training metadata (provenance only).
_CALIBRATORS: dict[str, dict[str, Any]] = {
    "U0_M7_Bias": {
        "route": "pq",
        "research_arm": "Original_U",
        "weight_kind": "baseline",
        "weight_lambda": 0.0,
        "model_key": "old392|Original_U|baseline|0.0",
        "selected_ridge": 0.03,
        "normalization": dict(_NORMALIZATION),
        "support": dict(_SUPPORT),
        "route_params_1080": dict(_ROUTE_PARAMS[("pq", 1080)]),
        "route_params_2160": dict(_ROUTE_PARAMS[("pq", 2160)]),
    },
    "U_v2_M7_Bias": {
        "route": "sdr",
        "research_arm": "U_v2",
        "weight_kind": "target_mos",
        "weight_lambda": 1.0,
        "model_key": "old392|U_v2|target|1.0",
        "selected_ridge": 0.03,
        "normalization": dict(_NORMALIZATION),
        "support": dict(_SUPPORT),
        "route_params_1080": dict(_ROUTE_PARAMS[("sdr", 1080)]),
        "route_params_2160": dict(_ROUTE_PARAMS[("sdr", 2160)]),
        "sdr_post_mos_offset": SDR_POST_MOS_OFFSET,
    },
}

_POLICY_SPEC: dict[str, Any] = {
    "policy_id": MOS_POLICY,
    "input_support_clamp": False,
    "input_support_policy": "unbounded_extrapolation",
    "calibration_version": CALIBRATION_VERSION,
    "calibration_label": CALIBRATION_LABEL,
    "weight_kind": WEIGHT_KIND,
    "hdr_weight_lambda": HDR_WEIGHT_LAMBDA,
    "sdr_weight_lambda": SDR_WEIGHT_LAMBDA,
    "coefficient_cohort": COEFFICIENT_COHORT,
    "coefficient_source": COEFFICIENT_SOURCE,
    "coefficient_source_json_sha256": SELECTED_COEFFICIENTS_SOURCE_JSON_SHA256,
    "manifest_id": MANIFEST_ID,
    "coefficients_bundle_sha256": FROZEN_S1_COEFFICIENTS_BUNDLE_SHA256,
    "post_output_offset_by_transfer": {"pq": 0.0, "sdr": SDR_POST_MOS_OFFSET},
    "sdr_selection_lock_sha256": SDR_SELECTION_LOCK_SHA256,
    "hdr_selection_lock_sha256": HDR_SELECTION_LOCK_SHA256,
    "evidence_caveat": EVIDENCE_CAVEAT,
    "calibrators": _CALIBRATORS,
    "formula": "clip(1+4*sigmoid(a+b*z)+d,1,5)",
}

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

_OBSOLETE_POLICIES = frozenset({
    "legacy_target_lambda1_both",
    "reviewed_s1_hdr0_sdr1",
})


def _norm_tag(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip().lower()


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def resolve_mos_policy(explicit: str | None = None) -> tuple[str | None, str | None]:
    """Return ``(policy_id, error_reason)``. Only the fixed selected policy is valid."""
    if explicit is not None:
        policy_id = str(explicit).strip()
        if policy_id != MOS_POLICY:
            return None, f"invalid_policy:{policy_id}"
        return MOS_POLICY, None
    env_raw = os.environ.get(ENV_MOS_POLICY, "").strip()
    if not env_raw or env_raw == MOS_POLICY:
        return MOS_POLICY, None
    if env_raw in _OBSOLETE_POLICIES:
        return None, f"obsolete_policy_env:{env_raw}"
    return None, f"obsolete_policy_env:{env_raw}"


def policy_spec(policy_id: str | None = None) -> dict[str, Any]:
    pid = policy_id or MOS_POLICY
    if pid != MOS_POLICY:
        raise ValueError(f"unknown MOS policy {pid!r}")
    return _POLICY_SPEC


def calibrators_for_policy(policy_id: str | None = None) -> dict[str, dict[str, Any]]:
    policy_spec(policy_id)
    return _CALIBRATORS


def input_support_clamp_enabled(policy_id: str | None = None) -> bool:
    policy_spec(policy_id)
    return False


def _coefficients_digest_payload() -> dict[str, Any]:
    return {
        "policy_id": MOS_POLICY,
        "s1_coefficients_bundle_sha256": FROZEN_S1_COEFFICIENTS_BUNDLE_SHA256,
        "post_output_offset_by_transfer": _POLICY_SPEC["post_output_offset_by_transfer"],
        "sdr_selection_lock_sha256": SDR_SELECTION_LOCK_SHA256,
        "hdr_selection_lock_sha256": HDR_SELECTION_LOCK_SHA256,
        "input_support_clamp": False,
    }


SELECTED_COEFFICIENTS_DIGEST_SHA256 = (
    "463b641347d68585ea9f99fc6d8c58b3ea274896458f6b1e58cbf58592b43cd3"
)


def _embedded_coefficients_payload() -> dict[str, Any]:
    return {
        "policy_id": MOS_POLICY,
        "formula": _POLICY_SPEC["formula"],
        "normalization": dict(_NORMALIZATION),
        "routes": {
            f"{transfer}_{height}": dict(params)
            for (transfer, height), params in sorted(_ROUTE_PARAMS.items())
        },
        "s1_coefficients_bundle_sha256": FROZEN_S1_COEFFICIENTS_BUNDLE_SHA256,
        "coefficient_source_json_sha256": SELECTED_COEFFICIENTS_SOURCE_JSON_SHA256,
        "sdr_selection_lock_sha256": SDR_SELECTION_LOCK_SHA256,
        "hdr_selection_lock_sha256": HDR_SELECTION_LOCK_SHA256,
        "post_output_offset_by_transfer": dict(_POLICY_SPEC["post_output_offset_by_transfer"]),
        "input_support_clamp": False,
    }


EMBEDDED_COEFFICIENTS_SHA256 = (
    "74cf758869796c5c3ba94ac7aba3b8ed808b664940f990e4b5f5c1b7920d6111"
)


def _embedded_coefficients_digest() -> str:
    return hashlib.sha256(
        json.dumps(_embedded_coefficients_payload(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _enforce_digest_match(computed: str, expected: str, *, label: str) -> str:
    if computed != expected:
        raise ValueError(f"{label} digest mismatch: expected {expected}, got {computed}")
    return computed


def coefficients_digest(policy_id: str | None = None) -> str:
    policy_spec(policy_id)
    digest = hashlib.sha256(
        json.dumps(_coefficients_digest_payload(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _enforce_digest_match(digest, SELECTED_COEFFICIENTS_DIGEST_SHA256, label="coefficients")


def embedded_coefficients_sha256(policy_id: str | None = None) -> str:
    policy_spec(policy_id)
    computed = _embedded_coefficients_digest()
    return _enforce_digest_match(
        computed, EMBEDDED_COEFFICIENTS_SHA256, label="embedded_coefficients"
    )


def post_output_offset_for_transfer(transfer: str) -> float:
    return SDR_POST_MOS_OFFSET if transfer == "sdr" else 0.0


def route_params(transfer: str, encoded_height: int) -> dict[str, float]:
    try:
        return dict(_ROUTE_PARAMS[(transfer, encoded_height)])
    except KeyError as exc:
        raise ValueError(f"unsupported route {transfer!r} height {encoded_height}") from exc


def calibrator_for_transfer(transfer: str) -> str | None:
    if transfer == "pq":
        return "U0_M7_Bias"
    if transfer == "sdr":
        return "U_v2_M7_Bias"
    return None


def _assert_calibrator_matches_transfer(calibrator: str, transfer: str) -> None:
    spec = _CALIBRATORS.get(calibrator)
    if spec is None:
        raise ValueError(f"unknown calibrator {calibrator!r}")
    if spec["route"] != transfer:
        raise ValueError(
            f"calibrator {calibrator!r} is trained for {spec['route']!r} content, "
            f"not {transfer!r}"
        )


def _evaluate_route(score: float, transfer: str, encoded_height: int) -> float:
    params = route_params(transfer, encoded_height)
    mean = _NORMALIZATION["mean"]
    std = _NORMALIZATION["std"]
    z = (score - mean) / std
    logit = params["a"] + params["b"] * z
    return _clamp(1.0 + 4.0 * _sigmoid(logit) + params["d"], 1.0, 5.0)


def predict_mos(
    canonical_cvqa_score: float,
    *,
    transfer: str,
    encoded_height: int,
    calibrator: str | None = None,
    policy: str | None = None,
) -> float:
    """Map canonical CVQA 0–10 to estimated perceptual quality 1–5."""
    return predict_mos_detailed(
        canonical_cvqa_score,
        transfer=transfer,
        encoded_height=encoded_height,
        calibrator=calibrator,
        policy=policy,
    ).estimated_perceptual_quality


def predict_mos_detailed(
    canonical_cvqa_score: float,
    *,
    transfer: str,
    encoded_height: int,
    calibrator: str | None = None,
    policy: str | None = None,
) -> "PredictMosResult":
    policy_id, policy_err = resolve_mos_policy(policy)
    if policy_err:
        raise ValueError(policy_err)
    try:
        requested = float(canonical_cvqa_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical_cvqa_score must be a finite number") from exc
    if not math.isfinite(requested):
        raise ValueError("canonical_cvqa_score must be finite")
    if transfer not in {"pq", "sdr"}:
        raise ValueError(f"unsupported transfer {transfer!r}")
    if encoded_height not in _SUPPORTED_HEIGHTS:
        raise ValueError(f"unsupported encoded height {encoded_height}")
    if calibrator is not None:
        key = calibrator
        _assert_calibrator_matches_transfer(key, transfer)
    else:
        key = calibrator_for_transfer(transfer)
        if key is None:
            raise ValueError(f"unsupported transfer {transfer!r}")
    spec = _CALIBRATORS[key]
    lo = float(_SUPPORT["score_min"])
    hi = float(_SUPPORT["score_max"])
    ood_below = requested < lo
    ood_above = requested > hi
    estimate = _evaluate_route(requested, transfer, encoded_height)
    route = route_params(transfer, encoded_height)
    return PredictMosResult(
        estimated_perceptual_quality=estimate,
        score_requested=requested,
        score_effective=requested,
        ood_below_support=ood_below,
        ood_above_support=ood_above,
        policy=policy_id,
        calibrator=key,
        route_weight_lambda=float(spec["weight_lambda"]),
        model_key=str(spec["model_key"]),
        post_mos_offset=post_output_offset_for_transfer(transfer),
        input_support_clamp=False,
        support_extrapolation=ood_below or ood_above,
        route_sigmoid_a=route["a"],
        route_sigmoid_b=route["b"],
        route_correction_d=route["d"],
    )


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


def minimal_unavailable_mos_record(
    canonical_score_input: object,
    reason: str,
    *,
    policy: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "research_status": RESEARCH_STATUS,
        "mos_policy": policy or MOS_POLICY,
        "canonical_score_input": _json_safe_number(canonical_score_input),
        "output_scale": OUTPUT_SCALE_LABEL,
        "reason": reason,
        "note": f"MOS mapping suppressed ({reason}); canonical CVQA score unchanged.",
    }


def apply_mos_calibration(
    canonical_cvqa_score: float,
    path: str,
    *,
    policy: str | None = None,
) -> MosCalibrationOutcome:
    """Attempt 1–5 estimate mapping; never mutates the canonical CVQA score."""
    policy_id, policy_err = resolve_mos_policy(policy)
    if policy_err:
        return MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason=policy_err,
            domain=None,
            calibrator=None,
            canonical_score_input=canonical_cvqa_score,
            policy=None,
        )
    try:
        requested = float(canonical_cvqa_score)
    except (TypeError, ValueError):
        return MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason="calibration_error:canonical_cvqa_score must be a finite number",
            domain=None,
            calibrator=None,
            canonical_score_input=canonical_cvqa_score,
            policy=policy_id,
        )
    if not math.isfinite(requested):
        return MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason="calibration_error:canonical_cvqa_score must be finite",
            domain=None,
            calibrator=None,
            canonical_score_input=canonical_cvqa_score,
            policy=policy_id,
        )
    domain, reason = resolve_mos_domain(path)
    if domain is None:
        return MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason=reason,
            domain=None,
            calibrator=None,
            canonical_score_input=canonical_cvqa_score,
            policy=policy_id,
        )
    calibrator = calibrator_for_transfer(domain.transfer)
    if calibrator is None:
        return MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason="transfer_unsupported",
            domain=domain,
            calibrator=None,
            canonical_score_input=canonical_cvqa_score,
            policy=policy_id,
        )
    try:
        detail = predict_mos_detailed(
            canonical_cvqa_score,
            transfer=domain.transfer,
            encoded_height=domain.encoded_height,
            calibrator=calibrator,
            policy=policy_id,
        )
    except ValueError as exc:
        return MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason=f"calibration_error:{exc}",
            domain=domain,
            calibrator=calibrator,
            canonical_score_input=canonical_cvqa_score,
            policy=policy_id,
        )
    try:
        embedded_coefficients_sha256()
        coefficients_digest()
    except ValueError as exc:
        return MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason=f"calibration_error:{exc}",
            domain=domain,
            calibrator=calibrator,
            canonical_score_input=canonical_cvqa_score,
            policy=policy_id,
        )
    return MosCalibrationOutcome(
        available=True,
        mos_calibrated_cvqa=detail.estimated_perceptual_quality,
        reason=None,
        domain=domain,
        calibrator=detail.calibrator,
        canonical_score_input=canonical_cvqa_score,
        policy=policy_id,
        score_requested=detail.score_requested,
        score_effective=detail.score_effective,
        ood_below_support=detail.ood_below_support,
        ood_above_support=detail.ood_above_support,
        route_weight_lambda=detail.route_weight_lambda,
        model_key=detail.model_key,
        post_mos_offset=detail.post_mos_offset,
        input_support_clamp=detail.input_support_clamp,
        support_extrapolation=detail.support_extrapolation,
        route_sigmoid_a=detail.route_sigmoid_a,
        route_sigmoid_b=detail.route_sigmoid_b,
        route_correction_d=detail.route_correction_d,
    )


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
    )


def mos_calibration_record(
    outcome: MosCalibrationOutcome,
    *,
    cvqa_upstream: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_id = outcome.policy or MOS_POLICY
    pspec = policy_spec(policy_id)
    record: dict[str, Any] = {
        "status": "available" if outcome.available else "unavailable",
        "research_status": RESEARCH_STATUS,
        "mos_policy": MOS_POLICY,
        "calibration_version": pspec["calibration_version"],
        "calibration_label": pspec["calibration_label"],
        "weight_kind": pspec["weight_kind"],
        "hdr_weight_lambda": pspec["hdr_weight_lambda"],
        "sdr_weight_lambda": pspec["sdr_weight_lambda"],
        "coefficient_cohort": pspec["coefficient_cohort"],
        "coefficient_source": pspec["coefficient_source"],
        "coefficient_source_json_sha256": pspec["coefficient_source_json_sha256"],
        "coefficients_digest_sha256": coefficients_digest(),
        "m1_implementation": "two_param_logistic_route_correction",
        "formula": pspec["formula"],
        "canonical_score_scale": "0-10",
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
            "Single selected calibration: no bilateral input clamp; scores outside "
            f"historical support ({_SUPPORT['score_min']}, {_SUPPORT['score_max']}) "
            "are extrapolated with OOD warnings only. Final output bounded to [1, 5]."
        ),
        "manifest_id": pspec["manifest_id"],
        "coefficients_source_bundle_sha256": pspec["coefficients_bundle_sha256"],
        "coefficients_digest_note": (
            "coefficients_digest_sha256 is a composite digest (frozen S1 bundle, source "
            "JSON SHA-256, post-output offsets, selection lock SHA-256s). "
            "embedded_coefficients_sha256 is recomputed from the effective route "
            "parameters embedded in this module."
        ),
        "embedded_coefficients_sha256": embedded_coefficients_sha256(),
        "post_output_offset_by_transfer": dict(pspec["post_output_offset_by_transfer"]),
        "sdr_selection_lock_sha256": pspec["sdr_selection_lock_sha256"],
        "hdr_selection_lock_sha256": pspec["hdr_selection_lock_sha256"],
        "obsolete_env_note": (
            f"{ENV_MOS_POLICY} is obsolete and ignored when unset or set to "
            f"{MOS_POLICY!r}; other non-empty values fail closed."
        ),
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
        record["ood_above_support"] = outcome.ood_above_support
        record["input_support_clamp"] = False
        if outcome.support_extrapolation:
            record["support_extrapolation"] = True
    if outcome.calibrator is not None:
        record["calibrator"] = outcome.calibrator
        spec = _CALIBRATORS[outcome.calibrator]
        record["research_arm"] = spec["research_arm"]
        record["route_weight_lambda"] = outcome.route_weight_lambda or spec["weight_lambda"]
        record["model_key"] = outcome.model_key or spec["model_key"]
        record["selected_ridge"] = spec["selected_ridge"]
        record["normalization"] = dict(spec["normalization"])
        record["support"] = dict(spec["support"])
        if outcome.route_sigmoid_a is not None:
            record["route_sigmoid"] = {
                "a": outcome.route_sigmoid_a,
                "b": outcome.route_sigmoid_b,
                "d": outcome.route_correction_d,
            }
        height = outcome.domain.encoded_height if outcome.domain else 1080
        rp = spec.get(f"route_params_{height}")
        if rp:
            record["route_params"] = dict(rp)
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
        f"{OUTPUT_SCALE_LABEL} is a provisional research mapping from canonical 0–10 "
        f"CVQA-HDR-NR scores using the scientifically selected formula; it has not been "
        "validated for production subjective-quality prediction."
    )
    return record