"""Transfer-separated CVQA-to-estimated-MOS calibration (SDR 200 / PQ 192).

The canonical CompressedVQA clip score remains unchanged. One selected M1 sigmoid
plus M7 metadata correction is applied automatically when validated ffprobe transfer
and encoded height are supported. Both route fits use only same-transfer observations.
A source-family-balanced SDR native-MOS post-offset is applied after the M7 stage.

The frozen v7 research source, retained historical v5 fixture and independent new
oracle are documented in docs/dev/CVQA_MOS_TRANSFER_SEPARATED_V7.md.
A low-score C2 tail adapter is separately documented in
docs/dev/CVQA_MOS_V7_LOCAL_TAIL_CONVEX.md; original fitted coefficients remain frozen.
This is retrospective research calibration, not a measured or certified MOS.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from pixvqa_cvqahdr import PINNED_CHECKPOINT_SHA256, PINNED_COMMIT, PINNED_PROFILE_SHA256
from pixvqa_video.ffmpeg_bins import resolve_ffprobe_legacy
from pixvqa_cvqahdr import mos_calibration_a2, mos_low_tail

# Separate algorithm selector (default frozen v7+C2). Do not overload MOS_POLICY for this.
ENV_MOS_CALIBRATION_ALGORITHM = "PIXVQA_MOS_CALIBRATION_ALGORITHM"
MOS_CALIBRATION_ALGORITHM_V7 = "v7_convex_c2"
DEFAULT_MOS_CALIBRATION_ALGORITHM = MOS_CALIBRATION_ALGORITHM_V7

# Obsolete: ignored when unset or when set to MOS_POLICY; any other non-empty value fails closed.
ENV_MOS_POLICY = "PIXVQA_CVQA_MOS_POLICY"
MOS_POLICY = "selected_s1_sdr_p1_hdr_c0"
DEFAULT_MOS_POLICY = MOS_POLICY
SELECTED_MOS_POLICY = MOS_POLICY

SELECTED_FITTED_MODELS_SHA256 = "9cdf829254e79b39ce7bd93502f5550be7ed3c3c19ac9985559ec8430f3db855"
SDR_SELECTION_LOCK_SHA256 = "f65a19498735f0ce17257079a913f2ece25e9692b56c8392e27ce7d29ae079ee"
HDR_SELECTION_LOCK_SHA256 = "7882d79f68f62fc06e034c3ed9f518cbdd0940f957b83df344b46858273b3f8c"
SELECTED_COEFFICIENTS_SOURCE_JSON_SHA256 = "e817ed479079c0e208e8ec0d84aad3b5836606eeb329585d27e2c461c24b8e73"
OUTPUT_SCALE_LABEL = "estimated perceptual quality 1-5"

CALIBRATION_VERSION = "selected-s1-sdr200-pq192-v7-convex-c2-20261119-v6"
CALIBRATION_LABEL = "Frozen v7 SDR/PQ M1+M7 plus local C2 low-tail correction"
WEIGHT_KIND = "transfer_isolated_source_study_balanced_baseline"
RESEARCH_STATUS = "provisional"
COEFFICIENT_COHORT = "SDR200_PQ192_separate"
COEFFICIENT_SOURCE = (
    "perceptual_optimization/calibration/production_readiness_20261113/"
    "CVQA_TO_MOS_TRANSFER_SEPARATED_DELIVERABLE_20261116/data/"
    "PIXVQA_CVQA_to_MOS_Selected_Coefficients_v7_Research.json"
)
MANIFEST_ID = "pixvqa_calib_s1_sdr200_pq192_20261119_convex_c2_v6"
HDR_WEIGHT_LAMBDA = 0.0
SDR_WEIGHT_LAMBDA = 0.0
SDR_POST_MOS_OFFSET = 0.5107732905216783
PQ_GAMMA_1080 = -0.131036744556774
GAMMA_2160_SDR = 0.08543516806280937
GAMMA_2160_PQ = 0.2289003839930633

EVIDENCE_CAVEAT = (
    "Research v7: SDR model trained on 200 SDR rows only (30 recorded sources); "
    "PQ model trained on 192 PQ rows only (35 recorded sources). M1 sigmoid with "
    "original M7 correction; SDR native-MOS offset fitted on five SDR54 families. "
    "Retrospective AVT comparisons are not blind. HDR35 MAE 0.6042 regresses "
    "against the historic pooled model, and high-native-MOS HDR is unvalidated. "
    "Input CVQA never support-clamped; per-transfer OOD flags warn on extrapolation. "
    "Research-only convex C2 endpoint correction sets MOS(0)=1; SDR 4.134 anchor "
    "is lowered 0.08 MOS with nondecreasing slope and returns to frozen v7 at CVQA 5.65. HDR local C2 "
    "joins at CVQA 6.26845. Native low-score MOS remains unvalidated. "
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

# The transfer-only cohorts have different fitted score distributions.
_NORMALIZATION_BY_TRANSFER = {
    "sdr": {"mean": 8.010332079849347, "std": 1.241145006178454},
    "pq": {"mean": 8.769753592063894, "std": 0.9333470039280067},
}
_SUPPORT_BY_TRANSFER = {
    "sdr": {"score_min": 4.1344840888613135, "score_max": 10.038337481465987},
    "pq": {"score_min": 6.268450107554486, "score_max": 10.051497077192835},
}
# Provenance-only union for older records; prediction OOD uses per-transfer ranges.
_SUPPORT = {
    "score_min": min(v["score_min"] for v in _SUPPORT_BY_TRANSFER.values()),
    "score_max": max(v["score_max"] for v in _SUPPORT_BY_TRANSFER.values()),
}

# Effective 2-degree-of-freedom coefficients from original frozen four-tuple.
# Runtime uses these embedded values; independent tests use the frozen tuple.
_ROUTE_PARAMS: dict[tuple[str, int], dict[str, float]] = {
    ("sdr", 1080): {'a': -0.49117628849094824, 'b': 1.0977838841771388, 'd': 0.5107732905216783},
    ("sdr", 2160): {'a': -0.49117628849094824, 'b': 1.0977838841771388, 'd': 0.5962084585844877},
    ("pq", 1080): {'a': 0.07227163628268213, 'b': 0.8293304839931255, 'd': -0.131036744556774},
    ("pq", 2160): {'a': 0.07227163628268213, 'b': 0.8293304839931255, 'd': 0.09786363943628931},
}

# Stable calibrator identifiers and frozen training metadata (provenance only).
_CALIBRATORS: dict[str, dict[str, Any]] = {
    "U0_M7_Bias": {
        "route": "pq",
        "research_arm": "Original_U",
        "weight_kind": "baseline",
        "weight_lambda": 0.0,
        "model_key": "PQ192|Original_U|baseline|0.0|ridge0.1",
        "selected_ridge": 0.1,
        "normalization": dict(_NORMALIZATION_BY_TRANSFER["pq"]),
        "support": dict(_SUPPORT_BY_TRANSFER["pq"]),
        "route_params_1080": dict(_ROUTE_PARAMS[("pq", 1080)]),
        "route_params_2160": dict(_ROUTE_PARAMS[("pq", 2160)]),
    },
    "U_v2_M7_Bias": {
        "route": "sdr",
        "research_arm": "U_v2",
        "weight_kind": "baseline",
        "weight_lambda": 0.0,
        "model_key": "SDR200|U_v2|baseline|0.0|ridge0.1",
        "selected_ridge": 0.1,
        "normalization": dict(_NORMALIZATION_BY_TRANSFER["sdr"]),
        "support": dict(_SUPPORT_BY_TRANSFER["sdr"]),
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
    "coefficients_bundle_sha256": SELECTED_FITTED_MODELS_SHA256,
    "post_output_offset_by_transfer": {"pq": 0.0, "sdr": SDR_POST_MOS_OFFSET},
    "sdr_selection_lock_sha256": SDR_SELECTION_LOCK_SHA256,
    "hdr_selection_lock_sha256": HDR_SELECTION_LOCK_SHA256,
    "evidence_caveat": EVIDENCE_CAVEAT,
    "calibrators": _CALIBRATORS,
    "formula": "frozen_v7_M1_M7_plus_local_C2_tail_below_transfer_join",
    "low_tail": mos_low_tail.low_tail_payload(),
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


_VALID_ALGORITHM_ALIASES: dict[str, str] = {
    "v7_convex_c2": MOS_CALIBRATION_ALGORITHM_V7,
    "v7": MOS_CALIBRATION_ALGORITHM_V7,
    "selected_s1_sdr_p1_hdr_c0": MOS_CALIBRATION_ALGORITHM_V7,
    mos_calibration_a2.MOS_CALIBRATION_ALGORITHM: mos_calibration_a2.MOS_CALIBRATION_ALGORITHM,
    "a2_tv123": mos_calibration_a2.MOS_CALIBRATION_ALGORITHM,
    "a2": mos_calibration_a2.MOS_CALIBRATION_ALGORITHM,
}


def resolve_mos_calibration_algorithm(explicit: str | None = None) -> tuple[str | None, str | None]:
    """Return ``(algorithm_id, error_reason)``. Default is frozen v7+C2."""
    if explicit is not None:
        key = str(explicit).strip().lower()
        resolved = _VALID_ALGORITHM_ALIASES.get(key)
        if resolved is None:
            return None, f"invalid_mos_calibration_algorithm:{explicit}"
        return resolved, None
    env_raw = os.environ.get(ENV_MOS_CALIBRATION_ALGORITHM, "").strip()
    if not env_raw:
        return DEFAULT_MOS_CALIBRATION_ALGORITHM, None
    key = env_raw.lower()
    resolved = _VALID_ALGORITHM_ALIASES.get(key)
    if resolved is None:
        return None, f"invalid_mos_calibration_algorithm:{env_raw}"
    return resolved, None


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
    """Return a snapshot; callers must not be able to mutate the frozen policy."""
    pid = policy_id or MOS_POLICY
    if pid != MOS_POLICY:
        raise ValueError(f"unknown MOS policy {pid!r}")
    return copy.deepcopy(_POLICY_SPEC)


def calibrators_for_policy(policy_id: str | None = None) -> dict[str, dict[str, Any]]:
    """Return independent calibrator metadata, never writable runtime state."""
    policy_spec(policy_id)
    return copy.deepcopy(_CALIBRATORS)


def input_support_clamp_enabled(policy_id: str | None = None) -> bool:
    policy_spec(policy_id)
    return False


def _coefficients_digest_payload() -> dict[str, Any]:
    return {
        "policy_id": MOS_POLICY,
        "selected_fitted_models_sha256": SELECTED_FITTED_MODELS_SHA256,
        "coefficient_source_json_sha256": SELECTED_COEFFICIENTS_SOURCE_JSON_SHA256,
        "post_output_offset_by_transfer": _POLICY_SPEC["post_output_offset_by_transfer"],
        "normalization_by_transfer": _NORMALIZATION_BY_TRANSFER,
        "support_by_transfer": _SUPPORT_BY_TRANSFER,
        "route_params": {f"{t}_{h}": dict(v) for (t, h), v in sorted(_ROUTE_PARAMS.items())},
        "sdr_selection_lock_sha256": SDR_SELECTION_LOCK_SHA256,
        "hdr_selection_lock_sha256": HDR_SELECTION_LOCK_SHA256,
        "input_support_clamp": False,
        "low_tail": mos_low_tail.low_tail_payload(),
    }


SELECTED_COEFFICIENTS_DIGEST_SHA256 = "3a1a130e776fb470eed89849717db37c9e4d9c038bc87706da56f957f19b7407"


def _embedded_coefficients_payload() -> dict[str, Any]:
    return {
        "policy_id": MOS_POLICY,
        "formula": _POLICY_SPEC["formula"],
        "normalization_by_transfer": {k: dict(v) for k, v in _NORMALIZATION_BY_TRANSFER.items()},
        "support_by_transfer": {k: dict(v) for k, v in _SUPPORT_BY_TRANSFER.items()},
        "routes": {
            f"{transfer}_{height}": dict(params)
            for (transfer, height), params in sorted(_ROUTE_PARAMS.items())
        },
        "selected_fitted_models_sha256": SELECTED_FITTED_MODELS_SHA256,
        "coefficient_source_json_sha256": SELECTED_COEFFICIENTS_SOURCE_JSON_SHA256,
        "sdr_selection_lock_sha256": SDR_SELECTION_LOCK_SHA256,
        "hdr_selection_lock_sha256": HDR_SELECTION_LOCK_SHA256,
        "post_output_offset_by_transfer": dict(_POLICY_SPEC["post_output_offset_by_transfer"]),
        "input_support_clamp": False,
        "low_tail": mos_low_tail.low_tail_payload(),
    }


EMBEDDED_COEFFICIENTS_SHA256 = "e8d5a971b86190b398a35083f8deff8a1843d1edb94434d142799d643bf384bc"


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
    mean = _NORMALIZATION_BY_TRANSFER[transfer]["mean"]
    std = _NORMALIZATION_BY_TRANSFER[transfer]["std"]
    z = (score - mean) / std
    logit = params["a"] + params["b"] * z
    frozen_v7_mos = _clamp(1.0 + 4.0 * _sigmoid(logit) + params["d"], 1.0, 5.0)
    return mos_low_tail.correct_low_tail(
        score, transfer=transfer, encoded_height=encoded_height,
        frozen_v7_mos=frozen_v7_mos,
    )


def predict_mos(
    canonical_cvqa_score: float,
    *,
    transfer: str,
    encoded_height: int,
    calibrator: str | None = None,
    policy: str | None = None,
    algorithm: str | None = None,
) -> float:
    """Map canonical CVQA 0–10 to estimated perceptual quality 1–5."""
    return predict_mos_detailed(
        canonical_cvqa_score,
        transfer=transfer,
        encoded_height=encoded_height,
        calibrator=calibrator,
        policy=policy,
        algorithm=algorithm,
    ).estimated_perceptual_quality


def predict_mos_detailed(
    canonical_cvqa_score: float,
    *,
    transfer: str,
    encoded_height: int,
    calibrator: str | None = None,
    policy: str | None = None,
    algorithm: str | None = None,
) -> "PredictMosResult":
    algorithm_id, algorithm_err = resolve_mos_calibration_algorithm(algorithm)
    if algorithm_err:
        raise ValueError(algorithm_err)
    if algorithm_id == mos_calibration_a2.MOS_CALIBRATION_ALGORITHM:
        detail = mos_calibration_a2.predict_mos_detailed(
            canonical_cvqa_score,
            transfer=transfer,
            encoded_height=encoded_height,
            calibrator=calibrator,
        )
        return PredictMosResult(
            estimated_perceptual_quality=detail.estimated_perceptual_quality,
            score_requested=detail.score_requested,
            score_effective=detail.score_effective,
            ood_below_support=detail.ood_below_support,
            ood_above_support=detail.ood_above_support,
            policy=detail.policy,
            calibrator=detail.calibrator,
            route_weight_lambda=detail.route_weight_lambda,
            model_key=detail.model_key,
            post_mos_offset=detail.post_mos_offset,
            input_support_clamp=detail.input_support_clamp,
            support_extrapolation=detail.support_extrapolation,
            route_sigmoid_a=detail.route_sigmoid_a,
            route_sigmoid_b=detail.route_sigmoid_b,
            route_correction_d=detail.route_correction_d,
            calibration_algorithm=algorithm_id,
        )
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
    # Standalone prediction must have the same integrity guarantees as the
    # adapter path. Previously only apply_mos_calibration() checked digests,
    # so direct callers could silently receive results from modified coefficients.
    embedded_coefficients_sha256()
    coefficients_digest()
    if calibrator is not None:
        key = calibrator
        _assert_calibrator_matches_transfer(key, transfer)
    else:
        key = calibrator_for_transfer(transfer)
        if key is None:
            raise ValueError(f"unsupported transfer {transfer!r}")
    spec = _CALIBRATORS[key]
    lo = float(_SUPPORT_BY_TRANSFER[transfer]["score_min"])
    hi = float(_SUPPORT_BY_TRANSFER[transfer]["score_max"])
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
        calibration_algorithm=algorithm_id,
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
    calibration_algorithm: str = DEFAULT_MOS_CALIBRATION_ALGORITHM


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
    calibration_algorithm: str | None = None


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
    algorithm: str | None = None,
) -> dict[str, Any]:
    algorithm_id, algorithm_err = resolve_mos_calibration_algorithm(algorithm)
    effective_algorithm = algorithm_id if algorithm_err is None else (algorithm or DEFAULT_MOS_CALIBRATION_ALGORITHM)
    record = {
        "status": "unavailable",
        "research_status": RESEARCH_STATUS,
        "mos_calibration_algorithm": effective_algorithm,
        "mos_policy": policy or (
            mos_calibration_a2.MOS_CALIBRATION_ALGORITHM
            if effective_algorithm == mos_calibration_a2.MOS_CALIBRATION_ALGORITHM
            else MOS_POLICY
        ),
        "canonical_score_input": _json_safe_number(canonical_score_input),
        "output_scale": OUTPUT_SCALE_LABEL,
        "reason": reason,
        "note": f"MOS mapping suppressed ({reason}); canonical CVQA score unchanged.",
    }
    if algorithm_err:
        record["reason"] = algorithm_err
    return record


def outcome_calibration_algorithm(outcome: MosCalibrationOutcome) -> str:
    """Resolve which calibration record builder to use (defensive for partial outcomes)."""
    alg = outcome.calibration_algorithm
    if alg in {MOS_CALIBRATION_ALGORITHM_V7, mos_calibration_a2.MOS_CALIBRATION_ALGORITHM}:
        return alg
    if outcome.policy == mos_calibration_a2.MOS_CALIBRATION_ALGORITHM:
        return mos_calibration_a2.MOS_CALIBRATION_ALGORITHM
    if outcome.policy == MOS_POLICY:
        return MOS_CALIBRATION_ALGORITHM_V7
    return alg or MOS_CALIBRATION_ALGORITHM_V7


def apply_mos_calibration(
    canonical_cvqa_score: float,
    path: str,
    *,
    policy: str | None = None,
    algorithm: str | None = None,
) -> MosCalibrationOutcome:
    """Attempt 1–5 estimate mapping; never mutates the canonical CVQA score."""
    algorithm_id, algorithm_err = resolve_mos_calibration_algorithm(algorithm)
    if algorithm_err:
        return MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason=algorithm_err,
            domain=None,
            calibrator=None,
            canonical_score_input=canonical_cvqa_score,
            policy=None,
            calibration_algorithm=None,
        )
    if algorithm_id == mos_calibration_a2.MOS_CALIBRATION_ALGORITHM:
        policy_id = mos_calibration_a2.MOS_CALIBRATION_ALGORITHM
    else:
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
                calibration_algorithm=algorithm_id,
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
            calibration_algorithm=algorithm_id,
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
            calibration_algorithm=algorithm_id,
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
            calibration_algorithm=algorithm_id,
        )
    if algorithm_id == mos_calibration_a2.MOS_CALIBRATION_ALGORITHM:
        try:
            calibrator = mos_calibration_a2.calibrator_for_transfer(domain.transfer)
        except ValueError:
            calibrator = None
    else:
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
            calibration_algorithm=algorithm_id,
        )
    try:
        detail = predict_mos_detailed(
            canonical_cvqa_score,
            transfer=domain.transfer,
            encoded_height=domain.encoded_height,
            calibrator=calibrator,
            policy=policy_id if algorithm_id != mos_calibration_a2.MOS_CALIBRATION_ALGORITHM else None,
            algorithm=algorithm_id,
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
            calibration_algorithm=algorithm_id,
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
        calibration_algorithm=algorithm_id,
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
        calibration_algorithm=outcome_calibration_algorithm(outcome),
    )


def mos_calibration_record(
    outcome: MosCalibrationOutcome,
    *,
    cvqa_upstream: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if outcome_calibration_algorithm(outcome) == mos_calibration_a2.MOS_CALIBRATION_ALGORITHM:
        return mos_calibration_a2.mos_calibration_record(outcome, cvqa_upstream=cvqa_upstream)
    policy_id = outcome.policy or MOS_POLICY
    pspec = policy_spec(policy_id)
    record: dict[str, Any] = {
        "status": "available" if outcome.available else "unavailable",
        "research_status": RESEARCH_STATUS,
        "mos_calibration_algorithm": MOS_CALIBRATION_ALGORITHM_V7,
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
        "m1_implementation": "two_param_logistic_route_correction_plus_local_c2_tail",
        "low_tail": pspec["low_tail"],
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
            "The input CVQA score is not support-clamped. SDR support: "
            f"{_SUPPORT_BY_TRANSFER['sdr']}; HDR PQ support: "
            f"{_SUPPORT_BY_TRANSFER['pq']}. Out-of-support predictions are "
            "extrapolations; the estimated MOS output is bounded to [1, 5]."
        ),
        "manifest_id": pspec["manifest_id"],
        "coefficients_source_bundle_sha256": pspec["coefficients_bundle_sha256"],
        "coefficients_digest_note": (
            "coefficients_digest_sha256 pins fitted-model SHA-256, source JSON hash, "
            "native-MOS offsets, low-tail C2 control points and transfer-specific "
            "fitting provenance. embedded_coefficients_sha256 is recomputed from "
            "the complete effective mapping, including the endpoint adapter."
        ),
        "embedded_coefficients_sha256": embedded_coefficients_sha256(),
        "post_output_offset_by_transfer": dict(pspec["post_output_offset_by_transfer"]),
        "sdr_selection_lock_sha256": pspec["sdr_selection_lock_sha256"],
        "hdr_selection_lock_sha256": pspec["hdr_selection_lock_sha256"],
        "obsolete_env_note": (
            f"{ENV_MOS_POLICY} is obsolete and ignored when unset or set to "
            f"{MOS_POLICY!r}; other non-empty values fail closed."
        ),
        "algorithm_selector_env": ENV_MOS_CALIBRATION_ALGORITHM,
        "algorithm_selector_note": (
            f"Default algorithm is {MOS_CALIBRATION_ALGORITHM_V7!r}. Experimental "
            f"{mos_calibration_a2.MOS_CALIBRATION_ALGORITHM!r} is opt-in via "
            f"{ENV_MOS_CALIBRATION_ALGORITHM} or CLI --mos-calibration-algorithm."
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