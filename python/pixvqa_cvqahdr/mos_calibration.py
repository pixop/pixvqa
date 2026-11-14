"""Estimated perceptual quality 1–5 calibration for CVQA-HDR-NR canonical 0–10 scores.

Three versioned policies are available:

* ``legacy_target_lambda1_both`` (**default**) — full-392 cohort, target-MOS
  reweighting λ=1.0 on **both** HDR PQ and SDR routes (user-selected provisional
  policy, 2026-11-14).
* ``reviewed_s1_hdr0_sdr1`` (**opt-in**) — independently reviewed S1 F1-392 pair:
  HDR PQ ``Original_U`` baseline λ=0, SDR ``U_v2`` target λ=1.0. Serialized from
  sealed ``s1_operational_20261114`` artifacts; HDR coefficients differ, SDR is
  identical to the legacy SDR route.
* ``selected_s1_sdr_p1_hdr_c0`` (**opt-in**) — user-selected research composite:
  frozen S1 reviewed M1+M7 on both routes, then SDR-only native-MOS offset
  ``+0.4105017796126768`` with final ``[1, 5]`` clamp; HDR PQ has no post-offset.
  **Unbounded input extrapolation** beyond historical training support (no bilateral
  input clamp); support bounds are observational only. **Shadow/research only —
  not production default.**

Select via ``PIXVQA_CVQA_MOS_POLICY`` or the ``policy`` argument on public APIs.
Invalid policy identifiers fail closed (MOS suppressed; canonical CVQA unchanged).

Output is **estimated perceptual quality on a 1–5 scale**, not measured subjective MOS.
U-domain target remapping is training-only and is not applied at inference.

Legacy and reviewed policies retain bilateral training-support input clamping
``[4.1344840888613135, 10.051497077192835]`` with final ``[1, 5]`` output bounding.
Selected policy allows monotonic extrapolation outside observed support; OOD flags
warn without capping. Monotonicity on a dense sweep does not prove MOS generalization.
Native HDR35 contains one CVQA score 10.07199 above historical support max.
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

ENV_MOS_POLICY = "PIXVQA_CVQA_MOS_POLICY"
DEFAULT_MOS_POLICY = "legacy_target_lambda1_both"
REVIEWED_MOS_POLICY = "reviewed_s1_hdr0_sdr1"
SELECTED_MOS_POLICY = "selected_s1_sdr_p1_hdr_c0"
VALID_MOS_POLICIES = frozenset({DEFAULT_MOS_POLICY, REVIEWED_MOS_POLICY, SELECTED_MOS_POLICY})
SELECTED_SDR_POST_MOS_OFFSET = 0.4105017796126768
SELECTED_HDR_POST_MOS_OFFSET = 0.0
FROZEN_S1_COEFFICIENTS_BUNDLE_SHA256 = (
    "67d39be487fccee6b0a08ef33a4dd7e8adf1745578eff9060ab48e394069a02c"
)
SDR_SELECTION_LOCK_SHA256 = (
    "15dbde98a70c3a4ac7e2f7036a084d589e4154f320ebf4dba657f9bb4c894581"
)
HDR_SELECTION_LOCK_SHA256 = (
    "5a4749f6fae7d3a7a9c6736910d83c02d41e4d7630e09bba4476ebfed0ee68cd"
)
OUTPUT_SCALE_LABEL = "estimated perceptual quality 1-5"

CALIBRATION_VERSION = "target-mos-reweight-lambda1-full392-20261114-v1"
CALIBRATION_LABEL = "domain-aware logistic MOS calibration"
WEIGHT_KIND = "target_mos"
WEIGHT_LAMBDA = 1.0
RESEARCH_STATUS = "provisional"
COEFFICIENT_COHORT = "full-392-20261113"
# Serialized from:
# .../mos_target_reweight_20261113/artifacts/endpoints.csv
# (original_ids, target, lambda_weight=1.0)
COEFFICIENT_SOURCE = (
    "perceptual_optimization/calibration/production_readiness_20261113/"
    "mos_target_reweight_20261113/artifacts/endpoints.csv"
)
EVIDENCE_CAVEAT = (
    "Target-MOS λ=1.0 improves exploratory native AVT error on SDR but worsens HDR "
    "relative to baseline; HDR adoption is a user policy choice, not empirically "
    "superior global accuracy. AVT ratings previously informed U-v2 routing and are "
    "not blind confirmation. NightTraffic alias-merged CV (55 groups) is a research "
    "sensitivity only — stimulus equivalence is not established. Scale-bridge pseudo-pairs "
    "(120, three families) are not verified identical across cohorts."
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

_SHARED_NORMALIZATION = {"mean": 8.382293636852387, "std": 1.1647912353499874}
_SHARED_SUPPORT = {
    "score_min": 4.1344840888613135,
    "score_max": 10.051497077192835,
}
_SHARED_RIDGE = 0.03

# SDR U_v2 target λ=1.0 — identical under legacy and reviewed policies.
_SDR_U_V2_TARGET: dict[str, Any] = {
    "route": "sdr",
    "research_arm": "U_v2",
    "offset": 0.075,
    "slope": 1.05,
    "weight_kind": "target_mos",
    "weight_lambda": 1.0,
    "model_key": "old392|U_v2|target|1.0",
    "selected_ridge": _SHARED_RIDGE,
    "normalization": dict(_SHARED_NORMALIZATION),
    "support": dict(_SHARED_SUPPORT),
    "m1_params": [
        0.056499133595354314,
        1.0284431102294815,
        0.15356247204499882,
        -0.028645113301033025,
    ],
    "gamma_pq": -0.13996576078661158,
    "gamma_2160": 0.16798189863001703,
}

_LEGACY_HDR_TARGET: dict[str, Any] = {
    "route": "pq",
    "research_arm": "Original_U",
    "offset": 0.0,
    "slope": 1.0,
    "weight_kind": "target_mos",
    "weight_lambda": 1.0,
    "model_key": "old392|Original_U|target|1.0",
    "selected_ridge": _SHARED_RIDGE,
    "normalization": dict(_SHARED_NORMALIZATION),
    "support": dict(_SHARED_SUPPORT),
    "m1_params": [
        -0.030839215032598753,
        1.0004416018615387,
        0.21501951333459912,
        -0.0004356605086738879,
    ],
    "gamma_pq": -0.12649612675629415,
    "gamma_2160": 0.15454554681584692,
}

_REVIEWED_HDR_BASELINE: dict[str, Any] = {
    "route": "pq",
    "research_arm": "Original_U",
    "offset": 0.0,
    "slope": 1.0,
    "weight_kind": "baseline",
    "weight_lambda": 0.0,
    "model_key": "old392|Original_U|baseline|0.0",
    "selected_ridge": _SHARED_RIDGE,
    "normalization": dict(_SHARED_NORMALIZATION),
    "support": dict(_SHARED_SUPPORT),
    "m1_params": [
        -0.11032944576639363,
        0.9870808746883416,
        0.20866534125962313,
        0.013064270126092849,
    ],
    "gamma_pq": -0.10199018084158194,
    "gamma_2160": 0.13694653969694998,
}

MOS_POLICY_SPECS: dict[str, dict[str, Any]] = {
    DEFAULT_MOS_POLICY: {
        "policy_id": DEFAULT_MOS_POLICY,
        "input_support_clamp": True,
        "calibration_version": "target-mos-reweight-lambda1-full392-20261114-v1",
        "calibration_label": "domain-aware logistic MOS calibration",
        "weight_kind": "target_mos",
        "hdr_weight_lambda": 1.0,
        "sdr_weight_lambda": 1.0,
        "coefficient_cohort": "full-392-20261113",
        "coefficient_source": (
            "perceptual_optimization/calibration/production_readiness_20261113/"
            "mos_target_reweight_20261113/artifacts/endpoints.csv"
        ),
        "manifest_id": None,
        "coefficients_bundle_sha256": None,
        "evidence_caveat": EVIDENCE_CAVEAT,
        "calibrators": {
            "U0_M7_Bias": _LEGACY_HDR_TARGET,
            "U_v2_M7_Bias": dict(_SDR_U_V2_TARGET),
        },
    },
    REVIEWED_MOS_POLICY: {
        "policy_id": REVIEWED_MOS_POLICY,
        "input_support_clamp": True,
        "calibration_version": "s1-f1-392-reviewed-hdr0-sdr1-20261114-v1",
        "calibration_label": "S1 F1 reviewed estimated perceptual quality calibration",
        "weight_kind": "route_mixed",
        "hdr_weight_lambda": 0.0,
        "sdr_weight_lambda": 1.0,
        "coefficient_cohort": "old392",
        "coefficient_source": (
            "perceptual_optimization/calibration/production_readiness_20261113/"
            "s1_operational_20261114/manifest/coefficients_frozen_v1.json"
        ),
        "manifest_id": "pixvqa_calib_s1_f1_392_20261114_v1",
        "coefficients_bundle_sha256": FROZEN_S1_COEFFICIENTS_BUNDLE_SHA256,
        "evidence_caveat": (
            "Independent reviewer-selected S1 F1-392 pair (HDR baseline λ=0, SDR target "
            "λ=1). Exploratory native AVT ratings informed routing; not blind confirmation. "
            "HDR adoption is n=8 with two source families; SDR primary n=54 with display/"
            "encode geometry mismatches on 30/54 stimuli. Scale-bridge pseudo-pairs are not "
            "verified stimulus-identical. Not production-subjective-quality certified."
        ),
        "calibrators": {
            "U0_M7_Bias": _REVIEWED_HDR_BASELINE,
            "U_v2_M7_Bias": dict(_SDR_U_V2_TARGET),
        },
    },
    SELECTED_MOS_POLICY: {
        "policy_id": SELECTED_MOS_POLICY,
        "input_support_clamp": False,
        "input_support_policy": "unbounded_extrapolation",
        "calibration_version": "selected-s1-sdr-p1-hdr-c0-unclamped-input-20261114-v2",
        "calibration_label": "Selected S1 composite with native MOS SDR offset (unclamped input)",
        "weight_kind": "route_mixed",
        "hdr_weight_lambda": 0.0,
        "sdr_weight_lambda": 1.0,
        "coefficient_cohort": "old392",
        "coefficient_source": (
            "perceptual_optimization/calibration/production_readiness_20261113/"
            "s1_operational_20261114/manifest/coefficients_frozen_v1.json"
        ),
        "manifest_id": "pixvqa_calib_s1_f1_392_20261114_v1",
        "coefficients_bundle_sha256": FROZEN_S1_COEFFICIENTS_BUNDLE_SHA256,
        "post_output_offset_by_transfer": {
            "pq": SELECTED_HDR_POST_MOS_OFFSET,
            "sdr": SELECTED_SDR_POST_MOS_OFFSET,
        },
        "sdr_selection_lock_sha256": SDR_SELECTION_LOCK_SHA256,
        "hdr_selection_lock_sha256": HDR_SELECTION_LOCK_SHA256,
        "selection_status": "shadow-not-prod",
        "evidence_caveat": (
            "User-selected research composite: frozen S1 reviewed M1+M7, then SDR-only "
            "additive native-MOS offset +0.4105017796126768 (full-fit deployed delta; not "
            "a single fixed offset validated on 54 held-out source-LOO rows). HDR C0 is "
            "exact reviewed S1 baseline with no post-offset. Input scores are NOT clamped "
            "to historical training support [4.1344840888613135, 10.051497077192835]; "
            "OOD flags warn only. CVQA may exceed 10 (HDR35 has score 10.07199 > support "
            "max). Extrapolation uncertainty accepted; monotonic sweep does not prove "
            "MOS generalization. AVT informed U-v2 routing; retrospective metrics only. "
            "VD6 full-fit offset MAE .4096→.3192 RMSE .5167→.4135. HDR35 C0 bias +0.343 "
            "MAE .555 RMSE .757. Not production certified."
        ),
        "calibrators": {
            "U0_M7_Bias": _REVIEWED_HDR_BASELINE,
            "U_v2_M7_Bias": dict(_SDR_U_V2_TARGET),
        },
    },
}

# Backward-compatible alias: legacy default calibrators (tests and old imports).
_CALIBRATORS: dict[str, dict[str, Any]] = MOS_POLICY_SPECS[DEFAULT_MOS_POLICY]["calibrators"]

_SUPPORTED_HEIGHTS = frozenset({1080, 2160})

# PQ is SMPTE ST 2084 only; ``bt2020`` names primaries or unrelated transfers, not PQ.
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


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _m1_base_mos(z: float, m1_params: list[float] | tuple[float, ...]) -> float:
    """M1 backbone: 1 + 4·σ(a + max(0,b)·(z−c) / max(1e−3, exp(log_scale)))."""
    a, b, c, log_scale = (float(x) for x in m1_params)
    b_eff = max(0.0, b)
    scale = max(1e-3, math.exp(log_scale))
    return 1.0 + 4.0 * _sigmoid(a + b_eff * (z - c) / scale)


def resolve_mos_policy(explicit: str | None = None) -> tuple[str | None, str | None]:
    """Return ``(policy_id, error_reason)``. Unknown policies fail closed."""
    raw = explicit if explicit is not None else os.environ.get(ENV_MOS_POLICY, DEFAULT_MOS_POLICY)
    policy_id = str(raw).strip()
    if policy_id not in VALID_MOS_POLICIES:
        return None, f"invalid_policy:{policy_id}"
    return policy_id, None


def policy_spec(policy_id: str) -> dict[str, Any]:
    try:
        return MOS_POLICY_SPECS[policy_id]
    except KeyError as exc:
        raise ValueError(f"unknown MOS policy {policy_id!r}") from exc


def calibrators_for_policy(policy_id: str) -> dict[str, dict[str, Any]]:
    return policy_spec(policy_id)["calibrators"]


def input_support_clamp_enabled(policy_id: str) -> bool:
    """Return whether canonical CVQA score is clamped to historical training support."""
    return bool(policy_spec(policy_id).get("input_support_clamp", True))


_LEGACY_COEFFICIENTS_DIGEST = (
    "3a92092d1de1e6e2ea413be0cb067c203945019b86e5832036105a3e2d0bb527"
)


def _legacy_coefficients_payload() -> dict[str, Any]:
    """Original digest payload (pre dual-policy); preserved for regression stability."""
    calibrators = MOS_POLICY_SPECS[DEFAULT_MOS_POLICY]["calibrators"]
    return {
        "weight_kind": WEIGHT_KIND,
        "weight_lambda": WEIGHT_LAMBDA,
        "calibrators": {
            key: {
                "m1_params": spec["m1_params"],
                "gamma_pq": spec["gamma_pq"],
                "gamma_2160": spec["gamma_2160"],
                "normalization": spec["normalization"],
                "support": spec["support"],
                "selected_ridge": spec["selected_ridge"],
            }
            for key, spec in sorted(calibrators.items())
        },
    }


def _selected_coefficients_digest_payload() -> dict[str, Any]:
    pspec = policy_spec(SELECTED_MOS_POLICY)
    return {
        "policy_id": SELECTED_MOS_POLICY,
        "s1_coefficients_bundle_sha256": pspec["coefficients_bundle_sha256"],
        "post_output_offset_by_transfer": pspec["post_output_offset_by_transfer"],
        "sdr_selection_lock_sha256": pspec["sdr_selection_lock_sha256"],
        "hdr_selection_lock_sha256": pspec["hdr_selection_lock_sha256"],
        "input_support_clamp": pspec["input_support_clamp"],
    }


SELECTED_COEFFICIENTS_DIGEST_SHA256 = (
    "463b641347d68585ea9f99fc6d8c58b3ea274896458f6b1e58cbf58592b43cd3"
)


def _enforce_digest_match(computed: str, expected: str, *, label: str) -> str:
    """Verify a digest under normal and ``python -O`` execution (asserts are disabled under -O)."""
    if computed != expected:
        raise ValueError(f"{label} digest mismatch: expected {expected}, got {computed}")
    return computed


def coefficients_digest(policy_id: str | None = None) -> str:
    """SHA256 digest for the active policy (legacy: runtime recompute; reviewed: source bundle)."""
    policy = policy_id or DEFAULT_MOS_POLICY
    if policy == DEFAULT_MOS_POLICY:
        payload = _legacy_coefficients_payload()
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return _enforce_digest_match(digest, _LEGACY_COEFFICIENTS_DIGEST, label="legacy_coefficients")
    if policy == SELECTED_MOS_POLICY:
        payload = _selected_coefficients_digest_payload()
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return _enforce_digest_match(
            digest, SELECTED_COEFFICIENTS_DIGEST_SHA256, label="selected_coefficients"
        )
    bundled = policy_spec(policy).get("coefficients_bundle_sha256")
    if bundled:
        return str(bundled)
    raise ValueError(f"no digest registered for policy {policy!r}")


def _embedded_coefficients_payload(policy_id: str) -> dict[str, Any]:
    spec = policy_spec(policy_id)
    calibrators = spec["calibrators"]
    payload: dict[str, Any] = {
        "policy_id": policy_id,
        "hdr_weight_lambda": spec["hdr_weight_lambda"],
        "sdr_weight_lambda": spec["sdr_weight_lambda"],
        "calibrators": {
            key: {
                "m1_params": cal["m1_params"],
                "gamma_pq": cal["gamma_pq"],
                "gamma_2160": cal["gamma_2160"],
                "weight_lambda": cal["weight_lambda"],
                "model_key": cal["model_key"],
                "normalization": cal["normalization"],
                "support": cal["support"],
                "selected_ridge": cal["selected_ridge"],
            }
            for key, cal in sorted(calibrators.items())
        },
    }
    if policy_id == SELECTED_MOS_POLICY:
        payload["post_output_offset_by_transfer"] = spec["post_output_offset_by_transfer"]
        payload["s1_coefficients_bundle_sha256"] = spec["coefficients_bundle_sha256"]
        payload["sdr_selection_lock_sha256"] = spec["sdr_selection_lock_sha256"]
        payload["hdr_selection_lock_sha256"] = spec["hdr_selection_lock_sha256"]
        payload["input_support_clamp"] = spec["input_support_clamp"]
    return payload


REVIEWED_EMBEDDED_COEFFICIENTS_SHA256 = (
    "037d22badb5b207609a485b37c7cae9c694bd7a30a473678a482a19b1d564022"
)
SELECTED_EMBEDDED_COEFFICIENTS_SHA256 = (
    "b855164620e66b208826cc68d516fb89297f240ff3c4f019288790d3b1d39916"
)


def post_output_offset_for_transfer(policy_id: str, transfer: str) -> float:
    """Route-specific additive MOS offset applied after M7 clamp (selected policy only)."""
    offsets = policy_spec(policy_id).get("post_output_offset_by_transfer")
    if offsets is None:
        return 0.0
    return float(offsets.get(transfer, 0.0))


def embedded_coefficients_sha256(policy_id: str) -> str:
    """Runtime digest of coefficients actually embedded in this module for *policy_id*."""
    payload = _embedded_coefficients_payload(policy_id)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if policy_id == REVIEWED_MOS_POLICY:
        return _enforce_digest_match(
            digest, REVIEWED_EMBEDDED_COEFFICIENTS_SHA256, label="reviewed_embedded_coefficients"
        )
    if policy_id == SELECTED_MOS_POLICY:
        return _enforce_digest_match(
            digest, SELECTED_EMBEDDED_COEFFICIENTS_SHA256, label="selected_embedded_coefficients"
        )
    return digest


def calibrator_for_transfer(transfer: str) -> str | None:
    """Map trusted PQ/SDR metadata to the domain-routed calibrator identifier."""
    if transfer == "pq":
        return "U0_M7_Bias"
    if transfer == "sdr":
        return "U_v2_M7_Bias"
    return None


def _assert_calibrator_matches_transfer(
    calibrator: str,
    transfer: str,
    *,
    policy_id: str,
) -> None:
    calibrators = calibrators_for_policy(policy_id)
    spec = calibrators.get(calibrator)
    if spec is None:
        raise ValueError(f"unknown calibrator {calibrator!r}")
    expected_route = spec["route"]
    if expected_route != transfer:
        raise ValueError(
            f"calibrator {calibrator!r} is trained for {expected_route!r} content, "
            f"not {transfer!r}"
        )


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
    """Like ``predict_mos`` but includes support OOD flags and effective score."""
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
    if calibrator is not None:
        key = calibrator
        _assert_calibrator_matches_transfer(key, transfer, policy_id=policy_id)
    else:
        key = calibrator_for_transfer(transfer)
        if key is None:
            raise ValueError(f"unsupported transfer {transfer!r}")
    calibrators = calibrators_for_policy(policy_id)
    spec = calibrators[key]
    if encoded_height not in _SUPPORTED_HEIGHTS:
        raise ValueError(f"unsupported encoded height {encoded_height}")
    lo = float(spec["support"]["score_min"])
    hi = float(spec["support"]["score_max"])
    mean = float(spec["normalization"]["mean"])
    std = float(spec["normalization"]["std"])
    if std <= 0:
        raise ValueError("calibrator std must be positive")
    ood_below = requested < lo
    ood_above = requested > hi
    clamp_input = input_support_clamp_enabled(policy_id)
    xclip = _clamp(requested, lo, hi) if clamp_input else requested
    z = (xclip - mean) / std
    base = _clamp(_m1_base_mos(z, spec["m1_params"]), 1.0, 5.0)
    correction = 0.0
    if transfer == "pq":
        correction += float(spec["gamma_pq"])
    if encoded_height == 2160:
        correction += float(spec["gamma_2160"])
    estimate = _clamp(base + correction, 1.0, 5.0)
    post_offset = post_output_offset_for_transfer(policy_id, transfer)
    if post_offset != 0.0:
        estimate = _clamp(estimate + post_offset, 1.0, 5.0)
    return PredictMosResult(
        estimated_perceptual_quality=estimate,
        score_requested=requested,
        score_effective=xclip,
        ood_below_support=ood_below,
        ood_above_support=ood_above,
        policy=policy_id,
        calibrator=key,
        route_weight_lambda=float(spec["weight_lambda"]),
        model_key=str(spec["model_key"]),
        post_mos_offset=post_offset,
        input_support_clamp=clamp_input,
        support_extrapolation=(ood_below or ood_above) and not clamp_input,
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
    input_support_clamp: bool = True
    support_extrapolation: bool = False


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


def _m1_params_sha256(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec["m1_params"], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
    """JSON-safe audit scalar: non-finite values become ``None`` (serialized as ``null``)."""
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
    """Fail-closed MOS metadata when the optional pipeline raises; skips digest verification."""
    return {
        "status": "unavailable",
        "research_status": RESEARCH_STATUS,
        "mos_policy": policy,
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
        embedded_coefficients_sha256(policy_id)
        if policy_id in {REVIEWED_MOS_POLICY, SELECTED_MOS_POLICY}:
            coefficients_digest(policy_id)
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
    )


def cvqa_upstream_from_run_record(record: dict[str, Any]) -> dict[str, Any]:
    """Build upstream gate input from a CVQA adapter JSON record (pure, testable)."""
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
    """Evaluate CVQA pin compatibility using explicit SHA comparison (metadata only)."""
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
    # Never trust commit_verified alone when actual_commit contradicts the pin.
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
    """Backward-compatible alias for mos_calibration_record metadata."""
    return cvqa_upstream_compatibility(cvqa_upstream)


_UPSTREAM_GATED_POLICIES = frozenset({REVIEWED_MOS_POLICY, SELECTED_MOS_POLICY})


def gate_reviewed_mos_outcome(
    outcome: MosCalibrationOutcome,
    cvqa_upstream: dict[str, Any] | None,
) -> MosCalibrationOutcome:
    """Block reviewed/selected MOS output when CVQA upstream pins are missing or incompatible."""
    if outcome.policy not in _UPSTREAM_GATED_POLICIES or not outcome.available:
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
    )


def mos_calibration_record(
    outcome: MosCalibrationOutcome,
    *,
    cvqa_upstream: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_id = outcome.policy
    if policy_id is None:
        record: dict[str, Any] = {
            "status": "unavailable",
            "research_status": RESEARCH_STATUS,
            "mos_policy": None,
            "canonical_score_input": _json_safe_number(outcome.canonical_score_input),
            "output_scale": OUTPUT_SCALE_LABEL,
            "reason": outcome.reason,
            "note": (
                f"MOS mapping suppressed ({outcome.reason}); canonical CVQA score unchanged."
            ),
        }
        return record
    pspec = policy_spec(policy_id)
    research_status = pspec.get("selection_status", RESEARCH_STATUS)
    record: dict[str, Any] = {
        "status": "available" if outcome.available else "unavailable",
        "research_status": research_status,
        "mos_policy": policy_id,
        "calibration_version": pspec["calibration_version"],
        "calibration_label": pspec["calibration_label"],
        "weight_kind": pspec["weight_kind"],
        "hdr_weight_lambda": pspec["hdr_weight_lambda"],
        "sdr_weight_lambda": pspec["sdr_weight_lambda"],
        "coefficient_cohort": pspec["coefficient_cohort"],
        "coefficient_source": pspec["coefficient_source"],
        "coefficients_digest_sha256": coefficients_digest(policy_id),
        "m1_implementation": "four_param_logistic_b_eff_nonneg",
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
    }
    input_clamp = bool(pspec.get("input_support_clamp", True))
    record["input_support_clamp"] = input_clamp
    if input_clamp:
        record["unresolved_policy_note"] = (
            "Training-support bilateral input clamping and final [1, 5] output bounding "
            "are retained. Removing the upper clamp or capping at 4.5 were discussed but "
            "not finalized for legacy/reviewed policies."
        )
    else:
        input_policy = pspec.get("input_support_policy", "unbounded_extrapolation")
        record["input_support_policy"] = input_policy
        record["unresolved_policy_note"] = (
            "Selected policy: no bilateral input clamp; scores outside historical support "
            f"({pspec['calibrators'][list(pspec['calibrators'])[0]]['support']['score_min']}, "
            f"{pspec['calibrators'][list(pspec['calibrators'])[0]]['support']['score_max']}) "
            "are extrapolated with OOD warnings only. Final output bounded to [1, 5]. "
            "Extrapolation uncertainty accepted by user policy choice."
        )
    manifest_id = pspec.get("manifest_id")
    if manifest_id:
        record["manifest_id"] = manifest_id
    if policy_id in {REVIEWED_MOS_POLICY, SELECTED_MOS_POLICY}:
        bundled = pspec.get("coefficients_bundle_sha256")
        if bundled:
            record["coefficients_source_bundle_sha256"] = bundled
        if policy_id == REVIEWED_MOS_POLICY:
            record["coefficients_digest_note"] = (
                "coefficients_digest_sha256 is the sealed research manifest bundle; "
                "embedded_coefficients_sha256 is recomputed from coefficients embedded in "
                "this module at runtime."
            )
        record["embedded_coefficients_sha256"] = embedded_coefficients_sha256(policy_id)
    if policy_id == SELECTED_MOS_POLICY:
        record["coefficients_digest_note"] = (
            "coefficients_digest_sha256 is a composite digest (frozen S1 bundle + "
            "post-output offsets + selection lock SHA-256s); it does not claim the "
            "native MOS offset is inside the S1 manifest. embedded_coefficients_sha256 "
            "includes offsets and lock provenance recomputed from this module."
        )
        record["post_output_offset_by_transfer"] = dict(
            pspec["post_output_offset_by_transfer"]
        )
        record["sdr_selection_lock_sha256"] = pspec["sdr_selection_lock_sha256"]
        record["hdr_selection_lock_sha256"] = pspec["hdr_selection_lock_sha256"]
        if outcome.post_mos_offset is not None:
            record["post_mos_offset"] = outcome.post_mos_offset
    if outcome.reason == "cvqa_upstream_incompatible":
        record["cvqa_upstream_gate_status"] = "blocked"
        record["status"] = "unavailable"
    if outcome.reason is not None:
        record["reason"] = outcome.reason
    score_requested = _json_safe_number(outcome.score_requested)
    score_effective = _json_safe_number(outcome.score_effective)
    if score_requested is not None:
        record["score_requested"] = score_requested
        record["score_effective"] = score_effective
        record["ood_below_support"] = outcome.ood_below_support
        record["ood_above_support"] = outcome.ood_above_support
        if outcome.input_support_clamp is not None:
            record["input_support_clamp"] = outcome.input_support_clamp
        if outcome.support_extrapolation:
            record["support_extrapolation"] = True
    if outcome.calibrator is not None:
        record["calibrator"] = outcome.calibrator
        spec = calibrators_for_policy(policy_id)[outcome.calibrator]
        record["research_arm"] = spec["research_arm"]
        record["route_weight_lambda"] = outcome.route_weight_lambda or spec["weight_lambda"]
        record["model_key"] = outcome.model_key or spec["model_key"]
        record["selected_ridge"] = spec["selected_ridge"]
        record["m1_params"] = list(spec["m1_params"])
        record["m1_params_sha256"] = _m1_params_sha256(spec)
        record["gamma_pq"] = spec["gamma_pq"]
        record["gamma_2160"] = spec["gamma_2160"]
        record["normalization"] = dict(spec["normalization"])
        record["support"] = dict(spec["support"])
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
        f"CVQA-HDR-NR scores using policy {policy_id!r}; it has not been validated for "
        "production subjective-quality prediction."
    )
    return record
