"""Endpoint-correct local CVQA-to-MOS tails for the frozen v7 calibration.

Research provenance (19 Nov 2026):
- SDR 1080p/2160p: monotone, convex C2 cubic B-splines with the original v7
  score=4.134484 anchor lowered by 0.080 MOS; return to v7 at 5.65.
- PQ 1080p: two-segment C2 quintic Bezier with a MOS=1 lower plateau.
- PQ 2160p: two-segment C2 cubic (truncated-power form).
- No original fitted v7 sigmoid coefficients are changed.
- No input support clipping. x<=0 is assigned MOS=1 as a domain floor.

The SDR derivative is now nondecreasing on the entire low tail. The curves
are research-only, with no subjective calibration data below
the splice boundaries. This module has no NumPy/SciPy runtime dependencies.
"""

from __future__ import annotations

from bisect import bisect_right

LOW_TAIL_VERSION = "v7-local-convex-c2-sdr-anchor-minus-008-join565-20261119"
RESEARCH_DELIVERABLE = (
    "perceptual_optimization/calibration/production_readiness_20261113/"
    "CVQA_MOS_V7_SDR_CONVEX_TAIL_20261119"
)
HDR_RESEARCH_DELIVERABLE = (
    "perceptual_optimization/calibration/production_readiness_20261113/"
    "CVQA_MOS_V7_LOCAL_SPLINE_C2_20261119"
)

SDR_ANCHOR_SCORE = 4.1344840888613135
SDR_ANCHOR_DELTA_MOS = 0.080
SDR_JOIN_SCORE = 5.65
PQ_JOIN_SCORE = 6.268450107554486

# Simple interior knots guarantee C2 continuity for the cubic B-spline.
# Six interior knots provide enough degrees of freedom to enforce global
# nonnegative curvature, unlike the previous two-interior-knot version.
_SDR_KNOTS = (
    0.0, 0.0, 0.0, 0.0, 0.19, 0.4308835476868419, 0.6717670953736838,
    0.7317670953736838, 0.8524719024555261, 0.9463534190747368, 1.0, 1.0, 1.0, 1.0,
)
_SDR_CONTROLS: dict[int, tuple[float, ...]] = {
    1080: (
        1.0, 1.0357279888964983, 1.1173093043172093, 1.2573954002992864, 1.3976062270202059,
        1.5266780666417807, 1.6183440814400967, 1.7123812243533703, 1.769388190642436, 1.7928154840710815,
    ),
    2160: (
        1.0, 1.0463399300133231, 1.1517705095252326, 1.3240360712851942, 1.4796095405353704,
        1.6129317920269242, 1.7045194371454926, 1.7978163924161799, 1.8548233587052454, 1.8782506521338909,
    ),
}

_PQ_1080_SPLIT = 0.575
_PQ_1080_RIGHT_CONTROLS = (
    0.0, 0.0, 0.0,
    0.015195872364660537, 0.10925972083011509, 0.286185374125175,
)
_PQ_2160_SPLIT = 0.775
_PQ_2160_COEFS = (
    4.497130586296919e-13,
    0.03565801406825231,
    0.424084613684314,
    4.858656163750658,
)


def low_tail_payload() -> dict:
    """JSON-serializable *effective* parameters for immutable identity hashing."""
    return {
        "version": LOW_TAIL_VERSION,
        "research": RESEARCH_DELIVERABLE,
        "hdr_research": HDR_RESEARCH_DELIVERABLE,
        "sdr_anchor_score": SDR_ANCHOR_SCORE,
        "sdr_anchor_delta_mos": SDR_ANCHOR_DELTA_MOS,
        "sdr_join_score": SDR_JOIN_SCORE,
        "sdr_knots": list(_SDR_KNOTS),
        "sdr_controls": {
            str(h): list(_SDR_CONTROLS[h]) for h in sorted(_SDR_CONTROLS)
        },
        "pq_join_score": PQ_JOIN_SCORE,
        "pq_1080_split": _PQ_1080_SPLIT,
        "pq_1080_right_controls": list(_PQ_1080_RIGHT_CONTROLS),
        "pq_2160_split": _PQ_2160_SPLIT,
        "pq_2160_coefs": list(_PQ_2160_COEFS),
        "floor_at_nonpositive_score": True,
    }


def _cubic_bspline(t: float, controls: tuple[float, ...]) -> float:
    """Cubic de Boor evaluation, including repeated clamped end knots."""
    knots = _SDR_KNOTS
    degree = 3
    span = min(bisect_right(knots, t) - 1, len(controls) - 1)
    span = max(span, degree)
    values = [controls[span - degree + j] for j in range(degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            lo = knots[span - degree + j]
            hi = knots[j + 1 + span - r]
            alpha = (t - lo) / (hi - lo) if hi != lo else 0.0
            values[j] = (1.0 - alpha) * values[j - 1] + alpha * values[j]
    return values[degree]


def _quintic_bezier(t: float, controls: tuple[float, ...]) -> float:
    """De Casteljau form for a quintic Bezier curve on t in [0,1]."""
    buf = list(controls)
    for n in range(len(buf) - 1, 0, -1):
        for j in range(n):
            buf[j] = (1.0 - t) * buf[j] + t * buf[j + 1]
    return buf[0]


def correct_low_tail(
    score: float,
    *,
    transfer: str,
    encoded_height: int,
    frozen_v7_mos: float,
) -> float:
    """Apply local endpoint correction; return v7 *unchanged* above each join.

    The original canonical CVQA score remains unmodified. Only the reported
    estimated MOS undergoes the lower-tail transfer. Negative scores are
    outside the model domain; conservatively report the MOS=1 floor.
    """
    if score <= 0.0:
        return 1.0
    if transfer == "sdr":
        if score >= SDR_JOIN_SCORE:
            return frozen_v7_mos
        return _cubic_bspline(score / SDR_JOIN_SCORE, _SDR_CONTROLS[encoded_height])
    if transfer == "pq":
        if score >= PQ_JOIN_SCORE:
            return frozen_v7_mos
        t = score / PQ_JOIN_SCORE
        if encoded_height == 1080:
            if t <= _PQ_1080_SPLIT:
                return 1.0
            rel = (t - _PQ_1080_SPLIT) / (1.0 - _PQ_1080_SPLIT)
            return 1.0 + _quintic_bezier(rel, _PQ_1080_RIGHT_CONTROLS)
        a, b, c, j = _PQ_2160_COEFS
        return 1.0 + a*t + b*t*t + c*t*t*t + j*max(t - _PQ_2160_SPLIT, 0.0)**3
    raise ValueError(f"unsupported low-tail transfer {transfer!r}")
