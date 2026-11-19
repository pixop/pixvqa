#!/usr/bin/env python3
"""Regression tests for CVQA-HDR-NR MOS 1–5 calibration (single selected formula)."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pixvqa_cvqahdr import PINNED_CHECKPOINT_SHA256, PINNED_COMMIT, PINNED_PROFILE_SHA256
from pixvqa_cvqahdr import mos_calibration as mc  # noqa: E402

SELECTED_FIXTURE = ROOT / "tests/fixtures/mos_calibration/avt_transfer_separated_v7_selected.json"
FROZEN_V7_RESEARCH_COEFFICIENTS = (
    ROOT / "tests/fixtures/mos_calibration/cvqa_mos_v7_source_coefficients.json"
)
SELECTED_NATIVE_CASES = (
    ROOT / "tests/fixtures/mos_calibration/avt_transfer_separated_v7_selected_per_clip.csv"
)
SELECTED_VD6_CASES = (
    ROOT / "tests/fixtures/mos_calibration/avt_transfer_separated_v7_vd6_per_clip.csv"
)
LOW_TAIL_RESEARCH_POINTS = (
    ROOT / "tests/fixtures/mos_calibration/v7_local_c2_minus008_join565_research_points.csv"
)
HISTORICAL_V5_FIXTURE = (
    ROOT / "tests/fixtures/mos_calibration/avt_selected_s1_sdr_p1_hdr_c0.json"
)
SELECTED_COEFFICIENTS_DIGEST_SHA256 = "24fc1811327281cff37f37ece8db129bccada6f9e95252f1db5037fcffa4f899"
EMBEDDED_COEFFICIENTS_SHA256 = mc.EMBEDDED_COEFFICIENTS_SHA256
SDR_POST_MOS_OFFSET = 0.5107732905216783
SDR_SELECTION_LOCK_SHA256 = "f65a19498735f0ce17257079a913f2ece25e9692b56c8392e27ce7d29ae079ee"
HDR_SELECTION_LOCK_SHA256 = "7882d79f68f62fc06e034c3ed9f518cbdd0940f957b83df344b46858273b3f8c"
SOURCE_JSON_SHA256 = "e817ed479079c0e208e8ec0d84aad3b5836606eeb329585d27e2c461c24b8e73"
SCRIPT_PATH = Path(__file__).resolve()

TRANSFER_NORMALIZATION = {
    "sdr": {'mean': 8.010332079849347, 'std': 1.241145006178454},
    "pq": {'mean': 8.769753592063894, 'std': 0.9333470039280067},
}
TRANSFER_SUPPORT = {
    "sdr": [4.1344840888613135, 10.038337481465987],
    "pq": [6.268450107554486, 10.051497077192835],
}

# Standalone full-precision v7 oracle values computed from the research four-tuple.
DELIVERABLE_ORACLE: tuple[tuple[str, int, float, float], ...] = (
    ('sdr', 1080, 8.0, 3.0206399295656867),
    ('sdr', 2160, 8.0, 3.106075097628496),
    ('pq', 1080, 8.0, 2.275651960915209),
    ('pq', 2160, 8.0, 2.5045523449082725),
    ('sdr', 1080, 9.0, 3.890294761627456),
    ('sdr', 2160, 9.0, 3.9757299296902655),
    ('pq', 1080, 9.0, 3.1440665780738164),
    ('pq', 2160, 9.0, 3.37296696206688),
    ('sdr', 1080, 9.5, 4.2931666670467274),
    ('sdr', 2160, 9.5, 4.378601835109537),
    ('pq', 1080, 9.5, 3.5603917444826365),
    ('pq', 2160, 9.5, 3.7892921284757),
    ('sdr', 1080, 10.0, 4.6328337676073605),
    ('sdr', 2160, 10.0, 4.71826893567017),
    ('pq', 1080, 10.0, 3.9182233198793646),
    ('pq', 2160, 10.0, 4.147123703872428),
    ('sdr', 1080, 11.0, 5),
    ('sdr', 2160, 11.0, 5),
    ('pq', 1080, 11.0, 4.414355471571109),
    ('pq', 2160, 11.0, 4.643255855564172),
)
BOUNDARY_SCORES = (0.0, 4.1344840888613135, 4.1345, 6.268450107554486, 8.5, 10.0, 10.051497077192835, 11.0, 12.0)
COMBO_CASES = (
    ("pq", 1080, "U0_M7_Bias"),
    ("pq", 2160, "U0_M7_Bias"),
    ("sdr", 1080, "U_v2_M7_Bias"),
    ("sdr", 2160, "U_v2_M7_Bias"),
)


def _require_fixture(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required MOS fixture missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 68:
        count = len(cases) if isinstance(cases, list) else 0
        raise AssertionError(f"fixture {path} must contain exactly 68 cases, got {count}")
    return data


def _pinned_upstream_payload() -> dict:
    return {
        "upstream": {"actual_commit": PINNED_COMMIT, "commit_verified": True},
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "profile_sha256": PINNED_PROFILE_SHA256,
    }


def _fake_adapter_record(*, upstream: dict | None = None) -> dict:
    return {
        "quality_score": 8.5,
        "upstream": upstream or {"actual_commit": PINNED_COMMIT},
        "checkpoint": {"sha256": PINNED_CHECKPOINT_SHA256},
        "calibration_profile": {"sha256": PINNED_PROFILE_SHA256},
    }


def normalize_protocol_scalar(value: str) -> str:
    if len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0]:
        return value[1:-1]
    return value


def summary_mos_branch(mos_status: str, mos_value: str) -> str:
    if mos_status == "available" and mos_value not in ("unknown", "unavailable"):
        return "show_mos"
    if mos_status == "unavailable":
        return "show_unavailable"
    return "omit_mos"


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _research_models() -> dict:
    return json.loads(FROZEN_V7_RESEARCH_COEFFICIENTS.read_text(encoding="utf-8"))["selected"]


def independent_oracle_predict(
    canonical_score: float,
    *,
    transfer: str,
    encoded_height: int,
) -> float:
    """Independent full four-tuple M1+M7 oracle; never reads runtime coefficients."""
    model = _research_models()["PQ" if transfer == "pq" else "SDR"]
    a, b, c, ls = model["M1_four_parameter_tuple"]
    norm = model["normalization"]
    z = (float(canonical_score) - norm["mean"]) / norm["std"]
    u = a + max(0.0, b) * (z - c) / max(0.001, math.exp(ls))
    base = 1 + 4 * _sigmoid(u)
    mid = min(max(base + (model["M7_PQ_correction"] if transfer == "pq" else 0.0)
                  + (model["M7_2160_correction"] if encoded_height == 2160 else 0.0), 1.0), 5.0)
    return min(max(mid + model["native_MOS_offset"], 1.0), 5.0)


def setUpModule() -> None:
    os.environ.pop(mc.ENV_MOS_POLICY, None)


class MosFormulaTests(unittest.TestCase):
    def test_route_params_match_deliverable(self) -> None:
        sdr = mc.route_params("sdr", 1080)
        self.assertAlmostEqual(sdr["a"], -0.49117628849094824, places=12)
        self.assertAlmostEqual(sdr["b"], 1.0977838841771388, places=12)
        self.assertAlmostEqual(sdr["d"], SDR_POST_MOS_OFFSET, places=12)
        pq = mc.route_params("pq", 1080)
        self.assertAlmostEqual(pq["a"], 0.07227163628268213, places=12)
        self.assertAlmostEqual(pq["b"], 0.8293304839931255, places=12)
        self.assertAlmostEqual(pq["d"], -0.131036744556774, places=12)

    def test_deliverable_oracle_all_routes(self) -> None:
        max_err = 0.0
        for transfer, height, score, expected in DELIVERABLE_ORACLE:
            independent = independent_oracle_predict(
                score, transfer=transfer, encoded_height=height
            )
            runtime = mc.predict_mos(score, transfer=transfer, encoded_height=height)
            self.assertAlmostEqual(independent, runtime, delta=1e-12)
            err = abs(runtime - expected)
            max_err = max(max_err, err)
            self.assertLessEqual(err, 5e-12, msg=f"{transfer}/{height} x={score}")
        self.assertLessEqual(max_err, 5e-12)

    def test_independent_oracle_all_boundaries(self) -> None:
        # Frozen v7 is still the independent oracle ABOVE the local splice.
        # Below it, test against the research-generated fixed-point fixture.
        for transfer, height, calibrator in COMBO_CASES:
            for score in BOUNDARY_SCORES:
                expected = independent_oracle_predict(
                    score, transfer=transfer, encoded_height=height
                )
                actual = mc.predict_mos(
                    score,
                    transfer=transfer,
                    encoded_height=height,
                    calibrator=calibrator,
                )
                join = mc.mos_low_tail.SDR_JOIN_SCORE if transfer == "sdr" else mc.mos_low_tail.PQ_JOIN_SCORE
                if score >= join:
                    self.assertAlmostEqual(actual, expected, delta=1e-12, msg=f"{transfer}/{height} x={score}")
                else:
                    self.assertGreaterEqual(actual, 1.0)
                    self.assertLessEqual(actual, 5.0)
                    if score == 0.0:
                        self.assertEqual(actual, 1.0)

    def test_pq_2160_correction_delta(self) -> None:
        score = 8.5
        mos_1080 = mc.predict_mos(score, transfer="pq", encoded_height=1080)
        mos_2160 = mc.predict_mos(score, transfer="pq", encoded_height=2160)
        self.assertGreater(mos_2160, mos_1080)
        delta = mc.route_params("pq", 2160)["d"] - mc.route_params("pq", 1080)["d"]
        self.assertAlmostEqual(mos_2160 - mos_1080, delta, delta=1e-12)

    def test_no_input_clamp_below_support(self) -> None:
        x = 3.0
        at_floor = mc.predict_mos(TRANSFER_SUPPORT["sdr"][0], transfer="sdr", encoded_height=1080)
        below = mc.predict_mos(x, transfer="sdr", encoded_height=1080)
        detail = mc.predict_mos_detailed(x, transfer="sdr", encoded_height=1080)
        self.assertNotAlmostEqual(below, at_floor, places=3)
        self.assertEqual(detail.score_effective, x)
        self.assertTrue(detail.ood_below_support)
        self.assertTrue(detail.support_extrapolation)
        self.assertFalse(detail.input_support_clamp)

    def test_extrapolation_above_support(self) -> None:
        at_support = mc.predict_mos(TRANSFER_SUPPORT["pq"][1], transfer="pq", encoded_height=1080)
        above = mc.predict_mos(11.0, transfer="pq", encoded_height=1080)
        self.assertGreater(above, at_support)
        self.assertEqual(mc.predict_mos(11.0, transfer="sdr", encoded_height=1080), 5.0)

    def test_monotonicity_all_routes(self) -> None:
        scores = [i * 0.25 for i in range(8, 48)]
        for transfer, height, _ in COMBO_CASES:
            mos = [mc.predict_mos(s, transfer=transfer, encoded_height=height) for s in scores]
            for left, right in zip(mos, mos[1:]):
                self.assertLessEqual(left, right, msg=f"{transfer}/{height}")

    def test_output_bounds(self) -> None:
        for transfer in ("pq", "sdr"):
            for score in (0.0, 4.0, 7.0, 10.0, 11.0, 20.0):
                for height in (1080, 2160):
                    mos = mc.predict_mos(score, transfer=transfer, encoded_height=height)
                    self.assertGreaterEqual(mos, 1.0)
                    self.assertLessEqual(mos, 5.0)
                    self.assertTrue(math.isfinite(mos))


    def test_transfer_separation_normalization_and_support(self) -> None:
        for transfer in ("sdr", "pq"):
            self.assertEqual(mc._NORMALIZATION_BY_TRANSFER[transfer], TRANSFER_NORMALIZATION[transfer])
            lo, hi = TRANSFER_SUPPORT[transfer]
            self.assertEqual(mc._SUPPORT_BY_TRANSFER[transfer], {"score_min": lo, "score_max": hi})
            detail_lo = mc.predict_mos_detailed(lo - 0.01, transfer=transfer, encoded_height=1080)
            detail_hi = mc.predict_mos_detailed(hi + 0.01, transfer=transfer, encoded_height=2160)
            self.assertTrue(detail_lo.ood_below_support)
            self.assertTrue(detail_hi.ood_above_support)
            self.assertFalse(detail_lo.input_support_clamp)
            self.assertEqual(detail_lo.score_effective, lo - 0.01)

    def test_source_coefficient_hash_matches_frozen_v7(self) -> None:
        import hashlib
        raw = FROZEN_V7_RESEARCH_COEFFICIENTS.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SOURCE_JSON_SHA256)
        data = json.loads(raw)
        self.assertEqual(data["selected"]["SDR"]["n_fit"], 200)
        self.assertEqual(data["selected"]["PQ"]["n_fit"], 192)
        self.assertEqual(data["selected"]["SDR"]["lambda"], 0.0)
        self.assertEqual(data["selected"]["PQ"]["lambda"], 0.0)
        self.assertNotEqual(data["selected"]["SDR"]["normalization"], data["selected"]["PQ"]["normalization"])


class MosLocalTailTests(unittest.TestCase):
    """Research-backed endpoints, joins and no-change regression checks."""

    def test_direct_research_oracle_points_all_routes(self) -> None:
        import csv
        with LOW_TAIL_RESEARCH_POINTS.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 48)
        for row in records:
            transfer = row["transfer"]
            height = int(row["encoded_height"])
            score = float(row["cvqa"])
            predicted = mc.predict_mos(score, transfer=transfer, encoded_height=height)
            self.assertAlmostEqual(predicted, float(row["expected_mos"]), delta=2e-11,
                msg=f"{transfer}/{height}:CVQA={score}")
            baseline = independent_oracle_predict(score, transfer=transfer, encoded_height=height)
            self.assertAlmostEqual(baseline, float(row["frozen_v7"]), delta=2e-11)

    def test_sdr_anchor_reduction_and_exact_zero(self) -> None:
        anchor = mc.mos_low_tail.SDR_ANCHOR_SCORE
        for height in (1080, 2160):
            original = independent_oracle_predict(anchor, transfer="sdr", encoded_height=height)
            revised = mc.predict_mos(anchor, transfer="sdr", encoded_height=height)
            self.assertAlmostEqual(original - revised, 0.080, delta=3e-12)
            self.assertEqual(mc.predict_mos(0., transfer="sdr", encoded_height=height), 1.0)
            self.assertEqual(mc.predict_mos(0., transfer="pq", encoded_height=height), 1.0)

    def test_no_high_score_change_in_any_route(self) -> None:
        for transfer, height, _ in COMBO_CASES:
            join = mc.mos_low_tail.SDR_JOIN_SCORE if transfer == "sdr" else mc.mos_low_tail.PQ_JOIN_SCORE
            for score in (join, math.nextafter(join, math.inf), join + .001, 6.0, 8.0, 10.0, 11.0, 15.0):
                if score < join:
                    continue
                actual = mc.predict_mos(score, transfer=transfer, encoded_height=height)
                reference = independent_oracle_predict(score, transfer=transfer, encoded_height=height)
                self.assertAlmostEqual(actual, reference, delta=1e-12,
                    msg=f"{transfer}/{height} x={score}")
                # Equality to the *original runtime arithmetic* is exact.
                params = mc.route_params(transfer, height)
                norm = mc._NORMALIZATION_BY_TRANSFER[transfer]
                z = (score - norm["mean"]) / norm["std"]
                frozen_runtime = mc._clamp(1 + 4*mc._sigmoid(params["a"] + params["b"]*z) + params["d"], 1., 5.)
                self.assertEqual(actual, frozen_runtime)

    def test_monotone_smooth_join_and_bounds(self) -> None:
        for transfer, height, _ in COMBO_CASES:
            scores = [i * .02 for i in range(601)]
            values = [mc.predict_mos(x, transfer=transfer, encoded_height=height) for x in scores]
            self.assertEqual(values[0], 1.0)
            self.assertGreaterEqual(min(values), 1.0)
            self.assertLessEqual(max(values), 5.0)
            self.assertTrue(all(a <= b + 1e-12 for a,b in zip(values,values[1:])))
            # The corrected low-tail is C2 at the cutoff; finite differences
            # test agreement to the independently frozen v7 derivatives there.
            c = mc.mos_low_tail.SDR_JOIN_SCORE if transfer == "sdr" else mc.mos_low_tail.PQ_JOIN_SCORE
            step = 0.001
            f = lambda x: mc.predict_mos(x, transfer=transfer, encoded_height=height)
            old = lambda x: independent_oracle_predict(x, transfer=transfer, encoded_height=height)
            slope_left = (f(c)-f(c-step))/step
            slope_right = (old(c+step)-old(c))/step
            self.assertAlmostEqual(slope_left, slope_right, delta=1e-3)
            curv_left = (f(c)-2*f(c-step)+f(c-2*step))/step**2
            curv_right = (old(c+2*step)-2*old(c+step)+old(c))/step**2
            self.assertAlmostEqual(curv_left, curv_right, delta=.02)

    def test_local_tail_provenance_and_digest_tamper(self) -> None:
        details = mc.policy_spec()
        self.assertEqual(details["low_tail"]["sdr_anchor_delta_mos"], 0.080)
        self.assertEqual(details["low_tail"]["sdr_join_score"], 5.65)
        self.assertIn("low_tail", mc._embedded_coefficients_payload())
        self.assertIn("low_tail", mc._coefficients_digest_payload())
        before = mc.predict_mos(2.0, transfer="sdr", encoded_height=1080)
        original = mc.mos_low_tail._SDR_CONTROLS[1080]
        try:
            mc.mos_low_tail._SDR_CONTROLS[1080] = (original[0], original[1] + .1, *original[2:])
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                mc.predict_mos(2.0, transfer="sdr", encoded_height=1080)
        finally:
            mc.mos_low_tail._SDR_CONTROLS[1080] = original
        self.assertEqual(mc.predict_mos(2.0, transfer="sdr", encoded_height=1080), before)


class MosFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _require_fixture(SELECTED_FIXTURE)

    def test_fixture_policy_and_count(self) -> None:
        self.assertEqual(self.fixture["policy"], mc.MOS_POLICY)
        self.assertEqual(len(self.fixture["cases"]), 68)

    def test_selected_golden_predictions_match(self) -> None:
        max_err = 0.0
        for case in self.fixture["cases"]:
            expected = case["expected_mos"]
            actual = mc.predict_mos(
                case["score"],
                transfer=case["transfer"],
                encoded_height=case["encoded_height"],
                calibrator=case["calibrator"],
            )
            err = abs(actual - expected)
            max_err = max(max_err, err)
            self.assertLessEqual(err, 1e-10, msg=f"{case['dataset']}:{case['id']}")
        self.assertLessEqual(max_err, 1e-10)

    def test_retrospective_native_clip_scores_reproduce_v7_core(self) -> None:
        import csv
        with SELECTED_NATIVE_CASES.open(newline="") as handle:
            cases = list(csv.DictReader(handle))
        self.assertEqual(len(cases), 89)
        for case in cases:
            t = "sdr" if case["route"] == "SDR" else "pq"
            h = 2160 if case["domain_key"].startswith("2160") else 1080
            expected = independent_oracle_predict(float(case["score"]), transfer=t, encoded_height=h)
            got = mc.predict_mos(float(case["score"]), transfer=t, encoded_height=h)
            self.assertAlmostEqual(got, expected, delta=1e-12)
            if t == "pq":
                self.assertAlmostEqual(got, float(case["core"]), delta=1e-12)
            else:
                self.assertAlmostEqual(got, min(5.0, float(case["core"]) + SDR_POST_MOS_OFFSET), delta=1e-12)

    def test_sdr_vd6_fixed_offset_transfer(self) -> None:
        import csv
        with SELECTED_VD6_CASES.open(newline="") as handle:
            cases = list(csv.DictReader(handle))
        self.assertEqual(len(cases), 6)
        for case in cases:
            h = 2160 if case["domain_key"].startswith("2160") else 1080
            got = mc.predict_mos(float(case["score"]), transfer="sdr", encoded_height=h)
            self.assertAlmostEqual(got, float(case["new_SDR_Uv2_with_SDR54_offset"]), delta=1e-12)

    def test_historical_fixture_is_preserved(self) -> None:
        old = _require_fixture(HISTORICAL_V5_FIXTURE)
        self.assertEqual(len(old["cases"]), 68)
        self.assertEqual(old["calibration_version"], "selected-s1-sdr-p1-hdr-c0-unclamped-input-20261114-v2")
        self.assertAlmostEqual(old["post_output_offset_by_transfer"]["sdr"], 0.4105017796126768)

    def test_fixture_missing_fails_hard(self) -> None:
        original_is_file = Path.is_file

        def _missing(self: Path) -> bool:
            if self == SELECTED_FIXTURE:
                return False
            return original_is_file(self)

        with patch.object(Path, "is_file", _missing):
            with self.assertRaises(FileNotFoundError):
                _require_fixture(SELECTED_FIXTURE)


class MosMetadataTests(unittest.TestCase):
    def test_pq_transfer_classification(self) -> None:
        transfer, reason = mc.classify_transfer_tag("smpte2084")
        self.assertEqual(transfer, "pq")
        self.assertIsNone(reason)

    def test_sdr_transfer_classification(self) -> None:
        transfer, reason = mc.classify_transfer_tag("bt709")
        self.assertEqual(transfer, "sdr")
        self.assertIsNone(reason)

    def test_hlg_rejected(self) -> None:
        transfer, reason = mc.classify_transfer_tag("arib-std-b67")
        self.assertIsNone(transfer)
        self.assertEqual(reason, "transfer_hlg")

    def test_unknown_transfer_rejected(self) -> None:
        transfer, reason = mc.classify_transfer_tag("")
        self.assertIsNone(transfer)
        self.assertEqual(reason, "transfer_missing")

    def test_bt2020_ambiguous_unsupported(self) -> None:
        transfer, reason = mc.classify_transfer_tag("bt2020")
        self.assertIsNone(transfer)
        self.assertEqual(reason, "transfer_unsupported")

    def test_apply_unavailable_height(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 720, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        self.assertFalse(out.available)
        self.assertEqual(out.reason, "height_unsupported")

    def test_apply_available_pq(self) -> None:
        score = 9.903605146801397
        expected = independent_oracle_predict(score, transfer="pq", encoded_height=1080)
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(score, "video.mp4")
        self.assertTrue(out.available)
        self.assertAlmostEqual(out.mos_calibrated_cvqa or 0.0, expected, delta=1e-12)

    def test_apply_available_sdr(self) -> None:
        score = 8.5
        expected = independent_oracle_predict(score, transfer="sdr", encoded_height=2160)
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 2160, "bt709")):
            out = mc.apply_mos_calibration(score, "video.mp4")
        self.assertTrue(out.available)
        self.assertAlmostEqual(out.mos_calibrated_cvqa or 0.0, expected, delta=1e-12)

    def test_ffprobe_errors_preserve_canonical(self) -> None:
        canonical = 7.25
        for probe_return in (
            (0, 0, "ffprobe_timeout"),
            (0, 0, "invalid_ffprobe_json"),
            (0, 0, "no_video_stream"),
        ):
            with patch.object(mc, "probe_encoded_geometry", return_value=probe_return):
                out = mc.apply_mos_calibration(canonical, "video.mp4")
            self.assertFalse(out.available)
            self.assertEqual(out.canonical_score_input, canonical)


class MosRoutingTests(unittest.TestCase):
    def test_domain_routing(self) -> None:
        self.assertEqual(mc.calibrator_for_transfer("pq"), "U0_M7_Bias")
        self.assertEqual(mc.calibrator_for_transfer("sdr"), "U_v2_M7_Bias")

    def test_calibrator_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            mc.predict_mos(8.5, transfer="pq", encoded_height=1080, calibrator="U_v2_M7_Bias")


class MosProvenanceTests(unittest.TestCase):
    def test_public_policy_metadata_is_defensively_copied(self) -> None:
        # A consumer modifying returned metadata must not silently change the
        # runtime model identity, source hashes, or selected route mapping.
        baseline_version = mc.policy_spec()["calibration_version"]
        baseline_offset = mc.policy_spec()["post_output_offset_by_transfer"]["sdr"]
        baseline_route = mc.calibrators_for_policy()["U_v2_M7_Bias"]["route"]
        before = mc.predict_mos(10.0, transfer="sdr", encoded_height=2160)
        snapshot = mc.policy_spec()
        snapshot["calibration_version"] = "modified_by_consumer"
        snapshot["post_output_offset_by_transfer"]["sdr"] = -4.0
        snapshot["calibrators"]["U_v2_M7_Bias"]["route"] = "pq"
        calibrators = mc.calibrators_for_policy()
        calibrators["U_v2_M7_Bias"]["normalization"]["mean"] = -1.0
        calibrators["U_v2_M7_Bias"]["route"] = "pq"
        self.assertEqual(mc.policy_spec()["calibration_version"], baseline_version)
        self.assertEqual(mc.policy_spec()["post_output_offset_by_transfer"]["sdr"], baseline_offset)
        self.assertEqual(mc.calibrators_for_policy()["U_v2_M7_Bias"]["route"], baseline_route)
        self.assertEqual(mc.coefficients_digest(), SELECTED_COEFFICIENTS_DIGEST_SHA256)
        self.assertEqual(mc.predict_mos(10.0, transfer="sdr", encoded_height=2160), before)

    def test_fixed_policy_metadata(self) -> None:
        self.assertEqual(mc.MOS_POLICY, "selected_s1_sdr_p1_hdr_c0")
        self.assertEqual(mc.DEFAULT_MOS_POLICY, mc.MOS_POLICY)
        self.assertEqual(mc.coefficients_digest(), SELECTED_COEFFICIENTS_DIGEST_SHA256)
        self.assertEqual(mc.embedded_coefficients_sha256(), EMBEDDED_COEFFICIENTS_SHA256)

    def test_record_includes_provenance(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        record = mc.mos_calibration_record(out, cvqa_upstream=_pinned_upstream_payload())
        self.assertEqual(record["mos_policy"], mc.MOS_POLICY)
        self.assertEqual(record["calibration_version"], mc.CALIBRATION_VERSION)
        self.assertEqual(record["coefficient_source_json_sha256"], SOURCE_JSON_SHA256)
        self.assertIn("route_sigmoid", record)
        self.assertIn("obsolete_env_note", record)
        self.assertFalse(record["input_support_clamp"])
        self.assertEqual(record["sdr_selection_lock_sha256"], SDR_SELECTION_LOCK_SHA256)
        self.assertEqual(record["coefficient_cohort"], "SDR200_PQ192_separate")
        self.assertEqual(record["route_weight_lambda"], 0.0)
        self.assertEqual(record["normalization"], TRANSFER_NORMALIZATION["pq"])
        self.assertEqual(record["support"], {
            "score_min": TRANSFER_SUPPORT["pq"][0],
            "score_max": TRANSFER_SUPPORT["pq"][1],
        })


class MosPolicyTests(unittest.TestCase):
    def test_default_policy_is_selected(self) -> None:
        policy, err = mc.resolve_mos_policy()
        self.assertIsNone(err)
        self.assertEqual(policy, mc.MOS_POLICY)

    def test_explicit_invalid_policy_fail_closed(self) -> None:
        policy, err = mc.resolve_mos_policy("legacy_target_lambda1_both")
        self.assertIsNone(policy)
        self.assertEqual(err, "invalid_policy:legacy_target_lambda1_both")

    @patch.dict(os.environ, {mc.ENV_MOS_POLICY: "legacy_target_lambda1_both"}, clear=False)
    def test_obsolete_env_fail_closed(self) -> None:
        policy, err = mc.resolve_mos_policy()
        self.assertIsNone(policy)
        self.assertEqual(err, "obsolete_policy_env:legacy_target_lambda1_both")
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        self.assertFalse(out.available)
        self.assertTrue(out.reason and out.reason.startswith("obsolete_policy_env:"))

    @patch.dict(os.environ, {mc.ENV_MOS_POLICY: mc.MOS_POLICY}, clear=False)
    def test_env_selected_policy_accepted(self) -> None:
        policy, err = mc.resolve_mos_policy()
        self.assertIsNone(err)
        self.assertEqual(policy, mc.MOS_POLICY)

    def test_predict_invalid_score_type(self) -> None:
        with self.assertRaises(ValueError):
            mc.predict_mos_detailed("bad", transfer="pq", encoded_height=1080)


class MosCvqaUpstreamGateTests(unittest.TestCase):
    def test_compatibility_all_pins_match(self) -> None:
        compat = mc.cvqa_upstream_compatibility(_pinned_upstream_payload())
        self.assertTrue(compat["compatible"])

    def test_gated_available_when_pins_match(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        self.assertTrue(out.available)
        gated = mc.gate_reviewed_mos_outcome(out, _pinned_upstream_payload())
        self.assertTrue(gated.available)

    def test_blocked_unpinned_checkpoint(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        upstream = _pinned_upstream_payload()
        upstream["checkpoint_sha256"] = "0" * 64
        gated = mc.gate_reviewed_mos_outcome(out, upstream)
        self.assertFalse(gated.available)
        self.assertEqual(gated.reason, "cvqa_upstream_incompatible")
        record = mc.mos_calibration_record(gated, cvqa_upstream=upstream)
        self.assertNotIn("mos_calibrated_cvqa", record)

    def test_gate_preserves_canonical_on_block(self) -> None:
        canonical = 8.502661863950266
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(canonical, "video.mp4")
        gated = mc.gate_reviewed_mos_outcome(out, None)
        self.assertEqual(gated.canonical_score_input, canonical)
        self.assertIsNone(gated.mos_calibrated_cvqa)


class MosProtocolTests(unittest.TestCase):
    def test_protocol_lines_include_mos_fields(self) -> None:
        from pixvqa_cvqahdr.__main__ import protocol_lines

        score = 8.502661863950266
        mos = mc.predict_mos(score, transfer="pq", encoded_height=1080)
        record = {
            "quality_score": score,
            "raw_model_score": 0.38,
            "sampling": {"sampled_frames": 10, "opencv_frame_count": 500, "opencv_fps": 50.0, "rounded_fps": 50},
            "temporal": {
                "sampling_density_hz": 1.0,
                "decoded_frame_count": None,
                "padded_samples": None,
                "pooled_parity": True,
                "qcs_eligible": True,
                "samples": [],
            },
            "device": "cuda",
            "device_requested": "cuda",
            "mos_calibrated_cvqa": mos,
            "mos_calibration": {"status": "available", "mos_policy": mc.MOS_POLICY},
        }
        joined = "\n".join(protocol_lines(record, "out.json"))
        self.assertIn(f"pixvqa-value mos_calibrated_cvqa {mos}", joined)
        self.assertIn("pixvqa-value mos_policy selected_s1_sdr_p1_hdr_c0", joined)

    def test_summary_mos_branch(self) -> None:
        self.assertEqual(summary_mos_branch("available", "3.5"), "show_mos")
        self.assertEqual(summary_mos_branch("unavailable", "unavailable"), "show_unavailable")


class MosIntegrityTests(unittest.TestCase):
    def test_direct_prediction_fails_closed_on_tampered_coefficients(self) -> None:
        # The public predict_mos API must not silently bypass the integrity
        # checks that already protect apply_mos_calibration().
        original = dict(mc._ROUTE_PARAMS[("sdr", 1080)])
        try:
            mc._ROUTE_PARAMS[("sdr", 1080)] = {**original, "b": original["b"] + 0.5}
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                mc.predict_mos(9.0, transfer="sdr", encoded_height=1080)
        finally:
            mc._ROUTE_PARAMS[("sdr", 1080)] = original
        self.assertAlmostEqual(
            mc.predict_mos(10.0, transfer="sdr", encoded_height=1080),
            4.6328337676073605, delta=1e-12,
        )

    def test_digest_raises_on_mismatch(self) -> None:
        with patch.object(mc, "SELECTED_COEFFICIENTS_DIGEST_SHA256", "0" * 64):
            with self.assertRaises(ValueError):
                mc.coefficients_digest()

    def test_tamper_yields_unavailable(self) -> None:
        original = dict(mc._ROUTE_PARAMS[("sdr", 1080)])
        try:
            mc._ROUTE_PARAMS[("sdr", 1080)] = {**original, "a": original["a"] + 1.0}
            with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
                out = mc.apply_mos_calibration(8.5, "video.mp4")
            self.assertFalse(out.available)
            self.assertIn("calibration_error", out.reason or "")
        finally:
            mc._ROUTE_PARAMS[("sdr", 1080)] = original

    def test_tamper_transfer_normalization_yields_unavailable(self) -> None:
        original = dict(mc._NORMALIZATION_BY_TRANSFER["sdr"])
        try:
            mc._NORMALIZATION_BY_TRANSFER["sdr"] = {**original, "mean": original["mean"] + 0.1}
            with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
                out = mc.apply_mos_calibration(8.5, "video.mp4")
            self.assertFalse(out.available)
            self.assertIn("calibration_error", out.reason or "")
        finally:
            mc._NORMALIZATION_BY_TRANSFER["sdr"] = original

    def test_minimal_unavailable_skips_digest(self) -> None:
        record = mc.minimal_unavailable_mos_record(8.5, "calibration_error:ValueError")
        self.assertNotIn("embedded_coefficients_sha256", record)
        json.dumps(record, allow_nan=False)


class MosAdapterPipelineSafetyTests(unittest.TestCase):
    def test_gate_failure_uses_minimal_record(self) -> None:
        try:
            mc.gate_reviewed_mos_outcome(
                mc.MosCalibrationOutcome(
                    available=True,
                    mos_calibrated_cvqa=3.5,
                    reason=None,
                    domain=mc.MosDomain("pq", 1080, "smpte2084", 1920),
                    calibrator="U0_M7_Bias",
                    canonical_score_input=8.5,
                    policy=mc.MOS_POLICY,
                ),
                _pinned_upstream_payload(),
            )
            mc.mos_calibration_record(
                mc.MosCalibrationOutcome(
                    available=True,
                    mos_calibrated_cvqa=3.5,
                    reason=None,
                    domain=mc.MosDomain("pq", 1080, "smpte2084", 1920),
                    calibrator="U0_M7_Bias",
                    canonical_score_input=8.5,
                    policy=mc.MOS_POLICY,
                ),
                cvqa_upstream=_pinned_upstream_payload(),
            )
        except Exception as exc:
            record = mc.minimal_unavailable_mos_record(8.5, f"calibration_error:{type(exc).__name__}")
            self.assertEqual(record["status"], "unavailable")

    def test_nonfinite_canonical_serializes_null(self) -> None:
        out = mc.MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason="calibration_error:canonical_cvqa_score must be finite",
            domain=None,
            calibrator=None,
            canonical_score_input=float("nan"),
            policy=mc.MOS_POLICY,
        )
        record = mc.mos_calibration_record(out)
        self.assertIsNone(record["canonical_score_input"])
        json.dumps(record, allow_nan=False)


def _subprocess_suite_once(extra_argv: list[str] | None = None) -> int:
    cmd = [sys.executable, *(extra_argv or []), str(SCRIPT_PATH)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print(f"FAIL: {len(result.skipped)} test(s) skipped (skips are not allowed)", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0 if result.wasSuccessful() else 1)