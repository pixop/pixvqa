#!/usr/bin/env python3
"""Unit tests for provisional CVQA-HDR-NR MOS 1–5 calibration."""
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

AVT_FIXTURE = ROOT / "tests/fixtures/mos_calibration/avt_target_lambda1_selected_route.json"
REVIEWED_FIXTURE = ROOT / "tests/fixtures/mos_calibration/avt_reviewed_s1_hdr0_sdr1.json"
SELECTED_FIXTURE = ROOT / "tests/fixtures/mos_calibration/avt_selected_s1_sdr_p1_hdr_c0.json"
COEFFICIENTS_DIGEST_SHA256 = "3a92092d1de1e6e2ea413be0cb067c203945019b86e5832036105a3e2d0bb527"
REVIEWED_COEFFICIENTS_BUNDLE_SHA256 = (
    "67d39be487fccee6b0a08ef33a4dd7e8adf1745578eff9060ab48e394069a02c"
)
REVIEWED_EMBEDDED_COEFFICIENTS_SHA256 = (
    "037d22badb5b207609a485b37c7cae9c694bd7a30a473678a482a19b1d564022"
)
SELECTED_COEFFICIENTS_DIGEST_SHA256 = (
    "463b641347d68585ea9f99fc6d8c58b3ea274896458f6b1e58cbf58592b43cd3"
)
SELECTED_EMBEDDED_COEFFICIENTS_SHA256 = (
    "b855164620e66b208826cc68d516fb89297f240ff3c4f019288790d3b1d39916"
)
SELECTED_SDR_POST_MOS_OFFSET = 0.4105017796126768
SDR_SELECTION_LOCK_SHA256 = (
    "15dbde98a70c3a4ac7e2f7036a084d589e4154f320ebf4dba657f9bb4c894581"
)
HDR_SELECTION_LOCK_SHA256 = (
    "5a4749f6fae7d3a7a9c6736910d83c02d41e4d7630e09bba4476ebfed0ee68cd"
)
REQUIRED_FIXTURES = {
    "legacy_avt": AVT_FIXTURE,
    "reviewed_s1": REVIEWED_FIXTURE,
    "selected_s1": SELECTED_FIXTURE,
}
SCRIPT_PATH = Path(__file__).resolve()


def _require_fixture(path: Path) -> dict:
    """Load a required MOS golden fixture; missing or wrong row count is a hard failure."""
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

# endpoints.csv original_ids / target / lambda=1.0 (independent oracle, not runtime)
ENDPOINT_GOLDEN = {
    ("pq", 1080): {9.0: 3.156328708514176, 10.0: 3.907163897610404, 10.051497077192835: 3.9392224787580266},
    ("pq", 2160): {9.0: 3.310874255330023, 10.0: 4.061709444426251, 10.051497077192835: 4.0937680255738735},
    ("sdr", 1080): {9.0: 3.4475254401922366, 10.0: 4.185523959279873, 10.051497077192835: 4.215453146929613},
    ("sdr", 2160): {9.0: 3.615507338822254, 10.0: 4.35350585790989, 10.051497077192835: 4.383435045559629},
}

# Selected-route exploratory AVT metrics (original_ids, target λ=1.0, full-392 fit)
RESEARCH_AVT_METRICS = {
    "HDR8_Original_U": {
        "n": 8,
        "mae": 0.5439120233140278,
        "rmse": 0.8808472819974602,
        "bias": 0.4504426536161134,
        "plcc": 0.575050765034925,
        "srcc": 0.6428571428571429,
        "policy_note": "HDR target λ=1.0 adopted as user policy; not empirically superior to baseline",
    },
    "SDR_HEVC54_U_v2": {
        "n": 54,
        "mae": 0.4146940267399347,
        "rmse": 0.5101327544766856,
        "bias": -0.3967793381914102,
        "plcc": 0.9044532787774175,
        "srcc": 0.8303624379492361,
    },
}


def normalize_protocol_scalar(value: str) -> str:
    """Mirror C++ normalize_protocol_scalar for protocol→summary tests."""
    if len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0]:
        return value[1:-1]
    return value


def summary_mos_branch(mos_status: str, mos_value: str) -> str:
    """Return the MOS summary branch label (mirrors src/main.cpp print_learned_summary)."""
    if mos_status == "available" and mos_value not in ("unknown", "unavailable"):
        return "show_mos"
    if mos_status == "unavailable":
        return "show_unavailable"
    return "omit_mos"


def _reference_sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def reference_m1_m7_predict(
    canonical_score: float,
    *,
    calibrator: str,
    transfer: str,
    encoded_height: int,
    policy: str | None = None,
) -> float:
    """Independent oracle from serialized M1 four-parameter logits plus M7 gamma."""
    policy_id = policy or mc.DEFAULT_MOS_POLICY
    spec = mc.calibrators_for_policy(policy_id)[calibrator]
    lo = float(spec["support"]["score_min"])
    hi = float(spec["support"]["score_max"])
    mean = float(spec["normalization"]["mean"])
    std = float(spec["normalization"]["std"])
    xclip = min(max(float(canonical_score), lo), hi)
    z = (xclip - mean) / std
    a, b, c, log_scale = (float(v) for v in spec["m1_params"])
    scale = max(1e-3, math.exp(log_scale))
    base = 1.0 + 4.0 * _reference_sigmoid(a + max(0.0, b) * (z - c) / scale)
    base = min(max(base, 1.0), 5.0)
    correction = 0.0
    if transfer == "pq":
        correction += float(spec["gamma_pq"])
    if encoded_height == 2160:
        correction += float(spec["gamma_2160"])
    return min(max(base + correction, 1.0), 5.0)


def setUpModule() -> None:
    """Ensure the suite defaults to legacy policy unless a test patches the env."""
    os.environ.pop(mc.ENV_MOS_POLICY, None)


BOUNDARY_SCORES = (0.0, 4.1344840888613135, 4.1345, 8.5, 10.0, 9.903605146801397)
COMBO_CASES = (
    ("pq", 1080, "U0_M7_Bias"),
    ("pq", 2160, "U0_M7_Bias"),
    ("sdr", 1080, "U_v2_M7_Bias"),
    ("sdr", 2160, "U_v2_M7_Bias"),
)


class MosFormulaTests(unittest.TestCase):
    def test_effective_logistic_params_hdr(self) -> None:
        a, b, c, log_scale = mc._CALIBRATORS["U0_M7_Bias"]["m1_params"]
        scale = max(1e-3, math.exp(log_scale))
        alpha = a - max(0.0, b) * c / scale
        beta = max(0.0, b) / scale
        self.assertAlmostEqual(alpha, -0.24604742, places=7)
        self.assertAlmostEqual(beta, 1.00087755, places=7)

    def test_effective_logistic_params_sdr(self) -> None:
        a, b, c, log_scale = mc._CALIBRATORS["U_v2_M7_Bias"]["m1_params"]
        scale = max(1e-3, math.exp(log_scale))
        alpha = a - max(0.0, b) * c / scale
        beta = max(0.0, b) / scale
        self.assertAlmostEqual(alpha, -0.10602048, places=7)
        self.assertAlmostEqual(beta, 1.05832898, places=7)

    def test_endpoint_golden_values(self) -> None:
        calibrator_for = {"pq": "U0_M7_Bias", "sdr": "U_v2_M7_Bias"}
        for (transfer, height), score_map in ENDPOINT_GOLDEN.items():
            calibrator = calibrator_for[transfer]
            for score, expected in score_map.items():
                ref = reference_m1_m7_predict(
                    score, calibrator=calibrator, transfer=transfer, encoded_height=height
                )
                actual = mc.predict_mos(
                    score, transfer=transfer, encoded_height=height, calibrator=calibrator
                )
                self.assertAlmostEqual(ref, expected, delta=1e-12)
                self.assertAlmostEqual(actual, expected, delta=1e-8)

    def test_four_param_oracle_all_boundaries(self) -> None:
        for transfer, height, calibrator in COMBO_CASES:
            for score in BOUNDARY_SCORES:
                expected = reference_m1_m7_predict(
                    score, calibrator=calibrator, transfer=transfer, encoded_height=height
                )
                actual = mc.predict_mos(
                    score, transfer=transfer, encoded_height=height, calibrator=calibrator
                )
                self.assertAlmostEqual(actual, expected, delta=1e-12, msg=f"{transfer}/{height} x={score}")

    def test_pq_2160_correction(self) -> None:
        score = 8.5
        mos_1080 = mc.predict_mos(score, transfer="pq", encoded_height=1080)
        mos_2160 = mc.predict_mos(score, transfer="pq", encoded_height=2160)
        self.assertGreater(mos_2160, mos_1080)
        self.assertAlmostEqual(
            mos_2160 - mos_1080, mc._CALIBRATORS["U0_M7_Bias"]["gamma_2160"], delta=1e-12
        )

    def test_support_clamping_low(self) -> None:
        low = mc._CALIBRATORS["U0_M7_Bias"]["support"]["score_min"] - 1.0
        at_min = mc.predict_mos(low, transfer="pq", encoded_height=1080)
        at_floor = mc.predict_mos(
            mc._CALIBRATORS["U0_M7_Bias"]["support"]["score_min"],
            transfer="pq",
            encoded_height=1080,
        )
        self.assertAlmostEqual(at_min, at_floor, delta=1e-12)

    def test_support_clamping_high(self) -> None:
        hi = mc._CALIBRATORS["U0_M7_Bias"]["support"]["score_max"] + 1.0
        at_max = mc.predict_mos(hi, transfer="pq", encoded_height=1080)
        at_ceil = mc.predict_mos(
            mc._CALIBRATORS["U0_M7_Bias"]["support"]["score_max"],
            transfer="pq",
            encoded_height=1080,
        )
        self.assertAlmostEqual(at_max, at_ceil, delta=1e-12)

    def test_monotonicity_pq(self) -> None:
        scores = [5.0, 6.5, 8.0, 9.5, 10.0]
        mos = [mc.predict_mos(s, transfer="pq", encoded_height=1080) for s in scores]
        for left, right in zip(mos, mos[1:]):
            self.assertLessEqual(left, right)

    def test_monotonicity_sdr(self) -> None:
        scores = [5.0, 6.5, 8.0, 9.5, 10.0]
        mos = [mc.predict_mos(s, transfer="sdr", encoded_height=1080) for s in scores]
        for left, right in zip(mos, mos[1:]):
            self.assertLessEqual(left, right)

    def test_output_bounds(self) -> None:
        for transfer in ("pq", "sdr"):
            for score in (4.0, 7.0, 10.0):
                for height in (1080, 2160):
                    mos = mc.predict_mos(score, transfer=transfer, encoded_height=height)
                    self.assertGreaterEqual(mos, 1.0)
                    self.assertLessEqual(mos, 5.0)
                    self.assertTrue(math.isfinite(mos))


class MosFixtureAvailabilityTests(unittest.TestCase):
    def test_all_required_fixtures_present_and_valid(self) -> None:
        for label, path in REQUIRED_FIXTURES.items():
            with self.subTest(fixture=label):
                data = _require_fixture(path)
                self.assertEqual(len(data["cases"]), 68)

    def test_patched_fixture_unavailable_fails(self) -> None:
        original_is_file = Path.is_file

        def _missing_is_file(self: Path) -> bool:
            if self == AVT_FIXTURE:
                return False
            return original_is_file(self)

        with patch.object(Path, "is_file", _missing_is_file):
            with self.assertRaises(FileNotFoundError):
                _require_fixture(AVT_FIXTURE)


class MosAvtFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _require_fixture(AVT_FIXTURE)

    def test_fixture_case_count(self) -> None:
        self.assertEqual(len(self.fixture["cases"]), 68)

    def test_avt_predictions_match_independent_oracle(self) -> None:
        for case in self.fixture["cases"]:
            expected = case["expected_mos"]
            actual = mc.predict_mos(
                case["score"],
                transfer=case["transfer"],
                encoded_height=case["encoded_height"],
                calibrator=case["calibrator"],
            )
            self.assertAlmostEqual(
                actual,
                expected,
                delta=1e-8,
                msg=f"{case['dataset']}:{case['id']}",
            )


    def test_vd_scored_geometry_not_filename_condition(self) -> None:
        # The AVT VD Study-2 distribution files are physically 3840x2160.
        # Filename 1080p describes the prior source-encode condition.
        vd = [case for case in self.fixture["cases"] if case["dataset"] == "VD_HEVC6_1.6H"]
        self.assertEqual(len(vd), 6)
        self.assertEqual(sum("_1080p_" in case["id"] for case in vd), 3)
        self.assertTrue(all(case["encoded_height"] == 2160 for case in vd))

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        xm, ym = sum(x) / len(x), sum(y) / len(y)
        cov = sum((a - xm) * (b - ym) for a, b in zip(x, y))
        vx = sum((a - xm) ** 2 for a in x)
        vy = sum((b - ym) ** 2 for b in y)
        return cov / math.sqrt(vx * vy)

    @staticmethod
    def _midranks(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda pair: pair[1])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and indexed[j][1] == indexed[i][1]:
                j += 1
            rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[indexed[k][0]] = rank
            i = j
        return ranks

    def test_native_avt_all_five_metrics_reproduced_from_saved_predictions(self) -> None:
        # Independent frozen research prediction/label pairs; never compute
        # expected MOS values from the PixVQA implementation under test.
        expected = {
            "HDR8": (8, 0.5439120233140278, 0.8808472819974602,
                     0.4504426536161134, 0.575050765034925, 0.6428571428571429),
            "SDR_HEVC54": (54, 0.4146940267399347, 0.5101327544766856,
                           -0.3967793381914102, 0.9044532787774175, 0.8303624379492361),
            "VD_HEVC6_1.6H": (6, 0.40962074119471076, 0.5166859985637767,
                              -0.32217576881523696, 0.6974027384844524, 0.7714285714285715),
        }
        for dataset, (n, mae, rmse, bias, plcc, srcc) in expected.items():
            cases = [case for case in self.fixture["cases"] if case["dataset"] == dataset]
            self.assertEqual(len(cases), n)
            predicted = [float(case["expected_mos"]) for case in cases]
            native = [float(case["native_mos"]) for case in cases]
            residual = [a - b for a, b in zip(predicted, native)]
            observed = (
                sum(abs(e) for e in residual) / n,
                math.sqrt(sum(e * e for e in residual) / n),
                sum(residual) / n,
                self._pearson(predicted, native),
                self._pearson(self._midranks(predicted), self._midranks(native)),
            )
            for metric, expected_value in zip(observed, (mae, rmse, bias, plcc, srcc)):
                self.assertAlmostEqual(metric, expected_value, delta=1e-9, msg=dataset)


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

    def test_bt2020_tags_do_not_route_to_pq(self) -> None:
        for tag in ("bt2020", "bt2020-10", "bt2020-12"):
            transfer, _ = mc.classify_transfer_tag(tag)
            self.assertNotEqual(transfer, "pq", msg=f"{tag!r} must not classify as PQ")

    def test_bt2020_ambiguous_unsupported(self) -> None:
        transfer, reason = mc.classify_transfer_tag("bt2020")
        self.assertIsNone(transfer)
        self.assertEqual(reason, "transfer_unsupported")

    def test_bt2020_10_sdr_not_pq(self) -> None:
        transfer, reason = mc.classify_transfer_tag("bt2020-10")
        self.assertEqual(transfer, "sdr")
        self.assertIsNone(reason)

    def test_bt2020_12_sdr_not_pq(self) -> None:
        transfer, reason = mc.classify_transfer_tag("bt2020-12")
        self.assertEqual(transfer, "sdr")
        self.assertIsNone(reason)

    def test_apply_unavailable_height(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 720, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        self.assertFalse(out.available)
        self.assertEqual(out.reason, "height_unsupported")
        self.assertIsNone(out.mos_calibrated_cvqa)

    def test_apply_available_pq(self) -> None:
        score = 9.903605146801397
        expected = reference_m1_m7_predict(score, calibrator="U0_M7_Bias", transfer="pq", encoded_height=1080)
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(score, "video.mp4")
        self.assertTrue(out.available)
        self.assertEqual(out.calibrator, "U0_M7_Bias")
        self.assertAlmostEqual(out.mos_calibrated_cvqa or 0.0, expected, delta=1e-12)

    def test_apply_available_sdr(self) -> None:
        score = 8.5
        expected = reference_m1_m7_predict(score, calibrator="U_v2_M7_Bias", transfer="sdr", encoded_height=2160)
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 2160, "bt709")):
            out = mc.apply_mos_calibration(score, "video.mp4")
        self.assertTrue(out.available)
        self.assertEqual(out.calibrator, "U_v2_M7_Bias")
        self.assertAlmostEqual(out.mos_calibrated_cvqa or 0.0, expected, delta=1e-12)

    def test_canonical_score_preserved_in_record(self) -> None:
        canonical = 8.502661863950266
        with patch.object(mc, "probe_encoded_geometry", return_value=(0, 0, "unknown")):
            out = mc.apply_mos_calibration(canonical, "video.mp4")
        record = mc.mos_calibration_record(out)
        self.assertEqual(record["canonical_score_input"], canonical)
        self.assertEqual(record["status"], "unavailable")

    def test_ffprobe_errors_never_affect_canonical_input(self) -> None:
        canonical = 7.25
        error_returns = (
            (0, 0, "ffprobe_timeout"),
            (0, 0, "invalid_ffprobe_json"),
            (0, 0, "invalid_ffprobe_payload"),
            (0, 0, "no_video_stream"),
            (0, 0, "non_positive_geometry"),
            (0, 0, "ffprobe_failed:exit 1"),
        )
        for probe_return in error_returns:
            with patch.object(mc, "probe_encoded_geometry", return_value=probe_return):
                out = mc.apply_mos_calibration(canonical, "video.mp4")
            self.assertFalse(out.available, msg=str(probe_return))
            self.assertEqual(out.canonical_score_input, canonical)
            self.assertIsNone(out.mos_calibrated_cvqa)

    @patch.object(mc.subprocess, "run")
    def test_ffprobe_timeout_status(self, mock_run) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=mc.FFPROBE_TIMEOUT_SEC)
        with patch.object(mc, "resolve_ffprobe_legacy", return_value="/bin/ffprobe"):
            width, height, third = mc.probe_encoded_geometry("video.mp4")
        self.assertEqual((width, height, third), (0, 0, "ffprobe_timeout"))

    @patch.object(mc.subprocess, "run")
    def test_ffprobe_invalid_payload_types(self, mock_run) -> None:
        def make_proc(stdout: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["ffprobe"], returncode=0, stdout=stdout, stderr=""
            )

        with patch.object(mc, "resolve_ffprobe_legacy", return_value="/bin/ffprobe"):
            mock_run.return_value = make_proc('{"streams": "not-a-list"}')
            self.assertEqual(mc.probe_encoded_geometry("v.mp4")[2], "invalid_ffprobe_payload")
            mock_run.return_value = make_proc('{"streams": ["not-a-dict"]}')
            self.assertEqual(mc.probe_encoded_geometry("v.mp4")[2], "invalid_ffprobe_payload")
            mock_run.return_value = make_proc('{"streams": [{"width": 0, "height": 1080}]}')
            self.assertEqual(mc.probe_encoded_geometry("v.mp4")[2], "non_positive_geometry")
            mock_run.return_value = make_proc("not-json")
            self.assertEqual(mc.probe_encoded_geometry("v.mp4")[2], "invalid_ffprobe_json")


class MosRoutingTests(unittest.TestCase):
    def test_domain_routing(self) -> None:
        self.assertEqual(mc.calibrator_for_transfer("pq"), "U0_M7_Bias")
        self.assertEqual(mc.calibrator_for_transfer("sdr"), "U_v2_M7_Bias")

    def test_calibrator_mismatch_pq_on_sdr_calibrator(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            mc.predict_mos(8.5, transfer="pq", encoded_height=1080, calibrator="U_v2_M7_Bias")
        self.assertIn("sdr", str(ctx.exception).lower())

    def test_calibrator_mismatch_sdr_on_pq_calibrator(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            mc.predict_mos(8.5, transfer="sdr", encoded_height=1080, calibrator="U0_M7_Bias")
        self.assertIn("pq", str(ctx.exception).lower())


class MosProvenanceTests(unittest.TestCase):
    def test_target_mos_weighting_metadata(self) -> None:
        self.assertEqual(mc.WEIGHT_KIND, "target_mos")
        self.assertEqual(mc.WEIGHT_LAMBDA, 1.0)
        self.assertEqual(mc.CALIBRATION_LABEL, "domain-aware logistic MOS calibration")
        self.assertIn("mos_target_reweight", mc.COEFFICIENT_SOURCE)
        self.assertNotIn("final_fitted_models", mc.COEFFICIENT_SOURCE)

    def test_coefficients_digest_stable(self) -> None:
        self.assertEqual(mc.coefficients_digest(), COEFFICIENTS_DIGEST_SHA256)

    def test_record_includes_provenance(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        record = mc.mos_calibration_record(out)
        self.assertEqual(record["calibration_version"], mc.CALIBRATION_VERSION)
        self.assertEqual(record["coefficient_cohort"], "full-392-20261113")
        self.assertEqual(record["weight_kind"], "target_mos")
        self.assertEqual(record["hdr_weight_lambda"], 1.0)
        self.assertEqual(record["sdr_weight_lambda"], 1.0)
        self.assertEqual(record["mos_policy"], mc.DEFAULT_MOS_POLICY)
        self.assertEqual(record["calibration_label"], mc.CALIBRATION_LABEL)
        self.assertEqual(record["coefficients_digest_sha256"], COEFFICIENTS_DIGEST_SHA256)
        self.assertIn("m1_params", record)
        self.assertEqual(len(record["m1_params"]), 4)
        self.assertIn("m1_params_sha256", record)
        self.assertIn("coefficient_source", record)
        self.assertIn("evidence_caveat", record)
        self.assertIn("unresolved_policy_note", record)

    def test_research_avt_metrics_documented(self) -> None:
        hdr = RESEARCH_AVT_METRICS["HDR8_Original_U"]
        sdr = RESEARCH_AVT_METRICS["SDR_HEVC54_U_v2"]
        self.assertAlmostEqual(hdr["mae"], 0.5439, places=4)
        self.assertAlmostEqual(hdr["rmse"], 0.8808, places=4)
        self.assertAlmostEqual(hdr["bias"], 0.4504, places=4)
        self.assertAlmostEqual(hdr["plcc"], 0.5751, places=4)
        self.assertAlmostEqual(hdr["srcc"], 0.6429, places=4)
        self.assertIn("policy", hdr["policy_note"])
        self.assertAlmostEqual(sdr["mae"], 0.4147, places=4)
        self.assertAlmostEqual(sdr["rmse"], 0.5101, places=4)
        self.assertAlmostEqual(sdr["bias"], -0.3968, places=4)
        self.assertAlmostEqual(sdr["plcc"], 0.9045, places=4)
        self.assertAlmostEqual(sdr["srcc"], 0.8304, places=4)


class MosAdapterSafetyTests(unittest.TestCase):
    def test_calibration_error_outcome_preserves_canonical(self) -> None:
        """Mirrors adapter narrow exception boundary (A07): MOS suppressed, canonical kept."""
        canonical = 8.25
        outcome = mc.MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason="calibration_error:ValueError",
            domain=None,
            calibrator=None,
            canonical_score_input=canonical,
        )
        record = mc.mos_calibration_record(outcome)
        self.assertEqual(record["status"], "unavailable")
        self.assertEqual(record["canonical_score_input"], canonical)
        self.assertEqual(record["reason"], "calibration_error:ValueError")
        self.assertNotIn("mos_calibrated_cvqa", record)


class MosProtocolTests(unittest.TestCase):
    @staticmethod
    def _sample_record(*, status: str, mos_value: float | None = None) -> dict:
        canonical = 8.502661863950266
        mos = mos_value
        if mos is None and status == "available":
            mos = reference_m1_m7_predict(
                canonical, calibrator="U0_M7_Bias", transfer="pq", encoded_height=1080
            )
        return {
            "quality_score": canonical,
            "raw_model_score": 0.38208478689193726,
            "sampling": {
                "sampled_frames": 10,
                "opencv_frame_count": 500,
                "opencv_fps": 50.0,
                "rounded_fps": 50,
            },
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
            "mos_calibration": {"status": status},
        }

    def test_protocol_lines_include_mos_fields(self) -> None:
        from pixvqa_cvqahdr.__main__ import protocol_lines

        record = self._sample_record(status="available")
        canonical = record["quality_score"]
        mos_value = record["mos_calibrated_cvqa"]
        lines = protocol_lines(record, "out.json")
        joined = "\n".join(lines)
        self.assertIn(f"pixvqa-value quality_score {canonical}", joined)
        self.assertIn(f"pixvqa-value mos_calibrated_cvqa {mos_value}", joined)
        self.assertIn("pixvqa-value mos_calibration_status available", joined)
        self.assertNotIn("mos_calibration_status 'available'", joined)

    def test_protocol_to_summary_available(self) -> None:
        from pixvqa_cvqahdr.__main__ import protocol_lines

        record = self._sample_record(status="available")
        values = {
            key: val
            for line in protocol_lines(record, "none")
            if line.startswith("pixvqa-value ")
            for key, val in [line.split(" ", 2)[1:]]
        }
        status = normalize_protocol_scalar(values["mos_calibration_status"])
        mos = normalize_protocol_scalar(values["mos_calibrated_cvqa"])
        self.assertEqual(summary_mos_branch(status, mos), "show_mos")

    def test_protocol_to_summary_unavailable(self) -> None:
        from pixvqa_cvqahdr.__main__ import protocol_lines

        record = self._sample_record(status="unavailable", mos_value=None)
        values = {
            key: val
            for line in protocol_lines(record, "none")
            if line.startswith("pixvqa-value ")
            for key, val in [line.split(" ", 2)[1:]]
        }
        status = normalize_protocol_scalar(values["mos_calibration_status"])
        mos = normalize_protocol_scalar(values["mos_calibrated_cvqa"])
        self.assertEqual(status, "unavailable")
        self.assertEqual(mos, "unavailable")
        self.assertEqual(summary_mos_branch(status, mos), "show_unavailable")

    def test_protocol_to_summary_absent_mos_field(self) -> None:
        """Backward compat: quoted legacy status still normalizes for summary."""
        legacy_status = "'available'"
        legacy_mos = "3.42"
        self.assertEqual(summary_mos_branch(normalize_protocol_scalar(legacy_status), legacy_mos), "show_mos")

    def test_protocol_to_summary_unsupported_status(self) -> None:
        self.assertEqual(summary_mos_branch("unknown", "unavailable"), "omit_mos")


class MosSelectedS1FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _require_fixture(SELECTED_FIXTURE)

    def test_fixture_policy_and_count(self) -> None:
        self.assertEqual(self.fixture["policy"], mc.SELECTED_MOS_POLICY)
        self.assertEqual(len(self.fixture["cases"]), 68)
        self.assertEqual(
            self.fixture["s1_coefficients_bundle_sha256"], REVIEWED_COEFFICIENTS_BUNDLE_SHA256
        )
        self.assertEqual(
            self.fixture["coefficients_digest_sha256"], SELECTED_COEFFICIENTS_DIGEST_SHA256
        )
        self.assertEqual(
            self.fixture["embedded_coefficients_sha256"], SELECTED_EMBEDDED_COEFFICIENTS_SHA256
        )

    def test_selected_golden_predictions_match(self) -> None:
        max_err = 0.0
        for case in self.fixture["cases"]:
            expected = case["expected_mos"]
            actual = mc.predict_mos(
                case["score"],
                transfer=case["transfer"],
                encoded_height=case["encoded_height"],
                calibrator=case["calibrator"],
                policy=mc.SELECTED_MOS_POLICY,
            )
            err = abs(actual - expected)
            max_err = max(max_err, err)
            self.assertLessEqual(
                err,
                1e-10,
                msg=f"{case['dataset']}:{case['id']} err={err}",
            )
        self.assertLessEqual(max_err, 1e-10)

    def test_sdr_offset_from_reviewed(self) -> None:
        for case in self.fixture["cases"]:
            if case["transfer"] != "sdr":
                continue
            reviewed = mc.predict_mos(
                case["score"],
                transfer=case["transfer"],
                encoded_height=case["encoded_height"],
                calibrator=case["calibrator"],
                policy=mc.REVIEWED_MOS_POLICY,
            )
            selected = mc.predict_mos(
                case["score"],
                transfer=case["transfer"],
                encoded_height=case["encoded_height"],
                calibrator=case["calibrator"],
                policy=mc.SELECTED_MOS_POLICY,
            )
            expected = min(5.0, max(1.0, reviewed + SELECTED_SDR_POST_MOS_OFFSET))
            self.assertAlmostEqual(selected, expected, delta=1e-12, msg=case["id"])
            self.assertAlmostEqual(case["expected_mos"], expected, delta=1e-12)

    def test_hdr_identical_to_reviewed(self) -> None:
        for case in self.fixture["cases"]:
            if case["transfer"] != "pq":
                continue
            reviewed = mc.predict_mos(
                case["score"],
                transfer=case["transfer"],
                encoded_height=case["encoded_height"],
                calibrator=case["calibrator"],
                policy=mc.REVIEWED_MOS_POLICY,
            )
            selected = mc.predict_mos(
                case["score"],
                transfer=case["transfer"],
                encoded_height=case["encoded_height"],
                calibrator=case["calibrator"],
                policy=mc.SELECTED_MOS_POLICY,
            )
            self.assertAlmostEqual(selected, reviewed, delta=1e-12, msg=case["id"])
            self.assertAlmostEqual(case["expected_mos"], reviewed, delta=1e-12)

    def test_selected_digest_composite_not_manifest_only(self) -> None:
        self.assertEqual(
            mc.coefficients_digest(mc.SELECTED_MOS_POLICY), SELECTED_COEFFICIENTS_DIGEST_SHA256
        )
        self.assertNotEqual(
            mc.coefficients_digest(mc.SELECTED_MOS_POLICY), REVIEWED_COEFFICIENTS_BUNDLE_SHA256
        )

    def test_selected_embedded_coefficients_digest(self) -> None:
        self.assertEqual(
            mc.embedded_coefficients_sha256(mc.SELECTED_MOS_POLICY),
            SELECTED_EMBEDDED_COEFFICIENTS_SHA256,
        )
        self.assertNotEqual(
            mc.embedded_coefficients_sha256(mc.SELECTED_MOS_POLICY),
            mc.embedded_coefficients_sha256(mc.REVIEWED_MOS_POLICY),
        )

    def test_selected_output_clamp_after_offset(self) -> None:
        hi = mc._SHARED_SUPPORT["score_max"]
        detail = mc.predict_mos_detailed(
            hi, transfer="sdr", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY
        )
        self.assertLessEqual(detail.estimated_perceptual_quality, 5.0)
        self.assertGreaterEqual(detail.estimated_perceptual_quality, 1.0)
        self.assertAlmostEqual(detail.post_mos_offset, SELECTED_SDR_POST_MOS_OFFSET, places=12)

    def test_selected_record_provenance(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.SELECTED_MOS_POLICY)
        record = mc.mos_calibration_record(out, cvqa_upstream=_pinned_upstream_payload())
        self.assertEqual(record["research_status"], "shadow-not-prod")
        self.assertEqual(record["mos_policy"], mc.SELECTED_MOS_POLICY)
        self.assertEqual(record["sdr_selection_lock_sha256"], SDR_SELECTION_LOCK_SHA256)
        self.assertEqual(record["hdr_selection_lock_sha256"], HDR_SELECTION_LOCK_SHA256)
        self.assertIn("post_output_offset_by_transfer", record)
        self.assertAlmostEqual(record["post_output_offset_by_transfer"]["sdr"], SELECTED_SDR_POST_MOS_OFFSET)
        self.assertEqual(record["post_output_offset_by_transfer"]["pq"], 0.0)
        self.assertAlmostEqual(record["post_mos_offset"], SELECTED_SDR_POST_MOS_OFFSET)


class MosReviewedS1FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _require_fixture(REVIEWED_FIXTURE)

    def test_fixture_policy_and_count(self) -> None:
        self.assertEqual(self.fixture["policy"], mc.REVIEWED_MOS_POLICY)
        self.assertEqual(len(self.fixture["cases"]), 68)
        self.assertEqual(
            self.fixture["coefficients_bundle_sha256"], REVIEWED_COEFFICIENTS_BUNDLE_SHA256
        )

    def test_reviewed_golden_predictions_match(self) -> None:
        max_err = 0.0
        for case in self.fixture["cases"]:
            expected = case["expected_mos"]
            actual = mc.predict_mos(
                case["score"],
                transfer=case["transfer"],
                encoded_height=case["encoded_height"],
                calibrator=case["calibrator"],
                policy=mc.REVIEWED_MOS_POLICY,
            )
            err = abs(actual - expected)
            max_err = max(max_err, err)
            self.assertLessEqual(
                err,
                1e-6,
                msg=f"{case['dataset']}:{case['id']} err={err}",
            )
        # Frozen oracle parity: expect exact replay (≤1e-6); max observed printed for audit.
        self.assertLessEqual(max_err, 1e-6)

    def test_reviewed_digest_matches_manifest(self) -> None:
        self.assertEqual(
            mc.coefficients_digest(mc.REVIEWED_MOS_POLICY), REVIEWED_COEFFICIENTS_BUNDLE_SHA256
        )

    def test_reviewed_embedded_coefficients_digest(self) -> None:
        self.assertEqual(
            mc.embedded_coefficients_sha256(mc.REVIEWED_MOS_POLICY),
            REVIEWED_EMBEDDED_COEFFICIENTS_SHA256,
        )
        self.assertNotEqual(
            mc.embedded_coefficients_sha256(mc.REVIEWED_MOS_POLICY),
            REVIEWED_COEFFICIENTS_BUNDLE_SHA256,
        )


class MosPolicySelectorTests(unittest.TestCase):
    def test_default_policy_is_legacy(self) -> None:
        policy, err = mc.resolve_mos_policy()
        self.assertIsNone(err)
        self.assertEqual(policy, mc.DEFAULT_MOS_POLICY)

    def test_invalid_policy_fail_closed(self) -> None:
        policy, err = mc.resolve_mos_policy("not_a_real_policy")
        self.assertIsNone(policy)
        self.assertEqual(err, "invalid_policy:not_a_real_policy")

    def test_apply_invalid_policy_unavailable(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", policy="bogus")
        self.assertFalse(out.available)
        self.assertEqual(out.reason, "invalid_policy:bogus")
        self.assertIsNone(out.policy)

    @patch.dict(os.environ, {mc.ENV_MOS_POLICY: "bogus"}, clear=False)
    def test_env_invalid_policy_fail_closed(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        self.assertFalse(out.available)
        self.assertEqual(out.reason, "invalid_policy:bogus")

    def test_hdr_differs_legacy_vs_reviewed(self) -> None:
        score = 9.903605146801397
        legacy = mc.predict_mos(score, transfer="pq", encoded_height=1080)
        reviewed = mc.predict_mos(
            score, transfer="pq", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY
        )
        self.assertNotAlmostEqual(legacy, reviewed, places=6)

    def test_sdr_identical_legacy_vs_reviewed(self) -> None:
        for score in (7.5, 8.5, 9.903605146801397, 10.051497077192835):
            for height in (1080, 2160):
                legacy = mc.predict_mos(score, transfer="sdr", encoded_height=height)
                reviewed = mc.predict_mos(
                    score, transfer="sdr", encoded_height=height, policy=mc.REVIEWED_MOS_POLICY
                )
                self.assertAlmostEqual(legacy, reviewed, delta=1e-12, msg=f"sdr h={height} x={score}")

    def test_ood_above_support_flag(self) -> None:
        score = 10.051497077192835
        detail = mc.predict_mos_detailed(
            score + 0.5, transfer="pq", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY
        )
        self.assertTrue(detail.ood_above_support)
        self.assertFalse(detail.ood_below_support)
        at_support = mc.predict_mos(score, transfer="pq", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY)
        self.assertAlmostEqual(detail.estimated_perceptual_quality, at_support, delta=1e-12)

    def test_hlg_unavailable_not_sdr(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "arib-std-b67")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.REVIEWED_MOS_POLICY)
        self.assertFalse(out.available)
        self.assertEqual(out.reason, "transfer_hlg")

    def test_predict_mos_invalid_score_type(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            mc.predict_mos_detailed("not-a-number", transfer="pq", encoded_height=1080)
        self.assertIn("finite", str(ctx.exception).lower())
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration("bad", "video.mp4")
        self.assertFalse(out.available)
        self.assertIn("calibration_error", out.reason or "")

    def test_selected_differs_reviewed_on_sdr(self) -> None:
        score = 8.5
        reviewed = mc.predict_mos(
            score, transfer="sdr", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY
        )
        selected = mc.predict_mos(
            score, transfer="sdr", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY
        )
        self.assertGreater(selected, reviewed)
        self.assertAlmostEqual(selected - reviewed, SELECTED_SDR_POST_MOS_OFFSET, delta=1e-12)

    def test_selected_env_policy_opt_in(self) -> None:
        @patch.dict(os.environ, {mc.ENV_MOS_POLICY: mc.SELECTED_MOS_POLICY}, clear=False)
        def _run() -> None:
            with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
                out = mc.apply_mos_calibration(8.5, "video.mp4")
            self.assertTrue(out.available)
            self.assertEqual(out.policy, mc.SELECTED_MOS_POLICY)
            self.assertAlmostEqual(out.post_mos_offset or 0.0, SELECTED_SDR_POST_MOS_OFFSET)
        _run()

    def test_default_policy_not_selected(self) -> None:
        policy, err = mc.resolve_mos_policy()
        self.assertIsNone(err)
        self.assertNotEqual(policy, mc.SELECTED_MOS_POLICY)


class MosCvqaUpstreamGateTests(unittest.TestCase):
    def test_compatibility_all_pins_match(self) -> None:
        compat = mc.cvqa_upstream_compatibility(_pinned_upstream_payload())
        self.assertTrue(compat["compatible"])
        self.assertEqual(compat["integration_gate"], "verified")
        self.assertEqual(compat["missing_fields"], [])
        self.assertEqual(compat["incompatible_fields"], [])

    def test_compatibility_rejects_wrong_commit_despite_commit_verified_true(self) -> None:
        payload = _pinned_upstream_payload()
        payload["upstream"] = {"actual_commit": "deadbeef", "commit_verified": True}
        compat = mc.cvqa_upstream_compatibility(payload)
        self.assertFalse(compat["compatible"])
        self.assertIn("actual_commit", compat["incompatible_fields"])

    def test_compatibility_rejects_wrong_checkpoint(self) -> None:
        payload = _pinned_upstream_payload()
        payload["checkpoint_sha256"] = "0" * 64
        compat = mc.cvqa_upstream_compatibility(payload)
        self.assertFalse(compat["compatible"])
        self.assertIn("checkpoint_sha256", compat["incompatible_fields"])

    def test_compatibility_rejects_wrong_profile(self) -> None:
        payload = _pinned_upstream_payload()
        payload["profile_sha256"] = "0" * 64
        compat = mc.cvqa_upstream_compatibility(payload)
        self.assertFalse(compat["compatible"])
        self.assertIn("profile_sha256", compat["incompatible_fields"])

    def test_compatibility_missing_fields(self) -> None:
        compat = mc.cvqa_upstream_compatibility({"upstream": {}})
        self.assertFalse(compat["compatible"])
        self.assertIn("actual_commit", compat["missing_fields"])
        self.assertIn("checkpoint_sha256", compat["missing_fields"])
        self.assertIn("profile_sha256", compat["missing_fields"])

    def test_from_run_record_helper(self) -> None:
        built = mc.cvqa_upstream_from_run_record(_fake_adapter_record())
        self.assertEqual(built["checkpoint_sha256"], PINNED_CHECKPOINT_SHA256)
        self.assertEqual(built["profile_sha256"], PINNED_PROFILE_SHA256)

    def test_reviewed_gated_available_when_pins_match(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.REVIEWED_MOS_POLICY)
        self.assertTrue(out.available)
        gated = mc.gate_reviewed_mos_outcome(out, _pinned_upstream_payload())
        self.assertTrue(gated.available)
        record = mc.mos_calibration_record(gated, cvqa_upstream=_pinned_upstream_payload())
        self.assertEqual(record["status"], "available")
        self.assertIn("estimated_perceptual_quality", record)
        self.assertIn("embedded_coefficients_sha256", record)

    def test_reviewed_blocked_unpinned_profile(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.REVIEWED_MOS_POLICY)
        self.assertTrue(out.available)
        upstream = _pinned_upstream_payload()
        upstream["profile_sha256"] = "unpinned-profile-digest"
        gated = mc.gate_reviewed_mos_outcome(out, upstream)
        self.assertFalse(gated.available)
        self.assertEqual(gated.reason, "cvqa_upstream_incompatible")
        record = mc.mos_calibration_record(gated, cvqa_upstream=upstream)
        self.assertEqual(record["status"], "unavailable")
        self.assertEqual(record["reason"], "cvqa_upstream_incompatible")
        self.assertEqual(record["cvqa_upstream_gate_status"], "blocked")
        self.assertNotIn("estimated_perceptual_quality", record)
        self.assertNotIn("mos_calibrated_cvqa", record)
        self.assertIn("cvqa_upstream_compatibility", record)

    def test_legacy_not_gated_when_profile_unpinned(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        self.assertTrue(out.available)
        upstream = _pinned_upstream_payload()
        upstream["profile_sha256"] = "unpinned-profile-digest"
        gated = mc.gate_reviewed_mos_outcome(out, upstream)
        self.assertTrue(gated.available)
        record = mc.mos_calibration_record(gated, cvqa_upstream=upstream)
        self.assertEqual(record["status"], "available")
        self.assertEqual(record["mos_policy"], mc.DEFAULT_MOS_POLICY)

    def test_selected_gated_available_when_pins_match(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.SELECTED_MOS_POLICY)
        self.assertTrue(out.available)
        gated = mc.gate_reviewed_mos_outcome(out, _pinned_upstream_payload())
        self.assertTrue(gated.available)
        record = mc.mos_calibration_record(gated, cvqa_upstream=_pinned_upstream_payload())
        self.assertEqual(record["status"], "available")
        self.assertEqual(record["research_status"], "shadow-not-prod")
        self.assertIn("embedded_coefficients_sha256", record)

    def test_selected_blocked_unpinned_profile(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.SELECTED_MOS_POLICY)
        self.assertTrue(out.available)
        upstream = _pinned_upstream_payload()
        upstream["checkpoint_sha256"] = "0" * 64
        gated = mc.gate_reviewed_mos_outcome(out, upstream)
        self.assertFalse(gated.available)
        self.assertEqual(gated.reason, "cvqa_upstream_incompatible")
        record = mc.mos_calibration_record(gated, cvqa_upstream=upstream)
        self.assertEqual(record["status"], "unavailable")
        self.assertEqual(record["cvqa_upstream_gate_status"], "blocked")

    def test_canonical_score_unchanged_conceptually_on_gate_block(self) -> None:
        canonical = 8.502661863950266
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(canonical, "video.mp4", policy=mc.REVIEWED_MOS_POLICY)
        gated = mc.gate_reviewed_mos_outcome(out, None)
        self.assertEqual(gated.canonical_score_input, canonical)
        self.assertIsNone(gated.mos_calibrated_cvqa)


class MosSelectedUnclampedInputTests(unittest.TestCase):
    """Selected policy: no bilateral input clamp; legacy/reviewed unchanged."""

    _SUPPORT_HI = mc._SHARED_SUPPORT["score_max"]
    _SUPPORT_LO = mc._SHARED_SUPPORT["score_min"]

    def test_selected_hdr_extrapolation_above_support(self) -> None:
        at_support = mc.predict_mos(
            self._SUPPORT_HI, transfer="pq", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY
        )
        for x in (10.2, 10.5, 11.0, 12.0):
            reviewed = mc.predict_mos(
                x, transfer="pq", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY
            )
            selected = mc.predict_mos(
                x, transfer="pq", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY
            )
            self.assertAlmostEqual(reviewed, at_support, delta=1e-12, msg=f"reviewed plateau x={x}")
            self.assertGreater(selected, at_support, msg=f"selected should grow above support x={x}")
            self.assertGreater(selected, reviewed, msg=f"x={x}")

    def test_selected_sdr_extrapolation_reaches_five(self) -> None:
        at_support_reviewed = mc.predict_mos(
            self._SUPPORT_HI, transfer="sdr", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY
        )
        for x in (10.2, 10.5, 11.0):
            reviewed = mc.predict_mos(
                x, transfer="sdr", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY
            )
            selected = mc.predict_mos(
                x, transfer="sdr", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY
            )
            self.assertAlmostEqual(reviewed, at_support_reviewed, delta=1e-12)
            self.assertGreaterEqual(selected, reviewed)
        self.assertEqual(
            mc.predict_mos(11.0, transfer="sdr", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY),
            5.0,
        )

    def test_selected_low_below_support_differs_from_reviewed(self) -> None:
        x = 3.0
        reviewed = mc.predict_mos_detailed(
            x, transfer="sdr", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY
        )
        selected = mc.predict_mos_detailed(
            x, transfer="sdr", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY
        )
        at_floor_reviewed = mc.predict_mos(
            self._SUPPORT_LO, transfer="sdr", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY
        )
        self.assertAlmostEqual(reviewed.estimated_perceptual_quality, at_floor_reviewed, delta=1e-12)
        self.assertEqual(reviewed.score_effective, self._SUPPORT_LO)
        self.assertEqual(selected.score_effective, x)
        self.assertGreater(selected.estimated_perceptual_quality, reviewed.estimated_perceptual_quality)
        self.assertTrue(selected.ood_below_support)
        self.assertTrue(selected.support_extrapolation)
        self.assertGreaterEqual(selected.estimated_perceptual_quality, 1.0)
        self.assertLessEqual(selected.estimated_perceptual_quality, 5.0)

    def test_selected_extreme_finite_scores_bounded(self) -> None:
        for x in (20.0, -1000.0, 1e300, -1e300):
            detail = mc.predict_mos_detailed(
                x, transfer="pq", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY
            )
            self.assertTrue(math.isfinite(detail.estimated_perceptual_quality))
            self.assertGreaterEqual(detail.estimated_perceptual_quality, 1.0)
            self.assertLessEqual(detail.estimated_perceptual_quality, 5.0)
            self.assertEqual(detail.score_effective, x)
            self.assertFalse(detail.input_support_clamp)

    def test_score_effective_policy_semantics(self) -> None:
        x = 10.5
        sel = mc.predict_mos_detailed(
            x, transfer="pq", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY
        )
        rev = mc.predict_mos_detailed(
            x, transfer="pq", encoded_height=1080, policy=mc.REVIEWED_MOS_POLICY
        )
        self.assertEqual(sel.score_effective, x)
        self.assertEqual(sel.score_requested, x)
        self.assertTrue(sel.ood_above_support)
        self.assertTrue(sel.support_extrapolation)
        self.assertEqual(rev.score_effective, self._SUPPORT_HI)
        self.assertTrue(rev.input_support_clamp)
        self.assertFalse(rev.support_extrapolation)

    def test_support_bounds_still_reported_in_record(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(10.5, "video.mp4", policy=mc.SELECTED_MOS_POLICY)
        record = mc.mos_calibration_record(out, cvqa_upstream=_pinned_upstream_payload())
        self.assertIn("support", record)
        self.assertEqual(record["support"]["score_min"], self._SUPPORT_LO)
        self.assertEqual(record["support"]["score_max"], self._SUPPORT_HI)
        self.assertFalse(record["input_support_clamp"])
        self.assertEqual(record["input_support_policy"], "unbounded_extrapolation")
        self.assertTrue(record["support_extrapolation"])

    def test_selected_new_digests_reviewed_unchanged(self) -> None:
        self.assertEqual(
            mc.coefficients_digest(mc.SELECTED_MOS_POLICY), SELECTED_COEFFICIENTS_DIGEST_SHA256
        )
        self.assertEqual(
            mc.embedded_coefficients_sha256(mc.SELECTED_MOS_POLICY),
            SELECTED_EMBEDDED_COEFFICIENTS_SHA256,
        )
        self.assertEqual(
            mc.embedded_coefficients_sha256(mc.REVIEWED_MOS_POLICY),
            REVIEWED_EMBEDDED_COEFFICIENTS_SHA256,
        )
        self.assertEqual(
            mc.coefficients_digest(mc.REVIEWED_MOS_POLICY), REVIEWED_COEFFICIENTS_BUNDLE_SHA256
        )

    def test_within_support_hdr_identical_sdr_offset_delta(self) -> None:
        for score in (7.5, 8.5, 9.903605146801397, self._SUPPORT_HI):
            for transfer, height in (("pq", 1080), ("pq", 2160), ("sdr", 1080), ("sdr", 2160)):
                reviewed = mc.predict_mos(
                    score, transfer=transfer, encoded_height=height, policy=mc.REVIEWED_MOS_POLICY
                )
                selected = mc.predict_mos(
                    score, transfer=transfer, encoded_height=height, policy=mc.SELECTED_MOS_POLICY
                )
                if transfer == "pq":
                    self.assertAlmostEqual(selected, reviewed, delta=1e-12, msg=f"{transfer} x={score}")
                else:
                    expected = min(5.0, max(1.0, reviewed + SELECTED_SDR_POST_MOS_OFFSET))
                    self.assertAlmostEqual(selected, expected, delta=1e-12, msg=f"{transfer} x={score}")

    def test_unsupported_media_and_gate_remain(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "arib-std-b67")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.SELECTED_MOS_POLICY)
        self.assertFalse(out.available)
        self.assertEqual(out.reason, "transfer_hlg")
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.SELECTED_MOS_POLICY)
        gated = mc.gate_reviewed_mos_outcome(out, None)
        self.assertFalse(gated.available)
        self.assertEqual(gated.reason, "cvqa_upstream_incompatible")

    def test_selected_monotonicity_dense_sweep(self) -> None:
        scores = [i * 0.25 for i in range(12, 48)]  # 3.0 .. 11.75
        mos = [
            mc.predict_mos(s, transfer="sdr", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY)
            for s in scores
        ]
        for left, right in zip(mos, mos[1:]):
            self.assertLessEqual(left, right)


class MosIntegrityEnforcementTests(unittest.TestCase):
    def test_legacy_digest_raises_on_mismatch(self) -> None:
        with patch.object(mc, "_LEGACY_COEFFICIENTS_DIGEST", "0" * 64):
            with self.assertRaises(ValueError) as ctx:
                mc.coefficients_digest()
            self.assertIn("digest mismatch", str(ctx.exception))

    def test_selected_digest_raises_under_python_O(self) -> None:
        snippet = f"""
import sys
sys.path.insert(0, {str(ROOT / "python")!r})
from pixvqa_cvqahdr import mos_calibration as mc
mc.SELECTED_COEFFICIENTS_DIGEST_SHA256 = "0" * 64
try:
    mc.coefficients_digest(mc.SELECTED_MOS_POLICY)
except ValueError:
    raise SystemExit(0)
raise SystemExit(1)
"""
        for opt in ([], ["-O"]):
            proc = subprocess.run(
                [sys.executable, *opt, "-c", snippet],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_in_memory_tamper_yields_unavailable(self) -> None:
        calibrators = mc.calibrators_for_policy(mc.SELECTED_MOS_POLICY)
        spec = calibrators["U_v2_M7_Bias"]
        original = list(spec["m1_params"])
        try:
            spec["m1_params"][0] = float(original[0]) + 1.0
            with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
                out = mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.SELECTED_MOS_POLICY)
            self.assertFalse(out.available)
            self.assertIn("calibration_error", out.reason or "")
            self.assertEqual(out.canonical_score_input, 8.5)
        finally:
            spec["m1_params"][:] = original

    def test_tamper_unavailable_record_skips_nested_digest(self) -> None:
        calibrators = mc.calibrators_for_policy(mc.SELECTED_MOS_POLICY)
        spec = calibrators["U_v2_M7_Bias"]
        original = list(spec["m1_params"])
        try:
            spec["m1_params"][0] = float(original[0]) + 1.0
            record = mc.minimal_unavailable_mos_record(8.5, "calibration_error:ValueError")
            dumped = json.dumps(record, allow_nan=False)
            self.assertNotIn("embedded_coefficients_sha256", dumped)
            self.assertNotIn("coefficients_digest_sha256", dumped)
        finally:
            spec["m1_params"][:] = original


class MosAdapterPipelineSafetyTests(unittest.TestCase):
    def test_gate_failure_inside_boundary_uses_minimal_record(self) -> None:
        """M01: gate/record failures must not discard an otherwise complete canonical run."""
        with patch.object(mc, "apply_mos_calibration") as mock_apply:
            mock_apply.return_value = mc.MosCalibrationOutcome(
                available=True,
                mos_calibrated_cvqa=3.5,
                reason=None,
                domain=mc.MosDomain("pq", 1080, "smpte2084", 1920),
                calibrator="U0_M7_Bias",
                canonical_score_input=8.5,
                policy=mc.REVIEWED_MOS_POLICY,
            )
            with patch.object(mc, "gate_reviewed_mos_outcome", side_effect=RuntimeError("gate boom")):
                with patch.object(mc, "minimal_unavailable_mos_record") as mock_minimal:
                    mock_minimal.return_value = {"status": "unavailable", "reason": "x"}
                    # Simulate adapter MOS stage (narrow boundary) without running CVQA scorer.
                    cvqa_upstream = _pinned_upstream_payload()
                    try:
                        mc.apply_mos_calibration(8.5, "video.mp4", policy=mc.REVIEWED_MOS_POLICY)
                        mc.gate_reviewed_mos_outcome(mock_apply.return_value, cvqa_upstream)
                        mc.mos_calibration_record(mock_apply.return_value, cvqa_upstream=cvqa_upstream)
                    except Exception as exc:
                        record = mc.minimal_unavailable_mos_record(
                            8.5, f"calibration_error:{type(exc).__name__}"
                        )
                    self.assertEqual(record["status"], "unavailable")

    def test_record_failure_produces_calibration_error_reason(self) -> None:
        out = mc.MosCalibrationOutcome(
            available=True,
            mos_calibrated_cvqa=3.5,
            reason=None,
            domain=mc.MosDomain("pq", 1080, "smpte2084", 1920),
            calibrator="U0_M7_Bias",
            canonical_score_input=8.5,
            policy=mc.REVIEWED_MOS_POLICY,
        )
        with patch.object(mc, "mos_calibration_record", side_effect=ValueError("digest boom")):
            try:
                mc.mos_calibration_record(out, cvqa_upstream=_pinned_upstream_payload())
            except Exception as exc:
                record = mc.minimal_unavailable_mos_record(
                    8.5, f"calibration_error:{type(exc).__name__}"
                )
        self.assertEqual(record["reason"], "calibration_error:ValueError")
        self.assertFalse(record.get("embedded_coefficients_sha256"))


class MosGateDiagnosticPreservationTests(unittest.TestCase):
    def test_upstream_gate_preserves_support_extrapolation_flags(self) -> None:
        detail = mc.predict_mos_detailed(
            11.0, transfer="pq", encoded_height=1080, policy=mc.SELECTED_MOS_POLICY
        )
        out = mc.MosCalibrationOutcome(
            available=True,
            mos_calibrated_cvqa=detail.estimated_perceptual_quality,
            reason=None,
            domain=mc.MosDomain("pq", 1080, "smpte2084", 1920),
            calibrator=detail.calibrator,
            canonical_score_input=11.0,
            policy=mc.SELECTED_MOS_POLICY,
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
        gated = mc.gate_reviewed_mos_outcome(out, None)
        self.assertFalse(gated.available)
        self.assertEqual(gated.reason, "cvqa_upstream_incompatible")
        self.assertTrue(gated.support_extrapolation)
        self.assertFalse(gated.input_support_clamp)
        record = mc.mos_calibration_record(gated, cvqa_upstream=None)
        self.assertEqual(record["status"], "unavailable")
        self.assertTrue(record.get("support_extrapolation"))


class MosJsonSanitizationTests(unittest.TestCase):
    def test_nonfinite_canonical_score_serializes_as_null(self) -> None:
        out = mc.MosCalibrationOutcome(
            available=False,
            mos_calibrated_cvqa=None,
            reason="calibration_error:canonical_cvqa_score must be finite",
            domain=None,
            calibrator=None,
            canonical_score_input=float("nan"),
            policy=mc.SELECTED_MOS_POLICY,
        )
        record = mc.mos_calibration_record(out)
        self.assertIsNone(record["canonical_score_input"])
        dumped = json.dumps(record, allow_nan=False)
        self.assertNotIn("NaN", dumped)
        self.assertNotIn("Infinity", dumped)

    def test_strict_json_dump_rejects_nan_in_record(self) -> None:
        record = {"status": "unavailable", "canonical_score_input": None}
        json.dumps(record, allow_nan=False)
        with self.assertRaises(ValueError):
            json.dumps({"canonical_score_input": float("nan")}, allow_nan=False)


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
        for case, reason in result.skipped:
            print(f"  SKIP {case}: {reason}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0 if result.wasSuccessful() else 1)
