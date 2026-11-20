#!/usr/bin/env python3
"""Regression tests for experimental A2-TV123 MOS calibration (opt-in algorithm)."""
from __future__ import annotations

import json
import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pixvqa_cvqahdr import PINNED_CHECKPOINT_SHA256, PINNED_COMMIT, PINNED_PROFILE_SHA256
from pixvqa_cvqahdr import mos_calibration as mc
from pixvqa_cvqahdr import mos_calibration_a2 as a2

ALG = a2.MOS_CALIBRATION_ALGORITHM
V7 = mc.MOS_CALIBRATION_ALGORITHM_V7

# Independent oracle from frozen decision coefficients (full precision).
A2_ORACLE: tuple[tuple[str, int, float, float], ...] = (
    ("sdr", 1080, 0.0, 1.3104004024803162),
    ("sdr", 1080, 7.75, 2.867904726155597),
    ("sdr", 1080, 8.5, 3.1566179466343987),
    ("sdr", 1080, 9.0, 3.384325127426959),
    ("sdr", 1080, 10.0, 3.7746611522181626),
    ("sdr", 2160, 0.0, 1.539449892807454),
    ("sdr", 2160, 7.75, 3.4752296168605286),
    ("sdr", 2160, 8.5, 3.744925351510499),
    ("sdr", 2160, 9.0, 3.9525753079006383),
    ("sdr", 2160, 10.0, 4.288779437564239),
    ("pq", 1080, 0.0, 1.3459771086108692),
    ("pq", 1080, 7.75, 2.985911122733601),
    ("pq", 1080, 8.5, 3.274088986964557),
    ("pq", 1080, 9.0, 3.499790416472738),
    ("pq", 1080, 10.0, 3.882582083130208),
    ("pq", 2160, 0.0, 1.5970311681290492),
    ("pq", 2160, 7.75, 3.585084224805294),
    ("pq", 2160, 8.5, 3.8471698199145643),
    ("pq", 2160, 9.0, 4.0488354613337),
    ("pq", 2160, 10.0, 4.37180563823194),
)

COMBO_CASES = (
    ("pq", 1080),
    ("pq", 2160),
    ("sdr", 1080),
    ("sdr", 2160),
)
REPRESENTATIVE_SCORES = (0.0, 7.75, 8.5, 9.0, 10.0)


def _pinned_upstream_payload() -> dict:
    return {
        "upstream": {"actual_commit": PINNED_COMMIT, "commit_verified": True},
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "profile_sha256": PINNED_PROFILE_SHA256,
    }


def setUpModule() -> None:
    os.environ.pop(mc.ENV_MOS_CALIBRATION_ALGORITHM, None)
    os.environ.pop(mc.ENV_MOS_POLICY, None)


class MosA2AlgorithmSelectorTests(unittest.TestCase):
    def test_default_algorithm_is_v7(self) -> None:
        alg, err = mc.resolve_mos_calibration_algorithm()
        self.assertIsNone(err)
        self.assertEqual(alg, V7)

    def test_explicit_a2_aliases(self) -> None:
        for alias in ("a2_tv123_ai25", "a2_tv123", "a2", "A2_TV123_AI25"):
            alg, err = mc.resolve_mos_calibration_algorithm(alias)
            self.assertIsNone(err, msg=alias)
            self.assertEqual(alg, ALG)

    def test_invalid_algorithm_fail_closed(self) -> None:
        alg, err = mc.resolve_mos_calibration_algorithm("bogus")
        self.assertIsNone(alg)
        self.assertEqual(err, "invalid_mos_calibration_algorithm:bogus")
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", algorithm="bogus")
        self.assertFalse(out.available)
        self.assertEqual(out.reason, "invalid_mos_calibration_algorithm:bogus")

    @patch.dict(os.environ, {mc.ENV_MOS_CALIBRATION_ALGORITHM: "a2_tv123_ai25"}, clear=False)
    def test_env_routes_to_a2(self) -> None:
        alg, err = mc.resolve_mos_calibration_algorithm()
        self.assertIsNone(err)
        self.assertEqual(alg, ALG)
        v7 = mc.predict_mos(10.0, transfer="sdr", encoded_height=1080, algorithm=V7)
        from_env = mc.predict_mos(10.0, transfer="sdr", encoded_height=1080)
        explicit_a2 = mc.predict_mos(10.0, transfer="sdr", encoded_height=1080, algorithm=ALG)
        self.assertNotAlmostEqual(v7, from_env, places=3)
        self.assertAlmostEqual(from_env, explicit_a2, places=12)

    def test_v7_default_unchanged_with_a2_present(self) -> None:
        before = mc.predict_mos(9.0, transfer="pq", encoded_height=2160)
        self.assertAlmostEqual(before, 3.37296696206688, delta=1e-12)


class MosA2FormulaTests(unittest.TestCase):
    def test_oracle_all_routes(self) -> None:
        for transfer, height, score, expected in A2_ORACLE:
            runtime = mc.predict_mos(
                score, transfer=transfer, encoded_height=height, algorithm=ALG
            )
            self.assertAlmostEqual(runtime, expected, delta=1e-12, msg=f"{transfer}/{height} x={score}")

    def test_monotonicity_all_routes(self) -> None:
        scores = [i * 0.25 for i in range(0, 41)]
        for transfer, height in COMBO_CASES:
            values = [
                mc.predict_mos(s, transfer=transfer, encoded_height=height, algorithm=ALG)
                for s in scores
            ]
            for left, right in zip(values, values[1:]):
                self.assertLessEqual(left, right, msg=f"{transfer}/{height}")

    def test_output_bounds(self) -> None:
        for transfer, height in COMBO_CASES:
            for score in (0.0, 4.0, 7.0, 11.0, 20.0):
                mos = mc.predict_mos(score, transfer=transfer, encoded_height=height, algorithm=ALG)
                self.assertGreaterEqual(mos, 1.0)
                self.assertLessEqual(mos, 5.0)

    def test_no_v7_sdr_offset(self) -> None:
        detail = mc.predict_mos_detailed(8.5, transfer="sdr", encoded_height=1080, algorithm=ALG)
        self.assertEqual(detail.post_mos_offset, 0.0)

    def test_a2_differs_from_v7_at_cvqa_10(self) -> None:
        for transfer, height in COMBO_CASES:
            v7 = mc.predict_mos(10.0, transfer=transfer, encoded_height=height, algorithm=V7)
            a2m = mc.predict_mos(10.0, transfer=transfer, encoded_height=height, algorithm=ALG)
            self.assertNotAlmostEqual(v7, a2m, places=2, msg=f"{transfer}/{height}")


class MosA2OodTests(unittest.TestCase):
    def test_ood_above_10(self) -> None:
        detail = mc.predict_mos_detailed(11.0, transfer="pq", encoded_height=1080, algorithm=ALG)
        self.assertTrue(detail.ood_above_support)
        self.assertTrue(detail.support_extrapolation)

    def test_below_zero_ood(self) -> None:
        detail = mc.predict_mos_detailed(-1.0, transfer="sdr", encoded_height=2160, algorithm=ALG)
        self.assertTrue(detail.ood_below_support)


class MosA2ProvenanceTests(unittest.TestCase):
    def test_digests_match(self) -> None:
        self.assertEqual(a2.coefficients_digest(), a2.SELECTED_COEFFICIENTS_DIGEST_SHA256)
        self.assertEqual(a2.embedded_coefficients_sha256(), a2.EMBEDDED_COEFFICIENTS_SHA256)

    def test_record_excludes_v7_provenance(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "smpte2084")):
            out = mc.apply_mos_calibration(8.5, "video.mp4", algorithm=ALG)
        record = mc.mos_calibration_record(out, cvqa_upstream=_pinned_upstream_payload())
        self.assertEqual(record["mos_calibration_algorithm"], ALG)
        self.assertFalse(record["v7_legacy_components"])
        self.assertNotIn("low_tail", record)
        self.assertNotIn("sdr_selection_lock_sha256", record)
        self.assertNotIn("hdr_selection_lock_sha256", record)
        self.assertIn("unified_sigmoid", record)
        self.assertIn("smoothstep_amplitude", record)
        self.assertEqual(record["post_output_offset_by_transfer"], {"pq": 0.0, "sdr": 0.0})

    def test_v7_record_includes_algorithm_id(self) -> None:
        with patch.object(mc, "probe_encoded_geometry", return_value=(1920, 1080, "bt709")):
            out = mc.apply_mos_calibration(8.5, "video.mp4")
        record = mc.mos_calibration_record(out, cvqa_upstream=_pinned_upstream_payload())
        self.assertEqual(record["mos_calibration_algorithm"], V7)
        self.assertIn("low_tail", record)

    def test_tamper_fail_closed(self) -> None:
        original = a2.A2_CVQA_COEF
        try:
            a2.A2_CVQA_COEF = original + 0.1
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                mc.predict_mos(8.5, transfer="pq", encoded_height=1080, algorithm=ALG)
        finally:
            a2.A2_CVQA_COEF = original


class MosA2AdapterRoutingTests(unittest.TestCase):
    def test_apply_sdr_2160(self) -> None:
        score = 9.0
        expected = mc.predict_mos(score, transfer="sdr", encoded_height=2160, algorithm=ALG)
        with patch.object(mc, "probe_encoded_geometry", return_value=(3840, 2160, "bt709")):
            out = mc.apply_mos_calibration(score, "video.mp4", algorithm=ALG)
        self.assertTrue(out.available)
        self.assertEqual(out.calibrator, "A2_TV123_Unified")
        self.assertAlmostEqual(out.mos_calibrated_cvqa or 0.0, expected, delta=1e-12)

    def test_protocol_metadata(self) -> None:
        from pixvqa_cvqahdr.__main__ import protocol_lines

        score = 8.5
        mos = mc.predict_mos(score, transfer="pq", encoded_height=1080, algorithm=ALG)
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
            "mos_calibration": {
                "status": "available",
                "mos_policy": ALG,
                "mos_calibration_algorithm": ALG,
            },
        }
        joined = "\n".join(protocol_lines(record, "out.json"))
        self.assertIn("pixvqa-value mos_calibration_algorithm a2_tv123_ai25", joined)

    def test_minimal_unavailable_includes_algorithm(self) -> None:
        record = mc.minimal_unavailable_mos_record(8.5, "height_unsupported", algorithm=ALG)
        self.assertEqual(record["mos_calibration_algorithm"], ALG)
        json.dumps(record, allow_nan=False)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print(f"FAIL: {len(result.skipped)} test(s) skipped", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0 if result.wasSuccessful() else 1)