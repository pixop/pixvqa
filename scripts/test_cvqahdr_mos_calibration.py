#!/usr/bin/env python3
"""Independent B08 numerical and integration contracts (stdlib; no model assets)."""
from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pixvqa_cvqahdr import PINNED_COMMIT, PINNED_CHECKPOINT_SHA256, PINNED_PROFILE_SHA256
from pixvqa_cvqahdr import cvqa_mos_b08 as b08, mos_calibration as mc
from pixvqa_cvqahdr import adapter
from pixvqa_cvqahdr.__main__ import main as cli_main, protocol_lines

ALGORITHM = "a2_tv123_ai25_h1_b08_w005"
DIGEST = "bf90bfac809f9e95ceba1cbf0ca5dc55135007d572d1fe5b888e4d5777741fc2"
ROUTES = (("pq", 2160), ("pq", 1080), ("sdr", 2160), ("sdr", 1080))
GOLDEN_6 = (3.0150888286801045, 2.428343691317508, 2.9002627040289912, 2.323318629280103)
GOLDEN_10 = (4.37180563823194, 3.882582083130208, 4.288779437564239, 3.7746611522181626)
L = 7.599227864935756

# Independent scalar expression using the frozen scientific specification.
# Do not read runtime coefficients or call any runtime formula here.
def sigmoid(z):
    if z < 0:
        u = math.exp(z)
        return u / (1.0 + u)
    return 1.0 / (1.0 + math.exp(-z))


def baseline_oracle(x, t, h):
    z = (0.09446591142061583 + 0.6046767170474217 * ((x - 8.5) / 2)
         + (0.6167771933889601 if h == 2160 else 0.0)
         + (0.11819874714947097 if t == "pq" else 0.0))
    u = max(0.0, min(1.0, (x - 7.75) / 2.25))
    return max(1.0, min(5.0, 1 + 4 * sigmoid(z) + 0.240 * u**2 * (3 - 2*u)))


def oracle(x, t, h):
    if x < L:
        r = -0.3696660530203457 + 0.24353569558155375 * math.expm1(x - L)
    else:
        a = baseline_oracle(x, "pq", 2160)
        mu = (a - 0.17481187557998495 + 0.674162834557142 * (x - 9.15)
              - 0.40763810029946124 * max(0.0, x - 9.25))
        offsets = (-0.2260022704806734, -0.08047014521274108, -0.8714780208480978,
                   0.7709868980706249, 0.33747765885675934, 0.2992962096844712,
                   -0.2298103300703432)
        e = 1 + sum(sigmoid((mu + off - cut)/0.6921736536804622)
                    for off in offsets for cut in (1.5, 2.5, 3.5, 4.5)) / 7
        r = (e - a + 0.18777110419520504) / (a - 1)
    return max(1.0, min(5.0, 1 + (baseline_oracle(x, t, h) - 1) * (1 + 0.05*r)))


def pins():
    return {"upstream": {"actual_commit": PINNED_COMMIT, "commit_verified": True},
            "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
            "profile_sha256": PINNED_PROFILE_SHA256}


def route_probe(t, h):
    return (3840 if h == 2160 else 1920, h, "smpte2084" if t == "pq" else "bt709")


def run_record():
    p = pins()
    return {"quality_score": 8.5, "raw_model_score": 0.38, "metric": "cvqa-hdr-nr",
            "sampling": {"sampled_frames": 10, "opencv_frame_count": 500,
                         "opencv_fps": 50.0, "rounded_fps": 50},
            "temporal": {"sampling_density_hz": 1.0, "decoded_frame_count": None,
                         "padded_samples": None, "pooled_parity": True,
                         "qcs_eligible": True, "samples": []},
            "device": "cpu", "upstream": p["upstream"],
            "checkpoint": {"sha256": p["checkpoint_sha256"]},
            "calibration_profile": {"sha256": p["profile_sha256"]}}


class FormulaTests(unittest.TestCase):
    def test_identity_and_coefficient_integrity(self):
        self.assertEqual(mc.MOS_CALIBRATION_ALGORITHM, ALGORITHM)
        self.assertEqual(b08.WEIGHT, 0.05)
        self.assertEqual(mc.coefficients_digest(), DIGEST)
        payload = b08.coefficient_payload()
        self.assertEqual(payload["collected"], 8)
        self.assertEqual(payload["collected_ratings"], 224)
        self.assertEqual(payload["retained_ratings"], 196)
        self.assertEqual(payload["observers"], ("B01","B03","B04","B05","B06","B07","B08"))
        self.assertEqual(payload["ai_loss_fraction"], .25)

    def test_fixed_golden_scores_and_anchor(self):
        for (t,h), golden6, golden10 in zip(ROUTES, GOLDEN_6, GOLDEN_10):
            with self.subTest(transfer=t, height=h):
                self.assertAlmostEqual(b08.predict_mos(6, transfer=t, encoded_height=h), golden6, delta=1e-13)
                self.assertEqual(b08.predict_mos(10, transfer=t, encoded_height=h), golden10)
                self.assertAlmostEqual(baseline_oracle(10,t,h), golden10, delta=1e-13)
                self.assertEqual(mc.predict_mos(10, transfer=t, encoded_height=h), golden10)

    def test_dense_independent_oracle_monotonicity_and_bounds(self):
        # 80,004 comparisons over the 0..20 grid, plus exact breakpoints below.
        for t,h in ROUTES:
            previous = 1.0
            for i in range(20001):
                x = i / 1000
                actual = b08.predict_mos(x, transfer=t, encoded_height=h)
                self.assertAlmostEqual(actual, oracle(x,t,h), delta=2e-12)
                self.assertEqual(mc.predict_mos(x, transfer=t, encoded_height=h), actual)
                self.assertLessEqual(previous, actual + 1e-12)
                self.assertLessEqual(1, actual)
                self.assertLessEqual(actual, 5)
                previous = actual

    def test_knots_continuity_and_extension(self):
        for t,h in ROUTES:
            for x in (L-1e-8,L,L+1e-8,7.75,9.15,9.25,10,10.098637140060532,12,20,-10):
                a = mc.predict_mos(x, transfer=t, encoded_height=h)
                self.assertAlmostEqual(a, oracle(x,t,h), delta=2e-12)
            for x in (L,7.75,9.25,10):
                lo = mc.predict_mos(x-1e-7, transfer=t, encoded_height=h)
                hi = mc.predict_mos(x+1e-7, transfer=t, encoded_height=h)
                self.assertLess(abs(hi-lo), 1e-6)

    def test_route_ordering(self):
        for i in range(1501):
            values = [mc.predict_mos(i/100, transfer=t, encoded_height=h)
                      for t,h in (("sdr",1080),("pq",1080),("sdr",2160),("pq",2160))]
            self.assertEqual(values, sorted(values))

    def test_no_input_clamp_above_or_below(self):
        for x in (-1.0, 10.1, 15.0):
            detail = mc.predict_mos_detailed(x, transfer="pq", encoded_height=2160)
            self.assertEqual(detail.score_effective, x)
            self.assertFalse(detail.input_support_clamp)
            self.assertEqual(detail.ood_below_support, x < 0)
            self.assertFalse(detail.ood_above_support)
        self.assertGreater(mc.predict_mos(11,transfer="pq",encoded_height=2160), GOLDEN_10[0])

    def test_invalid_inputs(self):
        for x in (float("nan"), float("inf"), -float("inf"), "not-a-score", None):
            with self.subTest(x=x):
                with self.assertRaises((ValueError,TypeError)):
                    b08.predict_mos(x, transfer="pq", encoded_height=2160)
                with self.assertRaises(ValueError):
                    mc.predict_mos(x, transfer="pq", encoded_height=2160)
        for t,h in (("hlg",2160),("pq",720),("sdr",1440),("",1080)):
            with self.assertRaises(ValueError):
                mc.predict_mos(8,transfer=t,encoded_height=h)

    def test_standalone_file_only_cli(self):
        # Copy only the mapper, no package, no repository import path, no site packages.
        with tempfile.TemporaryDirectory() as d:
            module = Path(d)/"cvqa_mos_b08.py"
            module.write_bytes((ROOT/"python/pixvqa_cvqahdr/cvqa_mos_b08.py").read_bytes())
            for t,h in ROUTES:
                r = subprocess.run([sys.executable,"-I","-S",str(module),"6","10",
                                    "--transfer",t,"--height",str(h)],
                                   cwd=d,check=True,text=True,capture_output=True)
                payload = json.loads(r.stdout)
                self.assertEqual(payload["algorithm"], ALGORITHM)
                self.assertEqual(payload["weight"], .05)
                self.assertEqual([x["estimated_mos"] for x in payload["results"]],
                                 [mc.predict_mos(x,transfer=t,encoded_height=h) for x in (6,10)])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        resolver = patch.object(mc, "resolve_ffprobe_legacy", return_value="ffprobe-test")
        resolver.start()
        self.addCleanup(resolver.stop)

    def test_all_four_probe_routes(self):
        for t,h in ROUTES:
            with patch.object(mc,"probe_encoded_geometry",return_value=route_probe(t,h)):
                out = mc.apply_mos_calibration(8.5,"test.mp4")
                self.assertTrue(out.available)
                self.assertEqual(out.domain.transfer,t)
                self.assertEqual(out.domain.encoded_height,h)
                self.assertAlmostEqual(out.mos_calibrated_cvqa,oracle(8.5,t,h),delta=1e-12)
                self.assertEqual(out.canonical_score_input,8.5)
                self.assertEqual(out.calibration_algorithm,ALGORITHM)

    def test_transfer_aliases(self):
        for tag in ("pq","st2084","smpte2084"," SMPTE2084 "):
            self.assertEqual(mc.classify_transfer_tag(tag),("pq",None))
        for tag in ("bt709","smpte170m","bt470bg","bt470m","iec61966-2-1",
                    "gamma22","gamma28","bt2020-10","bt2020-12"):
            self.assertEqual(mc.classify_transfer_tag(tag),("sdr",None))

    def test_unavailable_domains(self):
        cases = ((1920,720,"bt709","height_unsupported"),
                 (3840,2160,"arib-std-b67","transfer_hlg"),
                 (3840,2160,"hlg","transfer_hlg"),
                 (1920,1080,"bt2020","transfer_unsupported"),
                 (1920,1080,"unknown","transfer_missing"),
                 (1920,1080,"log100","transfer_unsupported"),
                 (1920,1080,None,"transfer_missing"))
        for w,h,tag,reason in cases:
            with self.subTest(tag=tag,height=h), patch.object(mc,"probe_encoded_geometry",return_value=(w,h,tag)):
                out=mc.apply_mos_calibration(8.5,"test.mp4")
                self.assertFalse(out.available)
                self.assertEqual(out.reason,reason)
                self.assertEqual(out.canonical_score_input,8.5)
                record=mc.mos_calibration_record(out)
                self.assertEqual(record["mos_calibration_algorithm"],ALGORITHM)
                self.assertNotIn("mos_calibrated_cvqa",record)

    def test_probe_parser_success(self):
        for t,h in ROUTES:
            w,_,tag=route_probe(t,h)
            payload=json.dumps({"streams":[{"width":w,"height":h,"color_transfer":tag}]})
            with patch.object(mc.subprocess,"run",return_value=subprocess.CompletedProcess([],0,payload,"")):
                self.assertEqual(mc.probe_encoded_geometry("test.mp4"),(w,h,tag))

    def test_probe_parser_errors_preserve_canonical(self):
        for data, reason in (("not-json","invalid_ffprobe_json"),("[]","invalid_ffprobe_payload"),
                            ('{"streams":[]}',"no_video_stream"),('{"streams":null}',"invalid_ffprobe_payload"),
                            ('{"streams":[null]}',"invalid_ffprobe_payload"),
                            ('{"streams":[{"width":"x","height":1080}]}',"invalid_geometry"),
                            ('{"streams":[{"width":0,"height":1080}]}',"non_positive_geometry")):
            with self.subTest(reason=reason),patch.object(mc.subprocess,"run",return_value=subprocess.CompletedProcess([],0,data,"")):
                out=mc.apply_mos_calibration(8.5,"test.mp4")
                self.assertFalse(out.available)
                self.assertEqual(out.reason,reason)
                self.assertEqual(out.canonical_score_input,8.5)

    def test_probe_exceptions(self):
        for exc, reason in ((subprocess.TimeoutExpired("ffprobe",30),"ffprobe_timeout"),
                            (OSError("missing"),"ffprobe_failed"),
                            (subprocess.CalledProcessError(1,"ffprobe"),"ffprobe_failed")):
            with patch.object(mc.subprocess,"run",side_effect=exc):
                out=mc.apply_mos_calibration(8.5,"test.mp4")
                self.assertFalse(out.available)
                self.assertEqual(out.reason,reason)

    def test_upstream_gate(self):
        with patch.object(mc,"probe_encoded_geometry",return_value=route_probe("pq",2160)):
            out=mc.apply_mos_calibration(8.5,"test.mp4")
        self.assertTrue(mc.cvqa_upstream_compatibility(pins())["compatible"])
        self.assertIs(mc.gate_reviewed_mos_outcome(out,pins()),out)
        for field in ("actual_commit","checkpoint_sha256","profile_sha256"):
            for value in (None,"invalid"):
                p=pins()
                if field=="actual_commit": p["upstream"][field]=value
                else: p[field]=value
                gated=mc.gate_reviewed_mos_outcome(out,p)
                self.assertFalse(gated.available)
                self.assertEqual(gated.reason,"cvqa_upstream_incompatible")
                self.assertEqual(gated.canonical_score_input,8.5)
                record=mc.mos_calibration_record(gated,cvqa_upstream=p)
                self.assertEqual(record["cvqa_upstream_gate_status"],"blocked")
                self.assertNotIn("mos_calibrated_cvqa",record)
        self.assertFalse(mc.gate_reviewed_mos_outcome(out,None).available)

    def test_provenance_json_and_defensive_copy(self):
        with patch.object(mc,"probe_encoded_geometry",return_value=route_probe("sdr",1080)):
            out=mc.apply_mos_calibration(8.5,"test.mp4")
        record=mc.mos_calibration_record(out,cvqa_upstream=pins())
        self.assertEqual(record["status"],"available")
        self.assertEqual(record["h1_shape_weight"],.05)
        self.assertEqual(record["ai_loss_fraction"],.25)
        self.assertEqual(record["coefficients_digest_sha256"],DIGEST)
        self.assertEqual(record["embedded_coefficients_sha256"],DIGEST)
        self.assertEqual(record["mos_calibration_algorithm"],ALGORITHM)
        self.assertFalse(record["v7_legacy_components"])
        for obsolete in ("low_tail","sdr_selection_lock_sha256","hdr_selection_lock_sha256","algorithm_selector_env"):
            self.assertNotIn(obsolete,record)
        json.dumps(record,allow_nan=False)
        record["h1_correction"]["h1_support"][0]=123
        self.assertEqual(mc.coefficients_digest(),DIGEST)

    def test_nonfinite_serializes_null(self):
        out=mc.apply_mos_calibration(float("nan"),"unused.mp4")
        record=mc.mos_calibration_record(out)
        self.assertFalse(out.available)
        self.assertIsNone(record["canonical_score_input"])
        json.dumps(record,allow_nan=False)

    def test_tampering_fails_closed(self):
        for field,value in (("WEIGHT",.06),("THETA",(0,0,0)),("A2",(1,2,3,4)),("RHO_L",0)):
            with self.subTest(field=field),patch.object(b08,field,value):
                with self.assertRaisesRegex(ValueError,"digest mismatch"):
                    mc.predict_mos(8.5,transfer="pq",encoded_height=2160)
                with patch.object(mc,"probe_encoded_geometry",return_value=route_probe("pq",2160)):
                    out=mc.apply_mos_calibration(8.5,"test.mp4")
                self.assertFalse(out.available)
                self.assertIn("digest mismatch",out.reason)
                rec=mc.minimal_unavailable_mos_record(8.5,"calibration_error:ValueError")
                self.assertEqual(rec["status"],"unavailable")
                self.assertNotIn("mos_calibrated_cvqa",rec)
        self.assertEqual(mc.coefficients_digest(),DIGEST)

    def test_removed_selectors(self):
        env={"PIXVQA_MOS_CALIBRATION_ALGORITHM":"v7_convex_c2",
             "PIXVQA_CVQA_MOS_POLICY":"legacy_target_lambda1_both"}
        with patch.dict(os.environ,env):
            self.assertEqual(mc.predict_mos(6,transfer="pq",encoded_height=2160),GOLDEN_6[0])
        for fn in (mc.predict_mos,mc.predict_mos_detailed,mc.apply_mos_calibration,adapter.run):
            self.assertNotIn("algorithm",inspect.signature(fn).parameters)
            self.assertNotIn("mos_calibration_algorithm",inspect.signature(fn).parameters)
            self.assertNotIn("policy",inspect.signature(fn).parameters)
        args=["--test","x","--repo","x","--checkpoint","x","--profile","x","--output-json","x"]
        with self.assertRaises(SystemExit) as exc:
            cli_main(args+["--mos-calibration-algorithm","a2"])
        self.assertEqual(exc.exception.code,2)

    def test_protocol_and_adapter_optional_stage(self):
        for t,h in ROUTES:
            rec=run_record()
            with patch.object(mc,"probe_encoded_geometry",return_value=route_probe(t,h)):
                adapter._attach_mos_calibration(rec,"test.mp4")
            self.assertEqual(rec["quality_score"],8.5)
            self.assertAlmostEqual(rec["mos_calibrated_cvqa"],oracle(8.5,t,h),delta=1e-12)
            joined="\n".join(protocol_lines(rec,"out.json"))
            self.assertIn(f"pixvqa-value mos_calibration_algorithm {ALGORITHM}",joined)
            self.assertIn(f"pixvqa-value mos_calibrated_cvqa {rec['mos_calibrated_cvqa']}",joined)
        for target in ("apply_mos_calibration","gate_reviewed_mos_outcome","mos_calibration_record"):
            rec=run_record()
            with patch.object(mc,"probe_encoded_geometry",return_value=route_probe("pq",2160)),patch.object(adapter,target,side_effect=RuntimeError("test")):
                adapter._attach_mos_calibration(rec,"test.mp4")
            self.assertEqual(rec["quality_score"],8.5)
            self.assertNotIn("mos_calibrated_cvqa",rec)
            self.assertEqual(rec["mos_calibration"]["status"],"unavailable")
            self.assertEqual(rec["mos_calibration"]["mos_calibration_algorithm"],ALGORITHM)
            self.assertIn("pixvqa-value mos_calibrated_cvqa unavailable","\n".join(protocol_lines(rec,"out.json")))
            json.dumps(rec,allow_nan=False)


if __name__ == "__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
