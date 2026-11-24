"""Regression tests for the opt-in H1-7 human ACR research overlay."""
from __future__ import annotations

import math
import unittest

from pixvqa_cvqahdr import mos_calibration_a2
from pixvqa_cvqahdr import mos_calibration_h1_research as h1
from pixvqa_cvqahdr import mos_calibration as master


class H1ResearchTests(unittest.TestCase):
    def test_source_provenance(self):
        m=h1.h1_research_metadata()
        self.assertEqual(m["observers_collected"], 7)
        self.assertEqual(m["human_ACR_ratings_screened"], 168)
        self.assertEqual(m["source_clips"], 28)
        self.assertEqual(m["observers_screened"],["B01","B03","B04","B05","B06","B07"])
        self.assertEqual(m["status"], "experimental_unvalidated")

    def test_numeric_curve_samples(self):
        expected={
            8.0:3.406362342533760,
            8.5:3.669229499892763,
            9.0:3.942895467151189,
            9.5:4.164849704157003,
            10.0:4.311259501370519,
        }
        for score,target in expected.items():
            with self.subTest(cvqa=score):
                out=h1.predict_mos_h1_research(score,transfer="pq",encoded_height=2160)
                self.assertAlmostEqual(out,target,places=11)

    def test_experimental_is_monotonic_over_evidence_support(self):
        previous=None
        for i in range(1501):
            score=h1.CVQA_H1_MIN+(h1.CVQA_H1_MAX-h1.CVQA_H1_MIN)*i/1500
            out=h1.predict_mos_h1_research(score,transfer="pq",encoded_height=2160)
            self.assertTrue(1.0<=out<=5.0)
            if previous is not None:self.assertGreater(out,previous)
            previous=out

    def test_score_above_10_is_valid_and_supported_to_empirical_end(self):
        out=h1.predict_mos_h1_research(10.05,transfer="pq",encoded_height=2160)
        self.assertTrue(4.0<out<5.0)
        for x in (5.0,6.0,7.0,7.5,11.0,10.2):
            with self.subTest(unsupported_cvqa=x):
                with self.assertRaisesRegex(ValueError,"unsupported"):
                    h1.predict_mos_h1_research(x,transfer="pq",encoded_height=2160)

    def test_other_a2_routes_are_bitwise_unchanged(self):
        for transfer,height in (("sdr",1080),("pq",1080),("sdr",2160)):
            for x in (0.,5.,6.,7.,7.6,8.,9.,10.,10.2,11.):
                with self.subTest(transfer=transfer,height=height,score=x):
                    actual=h1.predict_mos_h1_research(x,transfer=transfer,encoded_height=height)
                    prior=mos_calibration_a2.predict_mos(x,transfer=transfer,encoded_height=height)
                    self.assertEqual(actual,prior)

    def test_non_finite_rejected(self):
        for x in (float("nan"),float("inf"),-float("inf"),"not_a_score"):
            with self.assertRaises(ValueError):
                h1.predict_mos_h1_research(x,transfer="pq",encoded_height=2160)

    def test_production_algorithm_selector_not_modified(self):
        self.assertEqual(master.DEFAULT_MOS_CALIBRATION_ALGORITHM,"v7_convex_c2")
        self.assertEqual(master.resolve_mos_calibration_algorithm("a2")[0],
                         mos_calibration_a2.MOS_CALIBRATION_ALGORITHM)
        self.assertIsNotNone(master.resolve_mos_calibration_algorithm(
            h1.RESEARCH_VERSION)[1])


if __name__=="__main__":
    unittest.main()
