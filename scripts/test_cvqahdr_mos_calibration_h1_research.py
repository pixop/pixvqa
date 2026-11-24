"""Regression tests for 75/15/10 H1 seven-ballot ACR research module."""
from __future__ import annotations

import math
import unittest

from pixvqa_cvqahdr import mos_calibration as master
from pixvqa_cvqahdr import mos_calibration_a2
from pixvqa_cvqahdr import mos_calibration_h1_research as h1


class H1ResearchTests(unittest.TestCase):
    def test_cohort_weight_metadata(self):
        m=h1.h1_research_metadata()
        self.assertEqual(m["research_status"],"experimental_unvalidated")
        self.assertEqual(m["cohort_weights"],{
            "original_tv123_human":.75,"H1_human_ACR":.15,"AI_contact_sheets":.10})
        self.assertAlmostEqual(sum(m["cohort_weights"].values()),1.)
        self.assertEqual(m["observers_collected"],7)
        self.assertEqual(m["H1_ratings_screened"],168)
        self.assertEqual(m["source_clips"],28)
        self.assertEqual(m["observers_screened"],["B01","B03","B04","B05","B06","B07"])
        self.assertAlmostEqual(m["historical_amplitude_loss_ratio"]["old_TV123_human"],.75/.85)
        self.assertAlmostEqual(m["historical_amplitude_loss_ratio"]["AI40"],.10/.85)
        self.assertAlmostEqual(m["AI10_amplitude"],.235)
        self.assertEqual(m["calibration_method"],
            "staged two-objective sensitivity scenario, not a joint loss")

    def test_prediction_matches_research_fit(self):
        expected={
            8.0:3.5338587364845657,
            8.5:3.7570401751164386,
            9.0:3.9932196408321508,
            9.5:4.195670786311871,
            10.0:4.337083937762745,
            10.05:4.34782458578683,
        }
        for x,y in expected.items():
            with self.subTest(x=x):
                self.assertAlmostEqual(
                    h1.predict_mos_h1_research(x,transfer="pq",encoded_height=2160),
                    y,places=11)

    def test_ordinal_curve_monotonic_and_bounded_within_support(self):
        previous=None
        for i in range(2001):
            score=h1.CVQA_H1_MIN+(h1.CVQA_H1_MAX-h1.CVQA_H1_MIN)*i/2000
            value=h1.predict_mos_h1_research(score,transfer="pq",encoded_height=2160)
            self.assertTrue(1<=value<=5)
            if previous is not None:self.assertGreater(value,previous)
            previous=value

    def test_h1_unsupported_range_explicit_fail_closed(self):
        for value in (0,5,6,7,7.5,10.1,10.2,11):
            with self.subTest(x=value):
                with self.assertRaisesRegex(ValueError,"unsupported"):
                    h1.predict_mos_h1_research(value,transfer="pq",encoded_height=2160)
        self.assertTrue(4<h1.predict_mos_h1_research(10.05,transfer="pq",encoded_height=2160)<5)

    def test_other_routes_use_reweighted_ai10_base(self):
        for t,h in (("sdr",1080),("sdr",2160),("pq",1080)):
            for x in (0.,5.,6.,7.,7.75,8.5,9.,10.,11.):
                with self.subTest(transfer=t,height=h,x=x):
                    score=h1.predict_mos_h1_research(x,transfer=t,encoded_height=h)
                    old=mos_calibration_a2.predict_mos(x,transfer=t,encoded_height=h)
                    smooth=max(0,min(1,(x-7.75)/2.25))
                    expected=old-.005*smooth*smooth*(3-2*smooth)
                    self.assertAlmostEqual(score,expected,places=12)

    def test_ai10_delta_zero_below_boost_start_and_exact_full_above_10(self):
        for t,h in (("sdr",1080),("pq",1080),("sdr",2160)):
            for x in (0,5,6,7,7.75):
                self.assertEqual(h1.predict_mos_h1_research(x,transfer=t,encoded_height=h),
                                 mos_calibration_a2.predict_mos(x,transfer=t,encoded_height=h))
            self.assertAlmostEqual(
                h1.predict_mos_h1_research(10,transfer=t,encoded_height=h),
                mos_calibration_a2.predict_mos(10,transfer=t,encoded_height=h)-.005,places=12)

    def test_original_a2_unchanged(self):
        self.assertAlmostEqual(
          mos_calibration_a2.predict_mos(10,transfer="pq",encoded_height=2160),
          4.37180563823194,places=12)
        self.assertEqual(mos_calibration_a2.A2_AMPLITUDE,.240)
        self.assertEqual(mos_calibration_a2.A2_AI_LOSS_FRACTION,.25)

    def test_invalid_score_and_route_fail_closed(self):
        for score in (float("nan"),float("inf"),-float("inf"),"invalid"):
            with self.assertRaises(ValueError):
                h1.predict_mos_h1_research(score,transfer="pq",encoded_height=2160)
        for transfer,height in (("hlg",2160),("pq",720)):
            with self.assertRaises(ValueError):
                h1.predict_mos_h1_research(9,transfer=transfer,encoded_height=height)

    def test_production_selector_and_default_intact(self):
        self.assertEqual(master.DEFAULT_MOS_CALIBRATION_ALGORITHM,"v7_convex_c2")
        self.assertEqual(master.resolve_mos_calibration_algorithm("a2")[0],
                         mos_calibration_a2.MOS_CALIBRATION_ALGORITHM)
        self.assertIsNotNone(master.resolve_mos_calibration_algorithm(h1.RESEARCH_VERSION)[1])


if __name__=="__main__":
    unittest.main()
