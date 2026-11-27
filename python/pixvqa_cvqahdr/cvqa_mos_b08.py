"""Canonical B08 CVQA-to-estimated-MOS mapping, 2026-11-27.

No third-party dependencies. Use predict_mos(cvqa, transfer="pq", encoded_height=2160).
CLI: python cvqa_mos_b08.py 6 --transfer pq --height 2160
The fixed weight 0.05 is 5%, not a fraction of training observations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math

ALGORITHM = "a2_tv123_ai25_h1_b08_w005"
VERSION = "a2-tv123-ai25-h1-b08-w005-20261127-v1"
WEIGHT = 0.05
A2 = (0.09446591142061583, 0.6046767170474217,
      0.6167771933889601, 0.11819874714947097)
CENTER, SCALE, BOOST_START, BOOST_END, AMPLITUDE = 8.5, 2.0, 7.75, 10.0, 0.24
THETA = (-0.17481187557998495, 0.674162834557142, -0.40763810029946124)
BIAS = (-0.2260022704806734, -0.08047014521274108, -0.8714780208480978,
        0.7709868980706249, 0.33747765885675934, 0.2992962096844712,
        -0.2298103300703432)
SIGMA = 0.6921736536804622
CUTS = (1.5, 2.5, 3.5, 4.5)
HINGE_CENTER, HINGE_KNOT = 9.15, 9.25
L, H = 7.599227864935756, 10.098637140060532
RHO_L, RHO_SLOPE = -0.3696660530203457, 0.24353569558155375
ANCHOR = 10.0
D10 = -0.18777110419520504
TAU = 1.0
OBSERVERS = ("B01", "B03", "B04", "B05", "B06", "B07", "B08")

def coefficient_payload():
    return dict(algorithm=ALGORITHM, version=VERSION, weight=WEIGHT, a2=A2,
                center=CENTER, scale=SCALE, boost_start=BOOST_START,
                boost_end=BOOST_END, amplitude=AMPLITUDE, ai_loss_fraction=.25,
                theta=THETA, bias=BIAS, sigma=SIGMA, cuts=CUTS,
                hinge_center=HINGE_CENTER, hinge_knot=HINGE_KNOT,
                h1_support=[L,H], rho_L=RHO_L, rho_slope=RHO_SLOPE,
                anchor=ANCHOR, d10=D10, tau=TAU, observers=OBSERVERS,
                collected=8, retained_ratings=196, collected_ratings=224,
                clips=28, families=14)

def digest():
    return hashlib.sha256(json.dumps(coefficient_payload(),sort_keys=True,
                                    separators=(",",":")).encode()).hexdigest()

EXPECTED_DIGEST = "bf90bfac809f9e95ceba1cbf0ca5dc55135007d572d1fe5b888e4d5777741fc2"

def verified_digest():
    value = digest()
    if value != EXPECTED_DIGEST:
        raise ValueError("B08 coefficient digest mismatch")
    return value

def _sigmoid(z):
    if z >= 0:
        return 1.0/(1.0+math.exp(-z))
    u=math.exp(z)
    return u/(1.0+u)

def _clip(y):
    return min(5.0,max(1.0,y))

def baseline(x, transfer, encoded_height):
    if transfer not in ("pq","sdr") or encoded_height not in (1080,2160):
        raise ValueError("supported routes: pq/sdr and 1080/2160")
    x=float(x)
    if not math.isfinite(x):
        raise ValueError("CVQA must be finite")
    z=A2[0]+A2[1]*((x-CENTER)/SCALE)+A2[2]*(encoded_height==2160)+A2[3]*(transfer=="pq")
    t=min(1.,max(0.,(x-BOOST_START)/(BOOST_END-BOOST_START)))
    return _clip(1+4*_sigmoid(z)+AMPLITUDE*t*t*(3-2*t))

def expected_acr(x):
    mu=baseline(x,"pq",2160)+THETA[0]+THETA[1]*(x-HINGE_CENTER)+THETA[2]*max(0.,x-HINGE_KNOT)
    return 1+sum(_sigmoid((mu+b-k)/SIGMA) for b in BIAS for k in CUTS)/len(BIAS)

def rho(x):
    if x < L:
        return RHO_L+RHO_SLOPE*TAU*math.expm1((x-L)/TAU)
    if x == ANCHOR:
        return 0.
    a=baseline(x,"pq",2160)
    return (expected_acr(x)-a-D10)/(a-1.)

def explore(x, transfer, encoded_height, weight):
    """Research slider only; canonical callers use predict_mos."""
    x=float(x)
    a=baseline(x,transfer,encoded_height)
    if not math.isfinite(weight) or not 0 <= weight <= .3:
        raise ValueError("exploration weight must be in [0,0.3]")
    return _clip(1+(a-1)*(1+weight*rho(x)))

def predict_mos(cvqa, *, transfer, encoded_height):
    """Map canonical clip-level CVQA using the fixed B08 5% correction."""
    verified_digest()
    return explore(cvqa,transfer,encoded_height,WEIGHT)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cvqa",type=float,nargs="+")
    p.add_argument("--transfer",choices=["pq","sdr"],required=True)
    p.add_argument("--height",type=int,choices=[1080,2160],required=True)
    args=p.parse_args()
    print(json.dumps({"algorithm":ALGORITHM,"version":VERSION,"weight":WEIGHT,
        "transfer":args.transfer,"encoded_height":args.height,
        "results":[{"cvqa":x,"estimated_mos":predict_mos(x,transfer=args.transfer,encoded_height=args.height)} for x in args.cvqa]},allow_nan=False,indent=2))
if __name__=="__main__":
    main()
