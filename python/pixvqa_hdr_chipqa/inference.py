"""Bundled HDR-ChipQA SVR inference; features via plain numpy only."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from pixvqa_hdr_chipqa import (
    FEATURE_DIM,
    GOLDEN_FEATURE_SHA256,
    GOLDEN_SCORE,
    SCALER_NAME,
    SCALER_SHA256,
    SVR_NAME,
    SVR_SHA256,
)
from pixvqa_chipqa.inference import SetupError, InferenceError, digest, resolve_python


_BUNDLED = {SCALER_NAME: SCALER_SHA256, SVR_NAME: SVR_SHA256}


def validate_repo(repo: Path) -> dict[str, str]:
    repo = repo.expanduser().resolve()
    if not repo.is_dir():
        raise SetupError(f"HDR-ChipQA upstream checkout not found: {repo}")
    if not (repo / "hdr_chipqa.py").is_file():
        raise SetupError(f"upstream hdr_chipqa.py missing: {repo / 'hdr_chipqa.py'}")
    out: dict[str, str] = {"repo": str(repo)}
    for name, expected in _BUNDLED.items():
        path = repo / name
        if not path.is_file():
            raise SetupError(f"HDR-ChipQA bundled model missing: {path}")
        sha = digest(path)
        if sha != expected:
            raise SetupError(f"HDR-ChipQA model hash mismatch for {name}")
        out[f"{name}_sha256"] = sha
    return out


def features_to_npy(features_z: Path, features_npy: Path, feature_python: str) -> None:
    script = (
        "import joblib, numpy as np; "
        f"np.save({str(features_npy)!r}, joblib.load({str(features_z)!r})['features'])"
    )
    proc = subprocess.run([feature_python, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise InferenceError((proc.stderr or proc.stdout or "feature bridge failed")[-800:])


def predict_from_npy(features_npy: Path, repo: Path, inference_python: str | None = None) -> tuple[float, dict]:
    repo = repo.expanduser().resolve()
    models = validate_repo(repo)
    py = resolve_python(inference_python, "PIXVQA_HDR_CHIPQA_INFERENCE_PYTHON")
    script = f"""
import hashlib, json, numpy as np, joblib, os
os.chdir({str(repo)!r})
vec = np.load({str(features_npy)!r})
if vec.ndim != 1 or vec.shape[0] != {FEATURE_DIM}:
    raise SystemExit(f"feature dim {{vec.shape}} != {FEATURE_DIM}")
sha = hashlib.sha256(vec.tobytes()).hexdigest()
X = np.reshape(vec, (1, -1))
scaler = joblib.load({SCALER_NAME!r})
svr = joblib.load({SVR_NAME!r})
score = float(svr.predict(scaler.transform(X))[0])
print(json.dumps({{"quality_score": score, "feature_sha256": sha, "feature_dim": int(vec.shape[0])}}))
"""
    proc = subprocess.run([py, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise InferenceError((proc.stderr or proc.stdout or "inference failed")[-1200:])
    import json

    record = json.loads(proc.stdout.strip().splitlines()[-1])
    score = float(record["quality_score"])
    if not (score == score and abs(score) < 1e300):
        raise InferenceError("HDR-ChipQA predicted a non-finite score")
    return score, {
        **models,
        "feature_sha256": record["feature_sha256"],
        "feature_dim": record["feature_dim"],
        "inference_python": py,
        "golden_feature_sha256": GOLDEN_FEATURE_SHA256,
        "golden_score": GOLDEN_SCORE,
    }
