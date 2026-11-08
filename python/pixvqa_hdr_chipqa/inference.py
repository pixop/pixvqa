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
    SDR_GOLDEN_FEATURE_SHA256,
    SDR_GOLDEN_SCORE,
    SVR_NAME,
    SVR_SHA256,
)

_HDR_CHIPQA_SETUP_DOC = "docs/dev/CHIPQA_REFERENCE.md"

_BUNDLED = {SCALER_NAME: SCALER_SHA256, SVR_NAME: SVR_SHA256}


class SetupError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_python_path(raw: str) -> str:
    path = Path(raw).expanduser()
    if path.is_dir():
        candidate = path / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
        raise SetupError(f"venv directory has no bin/python: {path}")
    if path.is_file():
        return str(path)
    raise SetupError(f"Python interpreter not found: {raw!r}")


def resolve_python(explicit: str | None, env_var: str, *, required: bool = True) -> str:
    if explicit:
        return _normalize_python_path(explicit)
    from_env = os.environ.get(env_var, "").strip()
    if from_env:
        return _normalize_python_path(from_env)
    if not required:
        return ""
    raise SetupError(
        f"HDR-ChipQA Python not configured ({env_var}). "
        f"Set {env_var} or pass an explicit interpreter path (see {_HDR_CHIPQA_SETUP_DOC})."
    )


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
        "sdr_golden_feature_sha256": SDR_GOLDEN_FEATURE_SHA256,
        "sdr_golden_score": SDR_GOLDEN_SCORE,
    }


def git_revision(repo: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None
