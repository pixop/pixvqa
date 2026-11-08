#!/usr/bin/env bash
# HDR-ChipQA uses two interpreters: feature extraction (3.12+, with scikit-image) and
# SVR inference (3.10 + sklearn 1.0.2). See docs/dev/CHIPQA_REFERENCE.md.
#
# Usage:
#   scripts/setup_hdr_chipqa_envs.sh FEATURE_PYTHON INFERENCE_VENV_DIR
#
# FEATURE_PYTHON is typically the shared learned .venv interpreter. This script installs
# only missing feature packages into that environment. INFERENCE_VENV_DIR must not exist unless
# it is already a recognized pinned sklearn-1.0.2 environment.
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=learned_venv_helpers.sh
. "$_script_dir/learned_venv_helpers.sh"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 FEATURE_PYTHON INFERENCE_VENV_DIR" >&2
  exit 2
fi

feature_py="$1"
infer_dir="$2"
root="$(cd "$_script_dir/.." && pwd)"
sklearn_pin=1.0.2
stamp="$infer_dir/SKLEARN_PIN"

[[ "$infer_dir" = /* ]] || {
  echo "$0: INFERENCE_VENV_DIR must be absolute" >&2
  exit 2
}

if [[ ! -x "$feature_py" ]]; then
  echo "$0: feature interpreter missing: $feature_py" >&2
  exit 2
fi

"$feature_py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' || {
  echo "$0: HDR-ChipQA feature path must be CPython 3.12+: $feature_py" >&2
  exit 2
}

# Reused learned environments may predate HDR-ChipQA or have only a subset
# of its dependencies. Install missing modules individually without upgrading
# already working torch/numpy/scipy/OpenCV stacks.
feature_requirements=(
  'numpy:numpy>=1.24,<3'
  'scipy:scipy>=1.10,<2'
  'skimage:scikit-image>=0.21,<0.26'
  'cv2:opencv-python-headless==4.11.0.86'
  'joblib:joblib>=1.1,<2'
  'skvideo:scikit-video==1.1.11'
  'numba:numba>=0.60,<1'
  'matplotlib:matplotlib>=3.7,<4'
  'pandas:pandas>=2.0,<3'
)
pip_feature=("$feature_py" -m pip --disable-pip-version-check)
missing_feature=()
for entry in "${feature_requirements[@]}"; do
  module="${entry%%:*}"
  requirement="${entry#*:}"
  if ! "$feature_py" -c "import $module" 2>/dev/null; then
    missing_feature+=("$requirement")
  fi
done
if (( ${#missing_feature[@]} )); then
  printf '%s: installing missing HDR-ChipQA feature packages: %s\n' \
    "$0" "${missing_feature[*]}" >&2
  "${pip_feature[@]}" install --no-cache-dir "${missing_feature[@]}" >&2
fi
"$feature_py" -c 'import numpy, scipy, skimage, cv2, joblib, skvideo.io, numba, matplotlib.pyplot, pandas' || {
  echo "$0: HDR-ChipQA feature dependencies are still unavailable after installation" >&2
  exit 1
}

host_py310="$(bash "$_script_dir/ensure_chipqa_inference_python.sh")"
if [[ -x "$infer_dir/bin/python" && -f "$stamp" && "$(cat "$stamp")" == "$sklearn_pin" ]]; then
  printf 'Learned setup: reusing HDR-ChipQA inference env: %s\n' "$infer_dir" >&2
elif [[ -x "$infer_dir/bin/python" ]]; then
  # Resume a prior interrupted create (venv without SKLEARN_PIN stamp).
  printf 'Learned setup: completing HDR-ChipQA inference env at %s\n' "$infer_dir" >&2
  pip_infer=("$infer_dir/bin/python" -m pip --disable-pip-version-check)
  "${pip_infer[@]}" install --no-cache-dir \
    'numpy==1.22.4' "scikit-learn==${sklearn_pin}" 'joblib==1.1.1' >&2
  printf '%s\n' "$sklearn_pin" > "$stamp"
else
  if [[ -e "$infer_dir" ]]; then
    echo "$0: $infer_dir exists but is not a usable Python venv." >&2
    echo "Set PIXVQA_HDR_CHIPQA_INFERENCE_ENV_DIR to a new path or remove the partial directory." >&2
    exit 2
  fi
  venv_flags="$(learned_venv_create_flags "$infer_dir")"
  if [[ -n "$venv_flags" ]]; then
    echo "$0: creating inference venv with $venv_flags" >&2
  fi
  # shellcheck disable=SC2086
  "$host_py310" -m venv $venv_flags "$infer_dir"
  pip_infer=("$infer_dir/bin/python" -m pip --disable-pip-version-check)
  "${pip_infer[@]}" install --no-cache-dir \
    'numpy==1.22.4' "scikit-learn==${sklearn_pin}" 'joblib==1.1.1' >&2
  printf '%s\n' "$sklearn_pin" > "$stamp"
  printf 'Learned setup: created HDR-ChipQA inference env at %s (sklearn %s)\n' "$infer_dir" "$sklearn_pin" >&2
fi

# Never resolve the venv interpreter symlink: readlink -f can point to the
# host Python and lose the virtualenv site-packages and sklearn pin.
infer_py="$infer_dir/bin/python"
"$infer_py" -c "import sklearn; assert sklearn.__version__ == '${sklearn_pin}'" || {
  echo "$0: inference env sklearn version mismatch" >&2
  exit 1
}

# Enforce bundled model load in the inference interpreter when upstream checkout exists.
hdr_repo="${PIXVQA_HDR_CHIPQA_REPO:-$root/deps/hdr_chipqa}"
if [[ -d "$hdr_repo" && -f "$hdr_repo/hdrchipqa_livehdr_trained_svr.z" ]]; then
  PYTHONPATH="$root/python" "$infer_py" -c "
from pathlib import Path
import joblib
repo = Path('${hdr_repo}')
for name in ('hdrchipqa_livehdr_fitted_scaler.z', 'hdrchipqa_livehdr_trained_svr.z'):
    joblib.load(repo / name)
print('HDR-ChipQA SVR models loaded OK', file=__import__('sys').stderr)
" || {
    echo "$0: HDR-ChipQA model load failed in inference env" >&2
    exit 1
  }
fi

printf '%s\n' "$infer_py"
