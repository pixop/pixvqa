#!/usr/bin/env bash
# Create the shared learned-metrics environment (cvvdp, cvqa-hdr-nr, SimpleVQA deps)
# from python/requirements/learned-env.txt.
# See docs/LEARNED_METRICS.md for defaults, WSL2/NTFS notes, and checkpoint handling.
#
#   scripts/setup_learned_env.sh ENV_DIR COLORVIDEOVDP_CHECKOUT
#
# ENV_DIR must not exist. The ColorVideoVDP checkout is read, not modified: pip
# clones the pinned commit from it. Model checkpoints are never downloaded.
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=learned_venv_helpers.sh
. "$_script_dir/learned_venv_helpers.sh"

if [ $# -ne 2 ]; then
  echo "usage: $0 ENV_DIR COLORVIDEOVDP_CHECKOUT" >&2
  exit 2
fi
env_dir=$1
cvvdp_src=$(cd "$2" && pwd)
cvvdp_commit=166f5ae95ecbd841ae727984cd237de213875399
root=$(cd "$(dirname "$0")/.." && pwd)
lock=$root/python/requirements/learned-env.txt
python=${PYTHON:-python3.12}

if [ -e "$env_dir" ]; then
  echo "$0: $env_dir already exists" >&2
  exit 2
fi
if ! git -C "$cvvdp_src" cat-file -e "$cvvdp_commit^{commit}" 2>/dev/null; then
  echo "$0: $cvvdp_src does not contain ColorVideoVDP commit $cvvdp_commit" >&2
  exit 2
fi

venv_flags="$(learned_venv_create_flags "$env_dir")"
if [ -n "$venv_flags" ]; then
  echo "$0: creating venv with $venv_flags (host↔Linux mount such as 9p/drvfs; override with PIXVQA_LEARNED_VENV_USE_SYMLINKS=1); see docs/LEARNED_METRICS.md" >&2
fi
# shellcheck disable=SC2086
"$python" -m venv $venv_flags "$env_dir"
pip=("$env_dir/bin/python" -m pip --disable-pip-version-check)
# Only the PyTorch wheels come from the cu128 index; everything else from PyPI.
"${pip[@]}" install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
  -c "$lock" torch torchvision
"${pip[@]}" install --no-cache-dir -r "$lock"
# pip's git+file:// install fetches the commit by hash, which a shallow checkout
# refuses. Export exactly that commit's tree instead; the checkout is only read.
export_dir=$(mktemp -d)
trap 'rm -rf "$export_dir"' EXIT
git -C "$cvvdp_src" archive --format=tar "$cvvdp_commit" | tar -x -C "$export_dir"
"${pip[@]}" install --no-cache-dir --no-deps "$export_dir"
printf '%s\n' "$cvvdp_commit" > "$env_dir/COLORVIDEOVDP_COMMIT"
# cvvdp declares ffmpeg (a PyPI placeholder), matplotlib (plots), and pyexr (EXR
# images); none is used to score video. Any other unmet requirement fails.
check=$("${pip[@]}" check 2>&1 || true)
unexpected=$(printf '%s\n' "$check" | grep -vE '^(No broken requirements found\.|cvvdp [0-9.]+ requires (ffmpeg|matplotlib|pyexr), which is not installed\.)$' || true)
if [ -n "$unexpected" ]; then
  printf '%s\n' "$unexpected" >&2
  exit 1
fi
"$env_dir/bin/python" - <<'EOF'
import torch, pycvvdp, open_clip, timm, cv2
import pytorchvideo  # SimpleVQA SlowFast hub
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("pycvvdp", pycvvdp.__file__)
print("pytorchvideo", pytorchvideo.__file__)
EOF
