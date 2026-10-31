#!/usr/bin/env bash
# Create the shared cvvdp + cvqa-hdr-nr environment from python/requirements/learned-env.txt.
#
#   scripts/setup_learned_env.sh ENV_DIR COLORVIDEOVDP_CHECKOUT
#
# ENV_DIR must not exist. The ColorVideoVDP checkout is read, not modified: pip
# clones the pinned commit from it. Model checkpoints are never downloaded.
set -euo pipefail

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

"$python" -m venv "$env_dir"
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
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("pycvvdp", pycvvdp.__file__)
EOF
