#!/usr/bin/env bash
# Offline regression: reused feature interpreter missing cv2 must install it.
set -euo pipefail
repo="$(cd -- "$(dirname "$0")/.." && pwd -P)"
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
mkdir -p "$work/scripts" "$work/feature/bin" "$work/infer/bin"
cp "$repo/scripts/setup_hdr_chipqa_envs.sh" "$work/scripts/"
cp "$repo/scripts/learned_venv_helpers.sh" "$work/scripts/"
cat > "$work/scripts/ensure_chipqa_inference_python.sh" <<'SH'
#!/bin/bash
printf '%s\n' /bin/true
SH
chmod +x "$work/scripts/ensure_chipqa_inference_python.sh"
cat > "$work/feature/bin/python" <<'PY'
#!/bin/bash
if [[ "$1" == -c ]]; then
  case "$2" in
    *sys.version_info*) exit 0 ;;
    "import cv2") [[ -f "${0%/bin/python}/cv2-installed" ]] ;;
    "import skvideo") [[ -f "${0%/bin/python}/skvideo-installed" ]] ;;
    "import numba") exit 0 ;;
    "import numpy, scipy, skimage, cv2, joblib, skvideo.io, numba") [[ -f "${0%/bin/python}/cv2-installed" && -f "${0%/bin/python}/skvideo-installed" ]] ;;
    "import "*) exit 0 ;;
  esac
  exit $?
fi
if [[ "$1" == -m && "$2" == pip ]]; then
  printf '%s\n' "$*" >> "${0%/bin/python}/pip-calls"
  echo 'Collecting packages (simulated pip output)'
  case " $* " in
    *'opencv-python-headless==4.11.0.86'*'scikit-video==1.1.11'*) touch "${0%/bin/python}/cv2-installed" "${0%/bin/python}/skvideo-installed" ;;
    *) exit 3 ;;
  esac
  exit 0
fi
exit 1
PY
chmod +x "$work/feature/bin/python"
cat > "$work/infer/bin/python" <<'PY'
#!/bin/bash
exit 0
PY
chmod +x "$work/infer/bin/python"
printf '1.0.2\n' > "$work/infer/SKLEARN_PIN"
output="$("$work/scripts/setup_hdr_chipqa_envs.sh" "$work/feature/bin/python" "$work/infer")"
[[ "$output" == "$work/infer/bin/python" ]] || { echo "stdout contaminated: $output" >&2; exit 1; }
[[ -f "$work/feature/cv2-installed" && -f "$work/feature/skvideo-installed" ]] || exit 1
grep -q opencv-python-headless "$work/feature/pip-calls"
grep -q scikit-video "$work/feature/pip-calls"
output="$("$work/scripts/setup_hdr_chipqa_envs.sh" "$work/feature/bin/python" "$work/infer")"
[[ "$output" == "$work/infer/bin/python" ]] || { echo "stdout contaminated: $output" >&2; exit 1; }
[[ $(wc -l < "$work/feature/pip-calls") == 1 ]] || { echo 'reinstalled on second run' >&2; exit 1; }
echo 'PASS HDR-ChipQA reused feature env repairs cv2 and emits only inference path on stdout'
