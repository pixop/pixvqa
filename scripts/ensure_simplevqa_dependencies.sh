#!/usr/bin/env bash
# Repair SimpleVQA dependencies in a reused learned-metric environment.
# Avoid reinstalling GPU PyTorch wheels or resetting user environments.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 LEARNED_PYTHON LOCKFILE" >&2; exit 2; }
py="$1"
lock="$2"
[[ -x "$py" && -f "$lock" ]] || { echo 'SimpleVQA setup: missing Python interpreter or dependency lock' >&2; exit 2; }
pin_from_lock() {
  local dist="$1"
  local line
  line="$(grep -E "^${dist}==" "$lock" | head -1)" || true
  [[ -n "$line" ]] || {
    echo "SimpleVQA setup: pin for ${dist} missing from ${lock}" >&2
    return 1
  }
  printf '%s\n' "$line"
}
# Standard import preflight matches the adapter's REQUIRED_IMPORTS.
missing="$("$py" - <<'PY'
import importlib.util
modules = ('torch', 'torchvision', 'pytorchvideo', 'cv2', 'PIL', 'numpy')
print(' '.join(m for m in modules if importlib.util.find_spec(m) is None))
PY
)"
if [[ -n "$missing" ]]; then
  echo "SimpleVQA setup: missing imports in $py: $missing" >&2
  # Only repair packages whose pins are included in the shared lock. This is
  # idempotent and does not reinstall CUDA torch if it is already present.
  packages=()
  for module in $missing; do
    case "$module" in
      pytorchvideo) packages+=("$(pin_from_lock pytorchvideo)") ;;
      cv2) packages+=("$(pin_from_lock opencv-python)") ;;
      PIL) packages+=("$(pin_from_lock pillow)") ;;
      numpy) packages+=("$(pin_from_lock numpy)") ;;
      torch|torchvision)
        echo "SimpleVQA setup: missing $module in established .venv; cannot safely replace CUDA torch automatically." >&2
        exit 1 ;;
      *) echo "SimpleVQA setup: unexpected missing module $module" >&2; exit 1 ;;
    esac
  done
  if [[ -n "${PIXVQA_LEARNED_INSTALL_OFFLINE:-}" ]]; then
    echo 'SimpleVQA setup: offline mode prevents dependency repair.' >&2
    exit 1
  fi
  "$py" -m pip --disable-pip-version-check install --no-cache-dir -c "$lock" "${packages[@]}" >&2
fi
"$py" - <<'PY'
import importlib.util
modules = ('torch', 'torchvision', 'pytorchvideo', 'cv2', 'PIL', 'numpy')
missing = [m for m in modules if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('SimpleVQA setup: missing required Python modules: ' + ', '.join(missing))
PY
printf 'SimpleVQA setup: required modules present in %s\n' "$py" >&2
