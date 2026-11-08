#!/usr/bin/env bash
# Bring the installed cvvdp metadata and packages in line with its video runtime.
# Pinned ColorVideoVDP 166f5ae incorrectly declares both ffmpeg-python and the
# conflicting PyPI `ffmpeg` package. Never install the latter: it shadows the
# `ffmpeg` module that supplies .input(), .probe() and .run_async().
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'usage: ensure_cvvdp_dependencies.sh PYTHON CVVDP_GIT_CHECKOUT PINNED_COMMIT' >&2; exit 2; }
py="$1"; checkout="$2"; commit="$3"
[[ -x "$py" && -d "$checkout/.git" ]] || { echo 'CVVDP setup: missing interpreter or source checkout' >&2; exit 2; }
"$py" - <<'PY'
import importlib.metadata as md
import ffmpeg
assert all(callable(getattr(ffmpeg, key, None)) for key in ('input', 'probe', 'run_async')), 'ffmpeg-python API unavailable or shadowed by PyPI ffmpeg'
assert md.version('cvvdp') == '0.5.7', 'unexpected cvvdp version'
PY
# Correct the known bad upstream packaging declaration, without mutating
# checkouts, swapping ffmpeg modules, or editing installed dist-info in place.
if "$py" - <<'PY'
from importlib.metadata import distribution
import sys
lines = distribution('cvvdp').requires or []
sys.exit(0 if any(s.lower().startswith('ffmpeg>=1.4') for s in lines) else 1)
PY
then
  export_dir="$(mktemp -d)"
  trap 'rm -rf -- "$export_dir"' EXIT
  git -C "$checkout" archive --format=tar "$commit" | tar -x -C "$export_dir"
  "$py" - "$export_dir/pyproject.toml" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]); s = p.read_text(encoding='utf-8')
needle = '   "ffmpeg>=1.4",\n'
assert s.count(needle) == 1, 'unexpected pinned CVVDP dependency metadata'
p.write_text(s.replace(needle, ''), encoding='utf-8')
PY
  echo 'CVVDP setup: repairing conflicting upstream ffmpeg package metadata (keeping ffmpeg-python)' >&2
  "$py" -m pip --disable-pip-version-check install --no-deps --no-build-isolation --force-reinstall "$export_dir" >&2
fi
# Fix cvvdp's stale metadata BEFORE pip installs any missing dependencies.
# Otherwise pip reports a transient conflict against the old requirements.
# Install only genuinely missing declared dependencies; the FFmpeg executable
# is external and `ffmpeg-python` is already installed in the common venv.
missing="$("$py" - <<'PY'
import importlib.metadata as md
for name, spec in [('matplotlib', 'matplotlib>=3.8,<4'), ('PyEXR', 'PyEXR==0.5.0')]:
    try: md.version(name)
    except md.PackageNotFoundError: print(spec)
PY
)"
if [[ -n "$missing" ]]; then
  [[ -z "${PIXVQA_LEARNED_INSTALL_OFFLINE:-}" ]] || { echo 'CVVDP setup: missing declared packages in offline mode' >&2; exit 1; }
  mapfile -t packages <<< "$missing"
  "$py" -m pip --disable-pip-version-check install --no-cache-dir "${packages[@]}" >&2
fi
"$py" - <<'PY'
import importlib.metadata as md
import ffmpeg
assert all(callable(getattr(ffmpeg, key, None)) for key in ('input', 'probe', 'run_async'))
assert not any(x.lower().startswith('ffmpeg>=1.4') for x in md.distribution('cvvdp').requires or [])
import matplotlib, pyexr
PY
"$py" -m pip check
printf 'CVVDP setup: installed dependency metadata and ffmpeg-python API verified\n' >&2
