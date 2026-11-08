#!/usr/bin/env bash
# Offline bootstrap tests. Mock the package manager; no package download.
set -euo pipefail
root="$(cd -- "$(dirname "$0")/.." && pwd -P)"
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
mkdir -p "$work/bin" "$work/home"
cat > "$work/bin/micromamba" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == create ]] || exit 2
shift
prefix=""
while (( $# )); do
  if [[ "$1" == -p ]]; then prefix="$2"; shift 2; else shift; fi
done
[[ -n "$prefix" ]] || exit 2
mkdir -p "$prefix/bin"
cat > "$prefix/bin/python" <<'PY'
#!/usr/bin/env bash
if [[ "$1" == -c && "$2" == *sys.version_info* ]]; then exit 0; fi
if [[ "$1" == -c && "$2" == *sys.executable* ]]; then printf '%s\n' "$0"; exit 0; fi
exit 3
PY
chmod +x "$prefix/bin/python"
MOCK
chmod +x "$work/bin/micromamba"
# Use a mock python3.12 that fails validation to force provisioning.
cat > "$work/bin/python3.12" <<'MOCK'
#!/usr/bin/env bash
exit 1
MOCK
chmod +x "$work/bin/python3.12"
export HOME="$work/home" PIXVQA_DEPS_ROOT="$work/deps" PATH="$work/bin:$PATH"
unset PYTHON || true
first="$(bash "$root/scripts/ensure_learned_python.sh")"
[[ "$first" == "$work/deps/python312/bin/python" ]] || { echo "wrong python: $first" >&2; exit 1; }
second="$(bash "$root/scripts/ensure_learned_python.sh")"
[[ "$second" == "$first" ]] || exit 1
[[ -f "$first" ]] || exit 1
# Environment override must take priority over managed config.
PYTHON="$first" bash "$root/scripts/ensure_learned_python.sh" | grep -Fxq "$first"
# An invalid explicit interpreter must fail instead of silently provisioning.
if PYTHON=/bin/false bash "$root/scripts/ensure_learned_python.sh" >/dev/null 2>&1; then
  echo 'invalid interpreter accepted' >&2; exit 1
fi
printf 'PASS learned Python 3.12 bootstrap (provision/reuse/override/invalid)\n'
