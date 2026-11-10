#!/usr/bin/env bash
# Offline regression verifies missing pytorchvideo repair and rerun idempotency.
set -euo pipefail
root="$(cd -- "$(dirname "$0")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT
cat > "$temp/python" <<'PY'
#!/usr/bin/env bash
if [[ "$1" == '-' ]]; then
  if [[ -e "${0%/python}.installed" ]]; then printf '\n'; else printf 'pytorchvideo\n'; fi
  exit 0
fi
if [[ "$1" == '-m' && "$2" == pip ]]; then
  printf '%s\n' "$*" >> "${0%/python}.pip-log"
  touch "${0%/python}.installed"
  exit 0
fi
exit 2
PY
chmod +x "$temp/python"
PIXVQA_LEARNED_INSTALL_OFFLINE= bash "$root/scripts/ensure_simplevqa_dependencies.sh" "$temp/python" "$root/python/requirements/learned-env.txt"
[[ -f "$temp.installed" ]] || exit 1
grep -q 'pytorchvideo==0.1.5' "$temp.pip-log"
grep -q 'pin_from_lock' "$root/scripts/ensure_simplevqa_dependencies.sh" ||
  { echo 'ensure_simplevqa_dependencies.sh must read pins from the lockfile' >&2; exit 1; }
grep -q 'pillow==13\.0\.0' "$root/python/requirements/learned-env.txt" ||
  { echo 'learned-env.txt must pin pillow==13.0.0' >&2; exit 1; }
PIXVQA_LEARNED_INSTALL_OFFLINE= bash "$root/scripts/ensure_simplevqa_dependencies.sh" "$temp/python" "$root/python/requirements/learned-env.txt"
[[ $(wc -l < "$temp.pip-log") == 1 ]] || { echo 'dependency reinstalled on rerun' >&2; exit 1; }
echo 'PASS SimpleVQA dependency repair and reuse'
