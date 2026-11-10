#!/usr/bin/env bash
# Print a reproducible PixVQA build version. No writes and no network access.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
base=0.3.0-dev

if [[ -n "${PIXVQA_VERSION_OVERRIDE:-}" ]]; then
  version="$PIXVQA_VERSION_OVERRIDE"
  if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z][0-9A-Za-z.+-]*)?$ ]]; then
    echo "Invalid PIXVQA_VERSION_OVERRIDE: $version" >&2
    exit 2
  fi
  printf '%s\n' "$version"
  exit 0
fi

# Archives may be extracted inside another repository. Only inspect our own .git.
if [[ -e "$root/.git" ]] && command -v git >/dev/null 2>&1 &&
   [[ "$(git -C "$root" rev-parse --show-toplevel 2>/dev/null || true)" == "$root" ]]; then
  tag="$(git -C "$root" describe --exact-match --tags --match 'v[0-9]*' HEAD 2>/dev/null || true)"
  if [[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z][0-9A-Za-z.-]*)?$ ]]; then
    printf '%s\n' "${tag#v}"
  else
    sha="$(git -C "$root" rev-parse --short=10 HEAD)"
    printf '%s+%s\n' "$base" "$sha"
  fi
else
  # Source archives without Git metadata remain buildable and identifiable.
  printf '%s\n' "$base+unknown"
fi
