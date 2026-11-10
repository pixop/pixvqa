#!/usr/bin/env bash
# Update the generated header only when the build-version value changes.
set -euo pipefail
if (( $# != 1 )); then
  echo 'usage: write_version_header.sh OUTPUT_HEADER' >&2
  exit 2
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
version="$(bash "$root/scripts/version.sh")"
out="$1"
mkdir -p -- "$(dirname -- "$out")"
tmp="$(mktemp "${out}.XXXXXX")"
trap 'rm -f -- "$tmp"' EXIT
printf '#pragma once\n#define PIXVQA_VERSION "%s"\n' "$version" > "$tmp"
if [[ ! -f "$out" ]] || ! cmp -s "$tmp" "$out"; then
  mv -f -- "$tmp" "$out"
fi
