#!/usr/bin/env bash
# Remove intermediate compilation outputs after a successful install.
# Set PIXVQA_KEEP_BUILD=1 to retain $PIXVQA_SRC/build. Dependencies are not removed.
set -euo pipefail

src="${PIXVQA_SRC:?Set PIXVQA_SRC to the PixVQA checkout}"
src="$(cd -- "$src" && pwd -P)"
home_root="$(cd -- "$HOME" && pwd -P)"
if [[ "$src" == "/" || "$src" == "$home_root" ]]; then
  echo "STOP: refusing to remove build outputs from $src" >&2
  exit 1
fi
target="$src/build"
if [[ "${PIXVQA_KEEP_BUILD:-}" == 1 ]]; then
  echo "Keeping intermediate outputs in $target"
  exit 0
fi
rm -rf -- "$target"
echo "Removed intermediate outputs in $target"
