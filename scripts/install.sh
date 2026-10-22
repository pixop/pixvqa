#!/usr/bin/env bash
# Install a built pixvqa into PREFIX. Does not copy CUDA or FFmpeg libraries.
# Shared libraries are loaded from the build-time rpath or from LD_LIBRARY_PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${PREFIX:?Set PREFIX to the install directory}"
BIN="${PIXVQA_BIN:-$ROOT/build/cpu/pixvqa}"
if [[ ! -x "$BIN" ]]; then
  echo "PIXVQA_BIN=$BIN is not an executable." >&2
  exit 1
fi
install -d "$PREFIX/bin" "$PREFIX/share/pixvqa/model" "$PREFIX/share/doc/pixvqa/licenses"
install -m 755 "$BIN" "$PREFIX/bin/pixvqa"
cp -a "$ROOT/model/"*.json "$PREFIX/share/pixvqa/model/"
cp -a "$ROOT/README.md" "$ROOT/THIRD_PARTY_NOTICES.md" "$PREFIX/share/doc/pixvqa/"
cp -a "$ROOT/licenses/PU21_BSD.txt" "$PREFIX/share/doc/pixvqa/licenses/"
cp -a "$ROOT/third_party/netflix-vmaf/LICENSE" "$PREFIX/share/doc/pixvqa/licenses/VMAF_LICENSE"
cp -a "$ROOT/third_party/netflix-vmaf/NOTICE" "$PREFIX/share/doc/pixvqa/licenses/VMAF_NOTICE"
echo "Installed $BIN to $PREFIX/bin/pixvqa"
echo "Models are in $PREFIX/share/pixvqa/model. Pass --model with that path. There is no default model."
