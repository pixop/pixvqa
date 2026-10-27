#!/usr/bin/env bash
# Install each available PixVQA build into a temporary prefix and check it
# with LD_LIBRARY_PATH unset. Does not modify the checkout build/ or $HOME.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

bash -n scripts/install.sh
bash -n scripts/verify_installed_pixvqa.sh
bash -n scripts/finish_install.sh
bash -n install_pixvqa.sh
grep -q 'verify_installed_pixvqa.sh' install_pixvqa.sh || fail 'guided installer does not verify bin/pixvqa'
grep -q 'finish_install.sh' install_pixvqa.sh || fail 'guided installer does not remove build outputs'
if grep -q 'pixvqa-with-libs' install_pixvqa.sh; then
  fail 'guided installer still creates a launcher'
fi

install_one() {
  local src="$1" kind="$2" decode="$3"
  local prefix
  prefix="$(mktemp -d)"
  mkdir -p "$prefix/bin"
  printf '#!/bin/sh\nexit 1\n' > "$prefix/bin/pixvqa-with-libs"
  chmod +x "$prefix/bin/pixvqa-with-libs"
  PREFIX="$prefix" PIXVQA_BIN="$src" bash scripts/install.sh >/dev/null
  [[ ! -e "$prefix/bin/pixvqa-with-libs" ]] || fail "launcher remained for $src"
  bash scripts/verify_installed_pixvqa.sh "$prefix" "$kind" "$decode"
  rm -rf "$prefix"
  echo "OK   installed $src as bin/pixvqa ($kind / $decode)"
}

if [[ -x build/cpu/pixvqa ]]; then
  install_one build/cpu/pixvqa cpu none
else
  echo "SKIP cpu: build/cpu/pixvqa is absent"
fi
if [[ -x build/swdec/pixvqa ]]; then
  install_one build/swdec/pixvqa cpu software
else
  echo "SKIP software-decoding: build/swdec/pixvqa is absent"
fi
if [[ -x build/full/pixvqa ]]; then
  install_one build/full/pixvqa cuda "software nvdec"
else
  echo "SKIP nvidia: build/full/pixvqa is absent"
fi

scratch="$(mktemp -d)"
mkdir -p "$scratch/build"
printf 'keep\n' > "$scratch/build/marker"
PIXVQA_SRC="$scratch" bash scripts/finish_install.sh >/dev/null
[[ ! -e "$scratch/build" ]] || fail 'finish_install left build/ in place'
mkdir -p "$scratch/build"
printf 'keep\n' > "$scratch/build/marker"
PIXVQA_KEEP_BUILD=1 PIXVQA_SRC="$scratch" bash scripts/finish_install.sh >/dev/null
[[ -f "$scratch/build/marker" ]] || fail 'PIXVQA_KEEP_BUILD=1 removed build/'
rm -rf "$scratch"
echo "OK   build cleanup honors PIXVQA_KEEP_BUILD"

real_home="${HOME}"
before="$( [[ -e "$real_home/dav1d-1.5.1" ]] && stat -c '%d:%i:%Y' "$real_home/dav1d-1.5.1" || echo absent )"
marker="$ROOT/build/cpu/pixvqa"
before_build="$( [[ -e "$marker" ]] && stat -c '%d:%i:%Y' "$marker" || echo absent )"
after="$( [[ -e "$real_home/dav1d-1.5.1" ]] && stat -c '%d:%i:%Y' "$real_home/dav1d-1.5.1" || echo absent )"
after_build="$( [[ -e "$marker" ]] && stat -c '%d:%i:%Y' "$marker" || echo absent )"
[[ "$before" == "$after" ]] || fail 'home dav1d prefix changed'
[[ "$before_build" == "$after_build" ]] || fail 'checkout build/cpu/pixvqa changed'
echo "OK   checkout build and home dav1d were not modified"
echo "All install-layout checks passed"
