#!/usr/bin/env bash
# Optional FFV1 BT.2020 NC metadata auto-resolve (requires build/swdec/pixvqa and ffmpeg).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWDEC="$ROOT/build/swdec/pixvqa"
FFMPEG="${PIXVQA_FFMPEG:-ffmpeg}"

if [[ ! -x "$SWDEC" ]]; then
  echo "SKIP compressed metadata test (no $SWDEC; run: make swdec)"
  exit 0
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/pixvqa-ref-gamut-clamp-XXXXXXXX")"
trap 'rm -rf "$TMP"' EXIT

EXTREME="$TMP/extreme.yuv"
mkv_pq="$TMP/tagged-pq.mkv"
mkv_hlg="$TMP/tagged-hlg.mkv"

python3 - <<'PY' >"$EXTREME"
import struct, sys
n = 64
plane = lambda fill: struct.pack(f"<{n}H", *([fill] * n))
# In-gamut limited BT.709 codes (bit-identical after optional reference clamp).
sys.stdout.buffer.write(plane(502) + plane(470) + plane(512))
PY

common_ffmpeg=( -f rawvideo -pix_fmt yuv444p10le -s:v 8x8 -r 1 -i "$EXTREME" -frames:v 1 )
"$FFMPEG" -hide_banner -loglevel error -y "${common_ffmpeg[@]}" -c:v ffv1 \
  -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc -color_range tv "$mkv_pq"
"$FFMPEG" -hide_banner -loglevel error -y "${common_ffmpeg[@]}" -c:v ffv1 \
  -color_primaries bt2020 -color_trc arib-std-b67 -colorspace bt2020nc -color_range tv "$mkv_hlg"

run_case() {
  local tag=$1 path=$2
  local out rc avg
  set +e
  out="$("$SWDEC" --metrics psnr -r "$path" -t "$path" --reference-rgb-gamut-clamp -n 1 -v 0 2>&1)"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "FAIL $tag: pixvqa exit $rc"
    echo "$out"
    return 1
  fi
  if ! grep -q 'matrix=bt2020nc' <<<"$out"; then
    echo "FAIL $tag: missing bt2020nc matrix in summary"
    echo "$out"
    return 1
  fi
  if ! grep -q 'range=limited' <<<"$out"; then
    echo "FAIL $tag: missing limited range"
    echo "$out"
    return 1
  fi
  avg="$(printf '%s' "$out" | python3 -c "
import re, sys
text = sys.stdin.read()
m = re.search(r\"\\[\\s*PSNR\\s+.*?\\bavg:([\\d.]+|inf|999\\.99)\", text)
if not m:
    raise SystemExit('missing avg PSNR')
v = m.group(1).lower()
print('inf' if v in ('inf', '999.99') else v)
")"
  if [[ "$avg" != "inf" ]]; then
    echo "FAIL $tag: expected infinite PSNR (identity), got avg=$avg"
    echo "$out"
    return 1
  fi
  echo "OK   $tag compressed auto metadata"
}

run_case pq "$mkv_pq"
run_case hlg "$mkv_hlg"
echo "compressed metadata checks passed"