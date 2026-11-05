#!/usr/bin/env bash
set -euo pipefail

# Run from PixVQA repository root.
# Usage: VIDEO=/absolute/path/video.mp4 REPEATS=3 bash run_cvqa_operating_sweep.sh
# Optional: SAMPLE_FPS_LIST='1 2 4 15 30 60'
#           CHUNK_SEC_LIST='0.5 1 2 4 8' MAX_BATCH=32
#           OUT=cvqa_sweep_custom SOURCE_FPS=60

: "${VIDEO:?Set VIDEO=/absolute/path/to/video.mp4}"
BINARY="${BINARY:-./build/full/pixvqa}"
REPEATS="${REPEATS:-3}"
SAMPLE_FPS_LIST="${SAMPLE_FPS_LIST:-1 2 4 8 15 30 60}"
CHUNK_SEC_LIST="${CHUNK_SEC_LIST:-0.5 1 2 4 8}"
MAX_BATCH="${MAX_BATCH:-32}"
SOURCE_FPS="${SOURCE_FPS:-60}"
OUT="${OUT:-cvqa_sweep_$(date +%Y%m%d_%H%M%S)}"

[[ -f "$VIDEO" ]] || { echo "Missing video: $VIDEO" >&2; exit 2; }
[[ -x "$BINARY" ]] || { echo "Missing executable: $BINARY" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'REPEATS must be a positive integer' >&2; exit 2; }
for cmd in python3 nvidia-smi /usr/bin/time; do
  command -v "$cmd" >/dev/null || { echo "Missing $cmd" >&2; exit 2; }
done

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export VIDEO BINARY REPEATS SAMPLE_FPS_LIST CHUNK_SEC_LIST MAX_BATCH SOURCE_FPS OUT

# Build reproducible run manifest, rotating configuration order per repetition.
python3 - <<'PY'
import csv, os, pathlib
from decimal import Decimal, ROUND_HALF_UP
out = pathlib.Path(os.environ['OUT'])
rates = [Decimal(v) for v in os.environ['SAMPLE_FPS_LIST'].split()]
chunks = [Decimal(v) for v in os.environ['CHUNK_SEC_LIST'].split()]
cap = int(os.environ['MAX_BATCH'])
if cap < 1 or any(v <= 0 for v in rates + chunks):
    raise SystemExit('All rates/chunk durations and MAX_BATCH must be positive')
configs = []
for fps in rates:
    for chunk in chunks:
        batch = max(1, int((fps*chunk).quantize(Decimal('1'), rounding=ROUND_HALF_UP)))
        if batch > cap:
            continue
        configs.append((fps, chunk, batch))
if not configs:
    raise SystemExit('No configurations below MAX_BATCH; change matrix or cap')
config_keys = set()
for fps, chunk, batch in configs:
    key = (str(fps), str(chunk))
    if key in config_keys:
        raise SystemExit(f'Duplicate configuration: {key}')
    config_keys.add(key)
rows = []
for rep in range(1, int(os.environ['REPEATS'])+1):
    ordered = configs[(rep-1)%len(configs):] + configs[:(rep-1)%len(configs)]
    if rep % 2 == 0:
        ordered = list(reversed(ordered))
    for fps, chunk, batch in ordered:
        ftag = format(fps, 'f').replace('.', 'p')
        ctag = format(chunk, 'f').replace('.', 'p')
        prefix = f'fps_{ftag}_chunk_{ctag}_rep_{rep}'
        rows.append({'fps': str(fps), 'interval_sec': format(Decimal(1)/fps, '.12f'),
                     'chunk_sec': str(chunk), 'planned_batch': batch, 'repeat': rep, 'prefix': prefix})
with (out/'manifest.csv').open('w', newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
with (out/'config.txt').open('w') as f:
    for k in ('VIDEO','BINARY','REPEATS','SAMPLE_FPS_LIST','CHUNK_SEC_LIST','MAX_BATCH','SOURCE_FPS'):
        f.write(f'{k}={os.environ[k]}\n')
print(f'Planned {len(configs)} operating points x {os.environ["REPEATS"]} repeats = {len(rows)} runs')
print(f'Results directory: {out}')
PY

nvidia-smi --query-gpu=name,index,memory.total --format=csv > "$OUT/gpu_info.csv"
MON_PID=''
cleanup() {
  if [[ -n "$MON_PID" ]]; then
    kill "$MON_PID" 2>/dev/null || true
    wait "$MON_PID" 2>/dev/null || true
    MON_PID=''
  fi
}
trap cleanup EXIT INT TERM

# Read the manifest safely (no commas in its fields).
while IFS=, read -r fps interval chunk batch rep prefix; do
  [[ "$fps" == 'fps' ]] && continue
  base="$OUT/$prefix"
  if [[ -s "${base}_detail.json" && -s "${base}_time.txt" && -s "${base}_gpu.csv" && "$(cat "${base}_exit_code.txt" 2>/dev/null || echo 1)" == '0' ]]; then
    echo "SKIP completed: $prefix"
    continue
  fi
  echo "Running $prefix (sampling ${fps} fps, interval ${interval}s, chunk ${chunk}s, ~${batch} frames/batch)"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 \
    > "${base}_gpu.csv" 2> "${base}_gpu_stderr.log" &
  MON_PID=$!
  date +%s.%N > "${base}_start_unix.txt"
  set +e
  /usr/bin/time -f 'wall_sec=%e\ncpu_user_sec=%U\ncpu_sys_sec=%S\npeak_host_rss_kb=%M' \
    -o "${base}_time.txt" \
    env PIXVQA_CVQA_NR_PROFILE=1 "$BINARY" \
    --metrics cvqa-hdr-nr \
    --hdr-transfer sdr --hdr-primaries bt709 --hdr-matrix bt709 --hdr-range limited \
    --cvqa-nr-chunk-size-sec "$chunk" \
    --cvqa-nr-sample-interval-sec "$interval" \
    --cvqahdr-output-json "${base}_detail.json" \
    -v 1 -t "$VIDEO" > "${base}_pixvqa.log" 2>&1
  status=$?
  set -e
  date +%s.%N > "${base}_end_unix.txt"
  cleanup
  printf '%s\n' "$status" > "${base}_exit_code.txt"
  if (( status != 0 )); then
    echo "PixVQA failed ($status): ${base}_pixvqa.log" >&2
    exit "$status"
  fi
  [[ -s "${base}_detail.json" ]] || { echo "No detailed JSON: $base" >&2; exit 1; }
  cat "${base}_time.txt" | head -n 1
done < "$OUT/manifest.csv"

echo "Done: $OUT"
echo "Analyze: python3 analyze_cvqa_operating_sweep.py '$OUT'"
