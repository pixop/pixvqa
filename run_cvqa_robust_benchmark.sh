#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   VIDEO='/absolute/path/video.mp4' REPEATS=3 bash run_cvqa_robust_benchmark.sh
# Run from the PixVQA repository root. No code changes or rebuild needed.

: "${VIDEO:?Set VIDEO to the absolute path of the compressed source file}"
REPEATS="${REPEATS:-3}"
OUT="${OUT:-cvqa_robust_$(date +%Y%m%d_%H%M%S)}"
BINARY="${BINARY:-./build/full/pixvqa}"

if [[ ! -f "$VIDEO" ]]; then echo "Missing video: $VIDEO" >&2; exit 2; fi
if [[ ! -x "$BINARY" ]]; then echo "Missing executable: $BINARY" >&2; exit 2; fi
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then echo 'REPEATS must be a positive integer' >&2; exit 2; fi
for program in nvidia-smi python3 /usr/bin/time; do
  command -v "$program" >/dev/null || { echo "Missing $program" >&2; exit 2; }
done
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
printf 'video=%s\nrepeats=%s\nbinary=%s\n' "$VIDEO" "$REPEATS" "$BINARY" > "$OUT/config.txt"
nvidia-smi --query-gpu=name,index,memory.total --format=csv > "$OUT/gpu_info.csv"

MON_PID=''
cleanup() {
  if [[ -n "$MON_PID" ]]; then
    kill "$MON_PID" 2>/dev/null || true
    wait "$MON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Rotate execution order between repetitions to reduce systematic cache bias.
for ((rep=1; rep<=REPEATS; rep++)); do
  case "$((rep % 3))" in
    1) intervals=(1 0.5 0.25) ;;
    2) intervals=(0.5 0.25 1) ;;
    0) intervals=(0.25 1 0.5) ;;
  esac
  for interval in "${intervals[@]}"; do
    prefix="$OUT/interval_${interval}_rep_${rep}"
    echo "Running repeat $rep/$REPEATS at interval ${interval}s"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 > "${prefix}_gpu.csv" 2> "${prefix}_gpu_stderr.log" &
    MON_PID=$!
    python3 -c 'import time; print(time.time())' > "${prefix}_start_unix.txt"
    set +e
    /usr/bin/time -f 'wall_sec=%e\ncpu_user_sec=%U\ncpu_sys_sec=%S\npeak_host_rss_kb=%M' \
      -o "${prefix}_time.txt" \
      env PIXVQA_CVQA_NR_PROFILE=1 "$BINARY" \
        --metrics cvqa-hdr-nr \
        --hdr-transfer sdr --hdr-primaries bt709 --hdr-matrix bt709 --hdr-range limited \
        --cvqa-nr-chunk-size-sec 8 \
        --cvqa-nr-sample-interval-sec "$interval" \
        --cvqahdr-output-json "${prefix}_detail.json" \
        -v 1 -t "$VIDEO" \
        > "${prefix}_pixvqa.log" 2>&1
    status=$?
    set -e
    python3 -c 'import time; print(time.time())' > "${prefix}_end_unix.txt"
    cleanup
    MON_PID=''
    printf '%s\n' "$status" > "${prefix}_exit_code.txt"
    if ((status != 0)); then
      echo "PixVQA FAILED (exit=$status). See ${prefix}_pixvqa.log" >&2
      exit "$status"
    fi
    if [[ ! -s "${prefix}_detail.json" ]]; then
      echo "Missing detailed JSON: ${prefix}_detail.json" >&2
      exit 1
    fi
    cat "${prefix}_time.txt"
  done
done
printf 'Benchmark complete: %s\n' "$OUT"
printf 'To summarize: python3 summarize_cvqa_robust_benchmark.py "%s"\n' "$OUT"
