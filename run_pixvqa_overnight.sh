#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/codeccomp/metrics/pixvqa"

LOGDIR="$PWD/artifacts/overnight_qualification"
mkdir -p "$LOGDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

if [[ -x "$PWD/deps/learned-env/bin/python" ]]; then
    export PIXVQA_LEARNED_PYTHON="$PWD/deps/learned-env/bin/python"
fi

STATUS="$LOGDIR/status.tsv"
printf 'Stage\tStatus\n' > "$STATUS"

run_stage() {
    local name="$1"
    shift

    echo "===== START: $name ====="
    printf '%s\tRUNNING\n' "$name" >> "$STATUS"

    if "$@" > "$LOGDIR/$name.log" 2>&1; then
        printf '%s\tPASS\n' "$name" >> "$STATUS"
        echo "===== PASS: $name ====="
    else
        local rc=$?
        printf '%s\tFAIL (%s)\n' "$name" "$rc" >> "$STATUS"
        echo "===== FAILED: $name (exit $rc) ====="
        echo "See $LOGDIR/$name.log"
        exit "$rc"
    fi
}

run_stage smoke_146 \
    python3 scripts/run_real_video_integration.py \
    --phase all --frames 3 --resume --stop-on-fail

run_stage cli_42 \
    python3 scripts/test_metric_cli_conformance.py \
    --strict \
    --cpu-bin build/cpu/pixvqa \
    --swdec-bin build/swdec/pixvqa \
    --json "$LOGDIR/cli_42.json"

run_stage numerical \
    python3 scripts/test_metric_numerical_conformance.py \
    --profile comprehensive --strict \
    --json "$LOGDIR/numerical_comprehensive.json"

run_stage full_146_60f \
    python3 scripts/run_real_video_integration.py \
    --phase all --frames 60 --timeout 900 \
    --resume --stop-on-fail \
    --output artifacts/real_video_integration_60f

printf 'extended_600f\tPENDING\n' >> "$STATUS"

echo "===== AUTOMATED VALIDATION COMPLETED ====="
