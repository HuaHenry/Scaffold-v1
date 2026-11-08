#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if (( $# )); then
    if [[ $# -ne 2 || $1 != --tag || -z $2 ]]; then
        echo "usage: bash run_all.sh [--tag LABEL]" >&2
        exit 2
    fi
    LABEL=$2
fi
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3}
DATA=${DATA:-game_code/output/data_v3_simple_train_hard_eval_81920}

$PY -c 'from exp.shared import common, require_data; import sys; require_data(common.resolve(sys.argv[1]))' "$DATA"
$PY -m pytest -q -p no:cacheprovider exp/tests
export FW_RUN=${FW_RUN:-$($PY -m exp.new_run ${LABEL:+--label "$LABEL"})}
RUN_DIR=$($PY -m exp.new_run --path)
echo "Main experiment 03: $RUN_DIR"
$PY -m exp.audit --run "$FW_RUN" --data "$DATA"
$PY -m exp.make_jobs --run "$FW_RUN"
$PY -m exp.run_queue --run "$FW_RUN" --gpus "$GPUS"
$PY -m exp.analysis --run "$FW_RUN"
