#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if (( $# )); then
    if [[ $# -ne 2 || $1 != --tag || -z $2 ]]; then
        echo "usage: bash run_03.sh [--tag LABEL]" >&2
        exit 2
    fi
    LABEL=$2
fi
export FW_CURVE_EXP=07_03_budget_epochs
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3}
SEEDS=${SEEDS:-0}
EPOCHS=5
EPISODES=20480
DATA=${DATA:-../../game_code/output/data_v3_curve_20480}
SOURCE=${SOURCE:-../../game_code/output/data_v3}
PREFIX_SOURCE=${PREFIX_SOURCE:-../../game_code/output/data_v3_curve_4096}
VAL="$DATA/validation"
WORKERS=${WORKERS:-4}
VAL_EXTRA=${VAL_EXTRA:-512}

$PY -m pytest -q -p no:cacheprovider exp/tests
export FW_RUN=${FW_RUN:-$($PY -m exp.new_run ${LABEL:+--label "$LABEL"})}
RUN_DIR=$($PY -m exp.new_run --path)
echo "Experiment 07_03 (5 epochs, 20480 episodes): $RUN_DIR"
$PY -m exp.generate_data --source "$SOURCE" --prefix-source "$PREFIX_SOURCE" --out "$DATA" \
    --val "$VAL" --episodes "$EPISODES" --workers "$WORKERS" --val-extra "$VAL_EXTRA"
BENCH=$($PY -c 'from exp.shared import common; import sys; print(common.resolve(common.load_json(common.resolve(sys.argv[1])/"curve_source.json")["bench"]))' "$DATA")
$PY -m exp.audit --run "$FW_RUN" --data "$DATA" --val "$VAL" --bench "$BENCH" --epochs "$EPOCHS"
$PY -m exp.make_jobs --run "$FW_RUN" --data "$DATA" --val "$VAL" --seeds "$SEEDS" --epochs "$EPOCHS"
JOB_FILE="$RUN_DIR/jobs/curve_epochs${EPOCHS}_seeds${SEEDS//,/-}.json"
$PY -m exp.run_queue --jobs "$JOB_FILE" --gpus "$GPUS" --bench "$BENCH" --eval-after
$PY -m exp.analysis --run "$FW_RUN" --seeds "$SEEDS" --epochs "$EPOCHS" --data "$DATA" --bench "$BENCH"
