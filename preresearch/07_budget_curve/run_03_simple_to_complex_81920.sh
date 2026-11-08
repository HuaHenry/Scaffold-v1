#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if (( $# )); then
    if [[ $# -ne 2 || $1 != --tag || -z $2 ]]; then
        echo "usage: bash run_03_simple_to_complex_81920.sh [--tag LABEL]" >&2
        exit 2
    fi
    LABEL=$2
fi
export FW_CURVE_EXP=07_03_simple_to_complex_81920
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3}
WORKERS=${WORKERS:-4}
DATA=../../game_code/output/data_v3_simple_train_hard_eval_81920
PREFIX_SOURCE=../../game_code/output/data_v3_simple_train_hard_eval_40960
VAL="$DATA/validation"
EPISODES=81920
EPOCHS=5
SEEDS=0

$PY -m pytest -q -p no:cacheprovider exp/tests
export FW_RUN=${FW_RUN:-$($PY -m exp.new_run ${LABEL:+--label "$LABEL"})}
RUN_DIR=$($PY -m exp.new_run --path)
echo "Experiment 07_03 simple-to-complex 81920: $RUN_DIR"
$PY -u -m exp.generate_data --profile simple_to_complex --prefix-source "$PREFIX_SOURCE" \
    --out "$DATA" --val "$VAL" --episodes "$EPISODES" --workers "$WORKERS" --val-extra 1024 \
    2>&1 | tee "$RUN_DIR/data_generation.log"
BENCH=$($PY -c 'from exp.shared import common; import sys; print(common.resolve(common.load_json(common.resolve(sys.argv[1])/"curve_source.json")["bench"]))' "$DATA")
$PY -m exp.audit --run "$FW_RUN" --data "$DATA" --val "$VAL" --bench "$BENCH" --epochs "$EPOCHS"
$PY -m exp.make_jobs --run "$FW_RUN" --data "$DATA" --val "$VAL" --seeds "$SEEDS" --epochs "$EPOCHS"
JOB_FILE="$RUN_DIR/jobs/curve_epochs${EPOCHS}_seeds${SEEDS}.json"
$PY -m exp.run_queue --jobs "$JOB_FILE" --gpus "$GPUS" --bench "$BENCH" --eval-after
$PY -m exp.analysis --run "$FW_RUN" --seeds "$SEEDS" --epochs "$EPOCHS" --data "$DATA" --bench "$BENCH"
