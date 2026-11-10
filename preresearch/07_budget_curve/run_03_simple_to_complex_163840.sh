#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if (( $# )); then
    if [[ $# -ne 2 || $1 != --tag || -z $2 ]]; then
        echo "usage: bash run_03_simple_to_complex_163840.sh [--tag LABEL]" >&2
        exit 2
    fi
    LABEL=$2
fi
export FW_CURVE_EXP=07_03_simple_to_complex_163840
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3}
DATA=../../game_code/output/data_v3_simple_train_hard_eval_163840
EPOCHS=5
SEEDS=0

INPUTS=$($PY -c 'from exp.audit import frozen_inputs; import sys; val, bench = frozen_inputs(sys.argv[1]); print(val); print(bench)' "$DATA")
mapfile -t INPUT_PATHS <<< "$INPUTS"
VAL=${INPUT_PATHS[0]}
BENCH=${INPUT_PATHS[1]}
$PY -c 'import os; from exp.shared import EXP_NAME, common; from exp.make_jobs import check_audited_levels; run = os.environ.get("FW_RUN"); path = common.RESULTS_ROOT / EXP_NAME / run / "audit.json" if run else None; check_audited_levels(common.load_json(path)) if path is not None and path.is_file() else None'
$PY -m pytest -q -p no:cacheprovider exp/tests
export FW_RUN=${FW_RUN:-$($PY -m exp.new_run ${LABEL:+--label "$LABEL"})}
RUN_DIR=$($PY -m exp.new_run --path)
echo "Experiment 07_03 simple-to-complex full curve through 163840: $RUN_DIR"
echo "Reusing frozen training, validation and bench; no data will be generated."
echo "Auditing frozen inputs before 52 independent training jobs."
$PY -m exp.audit --run "$FW_RUN" --data "$DATA" --val "$VAL" --bench "$BENCH" --epochs "$EPOCHS"
$PY -m exp.make_jobs --run "$FW_RUN" --data "$DATA" --val "$VAL" --seeds "$SEEDS" --epochs "$EPOCHS"
JOB_FILE="$RUN_DIR/jobs/curve_epochs${EPOCHS}_seeds${SEEDS}.json"
$PY -m exp.run_queue --jobs "$JOB_FILE" --gpus "$GPUS" --bench "$BENCH" --eval-after
$PY -m exp.analysis --run "$FW_RUN" --seeds "$SEEDS" --epochs "$EPOCHS" --data "$DATA" --bench "$BENCH"
