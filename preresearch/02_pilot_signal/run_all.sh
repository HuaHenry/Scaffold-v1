#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3}
STEPS=${STEPS:-6000}
PILOT01=${PILOT01:?Set PILOT01 to the experiment 01 pilot_validity.json with per-split MDE}
FROZEN=${FROZEN:-../01_pilot_validity/exp/configs/frozen.json}

$PY -m pytest -q -p no:cacheprovider exp/tests
export FW_RUN=${FW_RUN:-$($PY -m exp.new_run ${LABEL:+--label "$LABEL"})}
RUN_DIR=$($PY -m exp.new_run --path)
echo "Experiment 02: $RUN_DIR"

VAL=${FW_VAL:-../../game_code/output/data_val_v3}
if [ ! -f "$VAL/train_state.npz" ]; then
    (cd ../01_pilot_validity && $PY -m exp.gen_val)
fi
$PY -m exp.make_jobs --pilot01 "$PILOT01" --frozen "$FROZEN" --steps "$STEPS"
$PY -m exp.analysis.coverage --run "$FW_RUN"
$PY -m exp.run_queue --jobs "$RUN_DIR/jobs/pilot_signal_steps${STEPS}.json" --gpus "$GPUS" --eval-after
$PY -m exp.analysis.formation_probe --run "$FW_RUN" --steps "$STEPS"
$PY -m exp.analysis.pilot_decision --run "$FW_RUN" --pilot01 "$PILOT01"
