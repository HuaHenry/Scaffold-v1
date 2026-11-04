#!/usr/bin/env bash
# 实验 01 全流程（在 preresearch/01_pilot_validity 目录下运行：bash run_all.sh）
# 一次运行 = 一个大文件夹： <项目>/experiments/preresearch/01_pilot_validity/<时间戳>[_标签]/
#   里面有 run.json jobs/ precheck/ tune/ compose/ train/ baseline/ analysis/ （可用 FW_RESULTS 改根目录）
# 每次运行 bash run_all.sh 都新建一个大文件夹，旧结果不动。
# 环境变量： GPUS=0,1,2,3  STEPS=6000  RUN_TUNE=1 才做一次性调参（默认跳过，使用 frozen.json 的默认值）
#           LABEL=xxx 给大文件夹加后缀   FW_RUN=<已有大文件夹名> 续跑那一次（已完成的任务自动跳过）
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3}
STEPS=${STEPS:-6000}

echo "== 0. 单元测试"
$PY -m pytest -q

export FW_RUN=${FW_RUN:-$($PY -m exp.new_run ${LABEL:+--label "$LABEL"})}
RUN_DIR=$($PY -m exp.new_run --path)
echo "== 本次运行的结果文件夹: $RUN_DIR"

echo "== 1. 验证集（仅用于训练监控）"
[ -f "${FW_VAL:-../../game_code/output/data_val_v3}/train_state.npz" ] || $PY -m exp.gen_val

echo "== 2. 前置检查 1/2/5（不需要训练好的模型）"
$PY -m exp.precheck --items 1,2,5

if [ "${RUN_TUNE:-0}" = "1" ]; then
  echo "== 3. 一次性调参（00 §13 第 7 项，可选）"
  $PY -m exp.make_jobs tune --steps "$STEPS"
  $PY -m exp.run_queue --jobs "$RUN_DIR/jobs/tune_steps${STEPS}.json" --gpus "$GPUS"
  $PY -m exp.analysis.freeze --steps "$STEPS"
fi

echo "== 4. 基线评测"
for b in copy_last apply_edit random_physics oracle_reseed; do
  $PY -m exp.evaluate --predictor "$b" --splits all --workers 8
done

echo "== 5. 检查 E：process / shuffle 的形成转移占比（短训练）"
$PY -m exp.make_jobs pilot_compose --steps "$STEPS"
$PY -m exp.run_queue --jobs "$RUN_DIR/jobs/pilot_compose_steps${STEPS}.json" --gpus "$GPUS"

echo "== 6. 12 个训练任务 + 评测"
$PY -m exp.make_jobs pilot_validity --steps "$STEPS"
$PY -m exp.run_queue --jobs "$RUN_DIR/jobs/pilot_validity_steps${STEPS}.json" --gpus "$GPUS" --eval-after

echo "== 7. 前置检查 3/4/6（需要 state_s0）"
$PY -m exp.precheck --items 3,4,6 --run state_s0 --steps "$STEPS" || echo "前置检查有未通过项，见 $RUN_DIR/precheck/ 下的结果"

echo "== 8. 分析（A-E）与验收 V1-V5"
$PY -m exp.analysis.pilot_validity --steps "$STEPS"
