#!/usr/bin/env bash
# 实验 01 全流程（在 preresearch/01_pilot_validity 目录下运行：bash run_all.sh）
# 所有结果写到 <项目>/experiments/preresearch/<实验目录>/<时间戳>_<类型>_<名称>_<设置>/ （可用 FW_RESULTS 改根目录）
# 环境变量： GPUS=0,1,2,3  STEPS=6000  RUN_TUNE=1 才做一次性调参（默认跳过，使用 frozen.json 的默认值）
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3}
STEPS=${STEPS:-6000}

echo "== 0. 单元测试"
$PY -m pytest -q

echo "== 1. 验证集（仅用于训练监控）"
[ -f "${FW_VAL:-../../game_code/output/data_val_v3}/train_state.npz" ] || $PY -m exp.gen_val

echo "== 2. 前置检查 1/2/5（不需要训练好的模型）"
$PY -m exp.precheck --items 1,2,5

if [ "${RUN_TUNE:-0}" = "1" ]; then
  echo "== 3. 一次性调参（00 §13 第 7 项，可选）"
  $PY -m exp.make_jobs tune --steps "$STEPS"
  $PY -m exp.run_queue --jobs "exp/jobs/tune_steps${STEPS}.json" --gpus "$GPUS"
  $PY -m exp.analysis.freeze --steps "$STEPS"
fi

echo "== 4. 基线评测"
for b in copy_last apply_edit random_physics oracle_reseed; do
  $PY -m exp.evaluate --predictor "$b" --splits all --workers 8
done

echo "== 5. 检查 E：process / shuffle 的形成转移占比（短训练）"
$PY -m exp.make_jobs pilot_compose --steps "$STEPS"
$PY -m exp.run_queue --jobs "exp/jobs/pilot_compose_steps${STEPS}.json" --gpus "$GPUS"

echo "== 6. 12 个训练任务 + 评测"
$PY -m exp.make_jobs pilot_validity --steps "$STEPS"
$PY -m exp.run_queue --jobs "exp/jobs/pilot_validity_steps${STEPS}.json" --gpus "$GPUS" --eval-after

echo "== 7. 前置检查 3/4/6（需要 state_s0）"
$PY -m exp.precheck --items 3,4,6 --run state_s0 --steps "$STEPS" || echo "前置检查有未通过项，见 experiments/preresearch/00_common/ 下最新的 *_precheck_* 目录"

echo "== 8. 分析（A-E）与验收 V1-V5"
$PY -m exp.analysis.pilot_validity --steps "$STEPS"
