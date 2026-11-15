#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
run=""
next_run=false
for arg in "$@"; do
  if $next_run; then
    run="$arg"
    next_run=false
  elif [[ "$arg" == "--run" || "$arg" == "--ru" || "$arg" == "--r" ]]; then
    next_run=true
  elif [[ "$arg" == --run=* || "$arg" == --ru=* || "$arg" == --r=* ]]; then
    run="${arg#*=}"
  fi
done
if [[ -n "$run" && -f "$run/subset.json" ]]; then
  printf '%s\n' '这是任务子集运行。请使用 main/run_subset.sh，避免执行原17任务队列。' >&2
  exit 2
fi
exec "${PY:-python}" -u -m main.training.main "$@"
