#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONDONTWRITEBYTECODE=1
exec "$PY" -u -m exp.main "$@"
