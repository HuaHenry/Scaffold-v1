#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1
exec "${PY:-python}" -u -m experiments.main.data.generate "$@"
