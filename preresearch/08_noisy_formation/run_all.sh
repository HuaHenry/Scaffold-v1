#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.."
export PYTHONDONTWRITEBYTECODE=1
exec "${PY:-python}" -u -m preresearch.08_noisy_formation.exp.main "$@"
