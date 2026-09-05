#!/usr/bin/env bash
# Optional entrypoint that prints the environment a result was produced in
# before running anything, so a saved log always identifies its own provenance.
set -euo pipefail

echo "=== AEGIS reproduction container ==="
python -c "import sys, numpy, aegis; print(f'aegis {aegis.__version__} | python {sys.version.split()[0]} | numpy {numpy.__version__}')"
echo "data root : ${AEGIS_DATA_ROOT:-/app/data}"
echo "results   : ${AEGIS_RESULTS:-/app/results}"
echo "threads   : OMP=${OMP_NUM_THREADS:-unset} hashseed=${PYTHONHASHSEED:-unset}"
echo "===================================="
exec "$@"
