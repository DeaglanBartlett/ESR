#!/usr/bin/env bash
# Submit this script through Glamdring's addqueue.  Its sole positional
# argument is a final unique-equations catalogue.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/unique_equations_<complexity>.txt" >&2
    exit 2
fi

module load python/3.11.4
export LD_LIBRARY_PATH="/usr/local/shared/python/3.11.4/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/extraspace/hdesmond/ESR:${PYTHONPATH:-}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec /usr/local/shared/python/3.11.4/bin/python3 \
    "${script_dir}/benchmark_numerical_fingerprint.py" "$1"
