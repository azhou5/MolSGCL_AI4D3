#!/bin/bash

# Usage:

set -euo pipefail

SMILES_JSON="${1:-}"
OUTPUT_JSON="${2:-}"
N_WORKERS="${3:-20}"
CHEMPROP_ENV= # TODO INSERT ENV NAME 
if [[ -z "$SMILES_JSON" || -z "$OUTPUT_JSON" ]]; then
  echo "Usage: $0 /abs/path/smiles.json /abs/path/output.json [n_workers]" >&2
  exit 1
fi

# Resolve script and repo paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

if command -v module >/dev/null 2>&1; then
  module load conda/miniforge3/24.11.3-0 || true
fi
eval "$(conda shell.bash hook)"


conda activate $CHEMPROP_ENV




# Ensure PYTHONUNBUFFERED for immediate logs
export PYTHONUNBUFFERED=1

python -u "$REPO_ROOT/code/LLM_assisted_plausibility/describe_cli.py" \
  --smiles_json "$SMILES_JSON" \
  --output_json "$OUTPUT_JSON" \
  --n_workers "$N_WORKERS"


