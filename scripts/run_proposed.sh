#!/usr/bin/env bash
# Dev-set temperature calibration plus the proposed novelty modules.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
    return
  fi
  local dir name candidate
  local IFS=':'
  for dir in $PATH; do
    for name in python3 python; do
      candidate="${dir%/}/$name"
      if [[ -x "$candidate" ]] && "$candidate" -c "import torch, yaml" >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return
      fi
    done
  done
  echo "No Python on PATH can import torch and yaml. Install requirements.txt or set PYTHON." >&2
  return 1
}

PY="$(pick_python)"
"$PY" train.py --config configs/proposed.yaml
"$PY" eval.py --config configs/proposed.yaml --checkpoint checkpoints/best_calibrated_visrag.pt
