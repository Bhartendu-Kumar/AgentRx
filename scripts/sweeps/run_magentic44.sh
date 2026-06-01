#!/bin/bash
# Magentic-44 sweep — full pipeline with n=3 judge samples per trajectory at
# the release default (paper-faithful PROMPT_STYLE, NL violations included).
#
# This is the full Magentic set used in tab:ablations row band
# "Magentic / One-Shot Constraint Generation" (44 trajectories). The
# tab:ablations row band "Magentic* / Step-by-Step Constraint Generation"
# uses the filtered 27-trajectory subset and has its own driver,
# scripts/sweeps/run_magentic27.sh.
#
# Trajectory IDs are pulled from agentrx.pipeline.globals.MAGENTIC_TASK_IDS
# (the single source of truth in code), not from any JSON file. This way
# the sweep cannot drift from the IDs the rest of the pipeline considers
# canonical.
#
# Inputs : data/magentic_dataset/<trajectory_id>.json (44 IDs)
# Outputs: runs/azure_m44_<trajectory_id>/judge_output/runs/run{1,2,3}.json
# Idempotent: skips trajectories whose run3.json already exists.
#
# Usage (from repo root):
#   scripts/sweeps/run_magentic44.sh
#   nohup scripts/sweeps/run_magentic44.sh > runs/azure_magentic44.log 2>&1 &

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${AGENTRX_PYTHON:-python}"
LOG="${AGENTRX_LOG:-runs/azure_magentic44.log}"
GT="data/ground_truth/magentic_one_ground_truth.json"

mkdir -p "$(dirname "$LOG")"

IDS=$($PY -c "from agentrx.pipeline.globals import MAGENTIC_TASK_IDS; [print(t) for t in MAGENTIC_TASK_IDS]")
TOTAL=$(echo "$IDS" | wc -l | tr -d ' ')
i=0
START_EPOCH=$(date +%s)

echo "==== Magentic-44 sweep started @ $(date -Iseconds) (TOTAL=$TOTAL) ====" | tee -a "$LOG"

for tid in $IDS; do
  i=$((i+1))
  RUN_NAME="azure_m44_${tid}"
  RUN_DIR="runs/$RUN_NAME"
  if [ -f "$RUN_DIR/judge_output/runs/run3.json" ]; then
    echo "[$i/$TOTAL] SKIP (already done) $tid" | tee -a "$LOG"
    continue
  fi
  T0=$(date +%s)
  echo "[$i/$TOTAL] START $tid @ $(date +%H:%M:%S)" | tee -a "$LOG"
  $PY run.py \
    "data/magentic_dataset/${tid}.json" \
    --domain magentic \
    --endpoint azure \
    --ground-truth "$GT" \
    --run-name "$RUN_NAME" \
    >> "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  TOTAL_ELAPSED=$((T1 - START_EPOCH))
  echo "[$i/$TOTAL] DONE  $tid rc=$RC ${ELAPSED}s (cum ${TOTAL_ELAPSED}s)" | tee -a "$LOG"
done

echo "==== Magentic-44 sweep finished @ $(date -Iseconds) ====" | tee -a "$LOG"
