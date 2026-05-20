#!/bin/bash
#SBATCH --job-name=bsc-dcase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --account=a127
#SBATCH --output=/capstor/store/cscs/swissai/a127/homes/jdoumeng/CUICUI/biosound-cluster/logs/dcase_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a127/homes/jdoumeng/CUICUI/biosound-cluster/logs/dcase_%j.err

set -euo pipefail

PROJECT=/capstor/store/cscs/swissai/a127/homes/jdoumeng/CUICUI/biosound-cluster
VENV=$PROJECT/.venv
DATASET=$PROJECT/data/dcase2024_task5/Development_Set

# Redirect HOME to scratch — /users/jdoumeng is over quota, and panns-inference
# and HF caches default to $HOME/.cache or $HOME/panns_data.
export HOME=/iopsstor/scratch/cscs/jdoumeng/panns_home
export HF_HOME=/iopsstor/scratch/cscs/jdoumeng/hf
mkdir -p "$HOME" "$HF_HOME"

SUBSET=${SUBSET:-ME}
SPLIT=${SPLIT:-validation}
MAX_FILES=${MAX_FILES:-2}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE:-5}
AUTO_PROFILE=${AUTO_PROFILE:-0}
SEMANTIC=${SEMANTIC:-0}
DENOISER=${DENOISER:-0}
RUN_TAG=${RUN_TAG:-quick_${SUBSET,,}_${MAX_FILES}}
OUT_DIR=$PROJECT/outputs/eval_${RUN_TAG}_${SLURM_JOB_ID:-local}

EXTRA_FLAGS=""
[ "$AUTO_PROFILE" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --enable-auto-profile"
[ "$SEMANTIC" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --enable-semantic-tagging"
[ "$DENOISER" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --enable-denoiser"

echo "==== Job started: $(date) ===="
echo "Node:        ${SLURMD_NODENAME:-unknown}"
echo "Job:         ${SLURM_JOB_ID:-unknown}"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-?}"
echo "Subset:      $SUBSET"
echo "Split:       $SPLIT"
echo "Max files:   $MAX_FILES"
echo "Min cluster: $MIN_CLUSTER_SIZE"
echo "Auto-profile: $AUTO_PROFILE   Semantic: $SEMANTIC   Denoiser: $DENOISER"
echo "Extra flags:  $EXTRA_FLAGS"
echo "Output:      $OUT_DIR"

cd "$PROJECT"
mkdir -p "$OUT_DIR"

echo "==== Running biosound-evaluate-dcase ===="
"$VENV/bin/biosound-evaluate-dcase" \
  --dataset-dir "$DATASET" \
  --output-dir "$OUT_DIR" \
  --split "$SPLIT" \
  --subset "$SUBSET" \
  --max-files "$MAX_FILES" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" \
  --no-clips \
  --force \
  $EXTRA_FLAGS

echo "==== Eval summary ===="
if [ -f "$OUT_DIR/evaluation_summary.json" ]; then
  "$VENV/bin/python" -c "
import json, sys
with open('$OUT_DIR/evaluation_summary.json') as f:
    s = json.load(f)
print(json.dumps(s, indent=2)[:2000])
"
fi

echo "==== Job done: $(date) ===="
