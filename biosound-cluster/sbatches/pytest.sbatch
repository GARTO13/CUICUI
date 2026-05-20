#!/bin/bash
#SBATCH --job-name=bsc-pytest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --account=a127
#SBATCH --output=/capstor/store/cscs/swissai/a127/homes/jdoumeng/CUICUI/biosound-cluster/logs/pytest_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a127/homes/jdoumeng/CUICUI/biosound-cluster/logs/pytest_%j.err

set -euo pipefail

PROJECT=/capstor/store/cscs/swissai/a127/homes/jdoumeng/CUICUI/biosound-cluster
VENV=$PROJECT/.venv

echo "==== Job started: $(date) ===="
echo "Node:    ${SLURMD_NODENAME:-unknown}"
echo "Job:     ${SLURM_JOB_ID:-unknown}"
echo "Project: $PROJECT"
echo "Venv:    $(readlink -f $VENV)"

cd "$PROJECT"

echo "==== Running pytest ===="
"$VENV/bin/python" -m pytest -q

echo "==== Job done: $(date) ===="
