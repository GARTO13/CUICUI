#!/bin/bash
#SBATCH --job-name=bsc-probe-denoiser
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --partition=normal
#SBATCH --account=a127
#SBATCH --output=/capstor/store/cscs/swissai/a127/homes/jdoumeng/CUICUI/biosound-cluster/logs/probe_denoiser_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a127/homes/jdoumeng/CUICUI/biosound-cluster/logs/probe_denoiser_%j.err

set -eu  # no pipefail — head pipes would SIGPIPE upstream commands
PROJECT=/capstor/store/cscs/swissai/a127/homes/jdoumeng/CUICUI/biosound-cluster
VENV=$PROJECT/.venv
export HOME=/iopsstor/scratch/cscs/jdoumeng/panns_home
export HF_HOME=/iopsstor/scratch/cscs/jdoumeng/hf
mkdir -p "$HOME" "$HF_HOME"

echo "==== Node info ===="
echo "Host:        $(hostname)"
echo "Date:        $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || true

echo "==== Torch import ===="
"$VENV/bin/python" -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('cuda version:', torch.version.cuda)
"

echo "==== biodenoising import + API inspection ===="
"$VENV/bin/python" -c "
import biodenoising
print('module file:', biodenoising.__file__)
print('top-level names:', [n for n in dir(biodenoising) if not n.startswith('_')])
import pkgutil
for finder, name, ispkg in pkgutil.iter_modules(biodenoising.__path__, biodenoising.__name__ + '.'):
    print(' submodule:', name, '(pkg)' if ispkg else '')
"

echo "==== Try common biodenoising loader patterns ===="
"$VENV/bin/python" <<'PY'
import biodenoising as bd
# Try a few likely API shapes
candidates = [
    'pretrained', 'load_pretrained', 'load_pretrained_denoiser',
    'denoise', 'Denoiser', 'BioDenoiser', 'load_model',
    'master64', 'demucs', 'load',
]
print("Searching top-level for:")
for c in candidates:
    if hasattr(bd, c):
        obj = getattr(bd, c)
        print(f"  bd.{c} -> {type(obj).__name__}: {obj}")
PY

echo "==== Job done: $(date) ===="
