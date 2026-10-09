#!/usr/bin/env bash
#SBATCH --job-name=bank-smoke
#SBATCH --account=isaac-utk0526
#SBATCH --partition=ai-tenn-debug
#SBATCH --qos=ai-tenn-debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:55:00
#SBATCH --chdir=/lustre/isaac24/proj/UTK0526/idpfn
#SBATCH --output=/lustre/isaac24/proj/UTK0526/idpfn/sbatch/out/bank-smoke-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aspannba@utk.edu

set -e
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

uv pip install --python .venv/bin/python --torch-backend cu129 'torch==2.12.1+cu129'
uv run --no-sync --no-dev python -c 'import torch; assert torch.cuda.is_available()'

time uv run --no-sync --no-dev python -m experiments.run_benchmark_experiment \
    --train-generator bank \
    --revelio-root /lustre/isaac24/proj/UTK0373 \
    --random-seed 1337 \
    --num-steps 13 \
    --eval-every 13 \
    --batch-size 2 \
    --record-representation mean_pool \
    --benchmark-name fake_1000 \
    --no-frozen-dgp \
    --max-categories 256 \
    --progress-every 1 \
    --save-models
