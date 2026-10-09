#!/usr/bin/env bash
#SBATCH --job-name=id-prior
#SBATCH --account=isaac-utk0526
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --chdir=/lustre/isaac24/proj/UTK0526/idpfn
#SBATCH --output=/lustre/isaac24/proj/UTK0526/idpfn/sbatch/out/id-prior-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aspannba@utk.edu

set -e
: "${STEPS:?Set STEPS}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
ollama_args=()

if [[ "${USE_OLLAMA:-0}" == 1 ]]; then
    export OLLAMA_MODELS=/lustre/isaac24/proj/UTK0526/ollama/models
    export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"
    /lustre/isaac24/proj/UTK0526/ollama/bin/ollama serve > "sbatch/out/ollama-${SLURM_JOB_ID}.log" 2>&1 &
    ollama_pid=$!
    trap 'kill "$ollama_pid" 2>/dev/null || true' EXIT
    sleep 5
    ollama_args=(--id-prior-allow-ollama --id-prior-ollama-model llama3.2:1b --id-prior-ollama-augment-rate 0 --id-prior-ollama-corrupt-rate 0.001)
fi

uv pip install --python .venv/bin/python --torch-backend cu129 'torch==2.12.1+cu129'
uv run --no-sync --no-dev python -c 'import torch; assert torch.cuda.is_available()'

time uv run --no-sync --no-dev python -m experiments.run_benchmark_experiment \
    --train-generator id_prior \
    --random-seed "${SEED:-1337}" \
    --revelio-root /lustre/isaac24/proj/UTK0373 \
    --num-steps "$STEPS" \
    --eval-every 50 \
    --batch-size 2 \
    --record-representation mean_pool \
    --benchmark-name fake_1000 \
    --no-frozen-dgp \
    --max-categories 256 \
    --progress-every 25 \
    --save-models \
    "${ollama_args[@]}"

