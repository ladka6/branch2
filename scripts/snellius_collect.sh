#!/bin/bash
#SBATCH --job-name=branch2-collect
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

# Adjust the module/venv lines to your Snellius environment.
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/venvs/branch2/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
cd "$SLURM_SUBMIT_DIR"

for CORPUS in wikitext2 humaneval gsm8k ultrachat; do
  python experiments/collect.py --out "runs/$CORPUS" --corpus "$CORPUS" \
    --sources 2 4 6 8 10 12 14 --depths 2 4 6 8 10 12 14 16 \
    --temperatures 0 0.6 1.0 --contexts 512 --window 32 --batch 8
  python experiments/analyze.py --run "runs/$CORPUS"
done
