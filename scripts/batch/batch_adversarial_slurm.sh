#!/bin/bash
#SBATCH -J batch_adversarial_attack
#SBATCH -A elbaumhpc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --export=ALL

# ==============================================================================
# Configuration
# ==============================================================================

# Seeds to run (space separated)
# Example: "1 2 3 4 5 6 7 8 9 10"
export SEEDS="1 2 3 4 5 6 7 8 9 10"

# Model Checkpoint
# Default path from original script is used if this is empty, but we can override it here.
# export CKPT="/path/to/checkpoint.pt"
# export CKPT="/home/buc9hh/work/research/pcla/agents/dave2/checkpoints/dave2v1_10000_per_town_ep50_aug/dave2v1_10000_per_town_ep50_aug.pt"
export CKPT="/home/buc9hh/work/research/pcla/agents/dave2/checkpoints/dave2v1_1250_per_town_ep50_aug/dave2v1_1250_per_town_ep50_aug.pt"

# Activate conda environment
source "/home/buc9hh/miniconda3/etc/profile.d/conda.sh"
conda activate attn-stability-drive

# ==============================================================================
# Execution
# ==============================================================================

echo "Job config:"
echo "SEEDS: $SEEDS"
echo "CKPT:  $CKPT"
echo "Running on host: $(hostname)"
echo "----------------------------------------------------------------"

# Run the script
/project/AV/attn-stability-drive/scripts/batch/batch_adversarial.sh