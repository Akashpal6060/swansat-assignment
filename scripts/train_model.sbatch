#!/bin/bash
#SBATCH --job-name=swansat_train
#SBATCH --account=cminds_anandi         # Mandatory project account
#SBATCH --partition=cn3_l40s            # Corrected: lowercase 'l' for L40S GPUs
#SBATCH --gres=gpu:1                    # Request 1 GPU (max allowed is 2)
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # No. of tasks
#SBATCH --cpus-per-task=8               # Max 8 CPUs per 1 GPU (Great for PyTorch dataloaders)
#SBATCH --mem=64G                       # Generous RAM request for multi-temporal patches
#SBATCH --time=12:00:00                 # Runtime (12 hours, well under the 24h strict limit)
#SBATCH --output=logs/%x_%j.out         # Stdout log
#SBATCH --error=logs/%x_%j.err          # Stderr log

# 1. Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swansat

# 2. Move to working directory
cd ~/SwanSAT_Assignment

# 3. Create logs directory if it doesn't exist to prevent silent failures
mkdir -p logs

# 4. Run your PyTorch training script
python scripts/03_train_model.py