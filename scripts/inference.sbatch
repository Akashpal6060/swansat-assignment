#!/bin/bash
#SBATCH --job-name=swansat_infer
#SBATCH --account=cminds_anandi         
#SBATCH --partition=cn3_l40s            
#SBATCH --gres=gpu:1                    
#SBATCH --nodes=1
#SBATCH --ntasks=1                      
#SBATCH --cpus-per-task=8               
#SBATCH --mem=64G                       
#SBATCH --time=01:00:00                 
#SBATCH --output=logs/%x_%j.out         
#SBATCH --error=logs/%x_%j.err          

# 1. Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swansat

# 2. Move to working directory
cd ~/SwanSAT_Assignment

# 3. Create logs directory (just in case)
mkdir -p logs

# 4. Run the inference script (with -u for real-time unbuffered logging)
python -u scripts/04_inference.py