#!/bin/bash
#
# SLURM Directives (Resource Requests)

# --- Job Identification & Resources ---
#SBATCH --job-name=SemanticVectorsExtractionChunks
#SBATCH --partition=interactive   # Use a known available partition
#SBATCH --mem=16G                 # Use a safe amount of RAM
#SBATCH --cpus-per-task=4         # Use multiple cores for Dataset I/O
#SBATCH --time=2:00:00            # Set a realistic time limit
#SBATCH --gres=gpu:a10:1

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#
# Setup Environment and Run Command
#

# 1. Change to the directory where you submitted the script (good practice)
#cd /cs/labs/oabend/tomer.shahaf/slurm

# 2. Activate your Conda Environment (assuming 'sharelm_research_env')
source /cs/labs/oabend/tomer.shahaf/miniconda3/etc/profile.d/conda.sh
conda activate sharelm_research_env


# 3. Run the PyTorch script
echo "Starting PyTorch GPU job on $(hostname) with $CUDA_VISIBLE_DEVICES..."
python -u slurm_jobs/extract_semantic_vectors_chunks.py
echo "SLURM script finished."
