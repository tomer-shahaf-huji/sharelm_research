#!/bin/bash
#
# SLURM Directives (Resource Requests)

# --- Job Identification & Resources ---
#SBATCH --job-name=TorchGPUMatmul
#SBATCH --partition=short         # CHANGE: Use the name of your GPU partition (e.g., 'gpu', 'g2_group')
#SBATCH --mem=8G                  # Request 8 GB of host RAM
#SBATCH --cpus-per-task=1         # Use 1 CPU core for management
#SBATCH --time=0:30:00            # Max runtime 30 minutes
#SBATCH --gres=gg:g0:1           # CHANGE: Request 1 GPU from group 'g0'


#SBATCH --output=slurm_jobs/logs/%x_%j.out
#SBATCH --error=slurm_jobs/logs/%x_%j.err

#
# Setup Environment and Run Command
#

# 1. Change to the directory where you submitted the script (good practice)
cd /cs/labs/oabend/tomer.shahaf/slurm_jobs

# 2. Activate your Conda Environment (assuming 'sharelm_research_env')
source /cs/labs/oabend/tomer.shahaf/miniconda3/etc/profile.d/conda.sh
conda activate sharelm_research_env


# 3. Run the PyTorch script
echo "Starting PyTorch GPU job on $(hostname) with $CUDA_VISIBLE_DEVICES..."
python torch_matmul_dummy.py
echo "SLURM script finished."