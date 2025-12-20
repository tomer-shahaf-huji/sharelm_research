#!/bin/bash
#
# SLURM Directives (Resource Requests)
#
#SBATCH --job-name=MatrixCompute
#SBATCH --mem=800M              # Increased memory slightly for NumPy
#SBATCH --cpus-per-task=1       # Use 1 CPU for this single-threaded operation
#SBATCH --time=0:10:00          # 10 minutes maximum runtime
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


# 3. Run the Python script
echo "Starting Python job on $(hostname)..."
python matrix_ops.py
echo "SLURM script finished."
