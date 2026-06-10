#!/bin/bash
# SLURM array job: runs run_site.py for each site in parallel.
#
# Submit all 10 sites:
#   sbatch --array=0-9 run_site.sh
#
# Submit only new sites (skip Madagascar which is already done):
#   sbatch --array=1-9 run_site.sh
#
# Submit a single site:
#   sbatch --array=3 run_site.sh

#SBATCH -J noahmp_causal
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=09:00:00
#SBATCH -o /Net/Groups/BGI/work_1/scratch/npapan/LLM_causal_discovery/NOAH_MP/logs/site_%a.out
#SBATCH -e /Net/Groups/BGI/work_1/scratch/npapan/LLM_causal_discovery/NOAH_MP/logs/site_%a.err

set -euo pipefail

SCRIPT_DIR="/Net/Groups/BGI/work_1/scratch/npapan/LLM_causal_discovery/NOAH_MP"
PYTHON="/User/homes/npapan/miniforge3/envs/ml_basic/bin/python"

mkdir -p "${SCRIPT_DIR}/logs"

echo "========================================"
echo "SLURM job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)  Date: $(date)"
echo "========================================"

cd "${SCRIPT_DIR}"

${PYTHON} run_site.py --site_id ${SLURM_ARRAY_TASK_ID}

echo "Site ${SLURM_ARRAY_TASK_ID} finished at $(date)"
