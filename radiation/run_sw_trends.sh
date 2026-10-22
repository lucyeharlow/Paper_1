#!/bin/bash --login
# SW trend maps (SW down and SW absorbed): a job array with one task per month (1-12),
# run in parallel. Submit from the radiation folder:   sbatch run_sw_trends.sh
# To redo a few months only:   sbatch --array=3,10 run_sw_trends.sh
#SBATCH --job-name=sw_trends
#SBATCH --array=1-12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --time=12:00:00
#SBATCH --partition=Long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com
#SBATCH --output=/soge-home/users/jesu4127/Paper_1/radiation/logs/sw_trends_%A_%a.out
#SBATCH --error=/soge-home/users/jesu4127/Paper_1/radiation/logs/sw_trends_%A_%a.err

set -euo pipefail
export PATH="/hn01-home/jesu4127/micromamba/envs/myenv/bin:$PATH"
cd /soge-home/users/jesu4127/Paper_1/radiation

echo "Month ${SLURM_ARRAY_TASK_ID} on $(hostname), started $(date)"
python -u shortwave_comparison_monthly.py
echo "Finished $(date)"