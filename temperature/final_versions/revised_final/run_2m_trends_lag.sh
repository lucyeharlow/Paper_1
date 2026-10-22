#!/bin/bash --login
#SBATCH --job-name=stats_assumptions
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --partition=Long
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com
#SBATCH --output=/soge-home/users/jesu4127/Paper_1/temperature/final_versions/revised_final/logs/stats_%j.out
#SBATCH --error=/soge-home/users/jesu4127/Paper_1/temperature/final_versions/revised_final/logs/stats_%j.err

set -euo pipefail

echo "============================================"
echo "CHIRTS overlay job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================"

WORKDIR="/soge-home/users/jesu4127/Paper_1/temperature/final_versions/revised_final"
/hn01-home/jesu4127/micromamba/envs/myenv/bin/python "$WORKDIR/lag_autocorr.py"

echo
echo "Python version:"
python --version

echo
echo "Starting analysis..."
echo "============================================"

python "$WORKDIR/lag_autocorr.py"

echo
echo "============================================"
echo "Finished: $(date)"
echo "============================================"