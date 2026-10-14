#!/bin/bash --login
#SBATCH --job-name=2m_trend_plotting
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --partition=Long
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com
#SBATCH --output=/soge-home/users/jesu4127/Paper_1/temperature/logs/overlay_2_%j.out
#SBATCH --error=/soge-home/users/jesu4127/Paper_1/temperature/logs/overlay_2_%j.err

set -euo pipefail

echo "============================================"
echo "CHIRTS overlay job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================"

WORKDIR="/soge-home/users/jesu4127/Paper_1/temperature"
/hn01-home/jesu4127/micromamba/envs/myenv/bin/python "$WORKDIR/2m_trends_monthly_breakdown.py"

echo
echo "Python version:"
python --version

echo
echo "Starting analysis..."
echo "============================================"

python "$WORKDIR/2m_trends_monthly_breakdown.py"

echo
echo "============================================"
echo "Finished: $(date)"
echo "============================================"