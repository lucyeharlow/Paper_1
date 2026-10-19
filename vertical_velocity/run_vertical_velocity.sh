#!/bin/bash --login
#SBATCH --job-name=f.vertical_velocity_calc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --partition=Long
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com
#SBATCH --output=/soge-home/users/jesu4127/Paper_1/vertical_velocity/fixed/logs/overlay_%j.out
#SBATCH --error=/soge-home/users/jesu4127/Paper_1/vertical_velocity/fixed/logs/overlay_%j.err

set -euo pipefail

echo "============================================"
echo "Vertical velocity calculation job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================"

WORKDIR="/soge-home/users/jesu4127/Paper_1/vertical_velocity/fixed"
/hn01-home/jesu4127/micromamba/envs/myenv/bin/python "$WORKDIR/october_trends.py"
echo
echo "Python version:"
python --version

echo
echo "Starting analysis..."
echo "============================================"

python "$WORKDIR/vertical_velocity_trends.py"

echo
echo "============================================"
echo "Finished: $(date)"
echo "============================================"