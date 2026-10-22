#!/bin/bash --login
#SBATCH --job-name=eof_pv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --partition=Long
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com
#SBATCH --output=/soge-home/users/jesu4127/Paper_1/exploration/logs/overlay_%j.out
#SBATCH --error=/soge-home/users/jesu4127/Paper_1/exploration/logs/overlay_%j.err

set -euo pipefail

echo "============================================"
echo "PV job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================"

WORKDIR="/soge-home/users/jesu4127/Paper_1/exploration/"
/hn01-home/jesu4127/micromamba/envs/myenv/bin/python "$WORKDIR/pv_omega_detrended_corr.py"
echo
echo "Python version:"
python --version

echo
echo "Starting analysis..."
echo "============================================"

python "$WORKDIR/pv_omega_detrended_corr.py"

echo
echo "============================================"
echo "Finished: $(date)"
echo "============================================"