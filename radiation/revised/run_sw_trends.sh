#!/bin/bash --login
# October SW assumption checks (regional series + per-grid-point maps).
# Needs sw_trends_monthly.py in the same folder. Submit:   sbatch run_sw_assumptions.sh
#SBATCH --job-name=sw_checks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --time=06:00:00
#SBATCH --partition=Short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com
#SBATCH --output=/soge-home/users/jesu4127/Paper_1/radiation/revised/logs/sw_checks_%j.out
#SBATCH --error=/soge-home/users/jesu4127/Paper_1/radiation/revised/logs/sw_checks_%j.err

set -euo pipefail
export PATH="/hn01-home/jesu4127/micromamba/envs/myenv/bin:$PATH"
cd /soge-home/users/jesu4127/Paper_1/radiation/revised

echo "Started $(date) on $(hostname)"
python -u sw_trends_boxplot_time_series.py
echo "Finished $(date)"