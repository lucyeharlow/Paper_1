#!/bin/bash
#SBATCH --job-name=omega_cf_monthly
#SBATCH --partition=Long
#SBATCH --array=1-12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --output=/soge-home/users/jesu4127/Paper_1/circulation/logs/omega_cf_monthly_%A_%a.out
#SBATCH --error=/soge-home/users/jesu4127/Paper_1/circulation/logs/omega_cf_monthly_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com

# One task per month (task 1 = January ... 12 = December), both datasets.
# The logs folder must exist before submitting:
#   mkdir -p /soge-home/users/jesu4127/Paper_1/circulation/logs
export PATH="/hn01-home/jesu4127/micromamba/envs/myenv/bin:$PATH"

cd /soge-home/users/jesu4127/Paper_1/circulation
python -u monthly_detrended_corr_3_row.py