#!/bin/bash
#SBATCH --job-name=omega_cf_corr
#SBATCH --partition=Long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --output=/soge-home/users/jesu4127/Paper_1/circulation/logs/omega_cf_corr_%j.out
#SBATCH --error=/soge-home/users/jesu4127/Paper_1/circulation/logs/omega_cf_corr_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com

# The logs folder must exist before submitting:
#   mkdir -p /soge-home/users/jesu4127/Paper_1/circulation/logs
export PATH="/hn01-home/jesu4127/micromamba/envs/myenv/bin:$PATH"

cd /soge-home/users/jesu4127/Paper_1/circulation
python -u merra2_3_row_detrended_corr.py