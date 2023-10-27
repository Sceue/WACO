#!/usr/bin/env bash

# Specifying every sbatch parameters will make things easier

SBATCH --nodes=1
SBATCH --ntasks=1
SBATCH --cpus-per-task=16
SBATCH --mem=32GB
SBATCH --gpus=2
SBATCH --constraint=xeon-4116 (some node property to request)
# SBATCH --partition=debug
SBATCH --time=1-2:34:56 (1 day 2 hour 34 min 56 sec)
SBATCH --dependency=afterok:job_id
SBATCH --array=1-3 ($SLURM_ARRAY_TASK_ID)
SBATCH --account=siqiouyang
SBATCH --mail-type=begin|end|fail|all (Email notification)
SBATCH --mail-user=xixu@cs.cmu.edu 
SBATCH --output=stdout.txt
SBATCH --error=stderr.txt

# The rest are your jobs

bash WACO/train/finetune/waco_mt_10h_ft_1h.sh
