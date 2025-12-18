#!/bin/bash
#SBATCH --job-name="extemp"
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --account=snapsi
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err

# executable
module load jaspy
python -u pipeline_gcms.py reduce $SLURM_ARRAY_TASK_ID > logs/extemp_"$SLURM_ARRAY_TASK_ID".log
