#!/bin/bash
#SBATCH --job-name="extemp"
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --account=snapsi
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err

# executable
module load jaspy
python -u pipeline_gcms.py reduce $SLURM_ARRAY_TASK_ID > "extemp_log_""$SLURM_ARRAY_TASK_ID"".txt"
