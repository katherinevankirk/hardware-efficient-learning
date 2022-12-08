#!/bin/bash
#
#SBATCH -p shared # partition name (shared, serial_requeue)
#SBATCH -n 1            # number of CPU per node
#SBATCH -N 1            # number of nodes
#SBATCH --mem 10000   # total memory allowed in MegaBytes (1000 MB = 1 GB)
#SBATCH -t 0-10:00  # day-hours:minutes
#SBATCH --job-name="gsu21000s3"
#SBATCH -o su2mmt3sv2_%a.out
#SBATCH -e su2mmt3sv2_%a.err
#SBATCH --array=0-499   # 0-499,  500 jobs
#SBATCH --open-mode=append      # append new output instead of overwriting to *.out and *.err when job restarts

export TASK_ID=$SLURM_ARRAY_TASK_ID 

export JOB_ID=$SLURM_ARRAY_JOB_ID 
export JID=$SLURM_JOB_ID         

# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step
module load Anaconda3/2020.11
python3 -u globalsu2shadowtomography.py "$SLURM_ARRAY_TASK_ID" 1000
