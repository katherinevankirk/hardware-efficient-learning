#!/bin/bash
#
#SBATCH -p shared # partition name (shared, serial_requeue)
#SBATCH -n 1		# number of CPU per node
#SBATCH -N 1 		# number of nodes
#SBATCH --mem 100000   # total memory allowed in MegaBytes 180000 (1000 MB = 1 GB)
#SBATCH -t 0-05:40  # day-hours:minutes
#SBATCH --job-name="topological data generation"
#SBATCH -o datatopological%a.out
#SBATCH -e datatopological%a.err
#SBATCH --array=0 	# can also take the format array=1-10,22,34,39-45 for example. 1000 jobs: array=0-99
#SBATCH --open-mode=append 	# append new output instead of overwriting to *.out and *.err when job restarts

export TASK_ID=$SLURM_ARRAY_TASK_ID 

export JOB_ID=$SLURM_ARRAY_JOB_ID 
export JID=$SLURM_JOB_ID 		  

g++ -O3 -std=c++0x torictrivial_datageneration.cpp -o datageneration
./datageneration 10 10000 100 5 13131 1.0 > datatopological.txt
