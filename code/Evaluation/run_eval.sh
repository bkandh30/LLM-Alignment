#!/bin/bash
#SBATCH -N 1
#SBATCH -t 07:00:00                      # wall time (D-HH:MM:SS)
#SBATCH -p general
#SBATCH -G a100:2
#SBATCH --mem 0                          # when you want to get all the memory on node (not recommended)    

#SBATCH -o /scratch/sudamshu/dm_proj/logs_directory/logs/your_runname.out                       # STDOUT (%j = JobId)
#SBATCH -e /scratch/sudamshu/dm_proj/logs_directory/logs/your_runname.err                       # STDERR (%j = JobId)
##SBATCH -A sudamshu                       # Account hours will be pulled from (commented out with double # in front)
#SBATCH --mail-type=ALL                 # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=sudamshu@asu.edu       # send-to address
#module purge                           # Always purge modules to ensure a consistent environment
#mamba activate DM                      # Activate the conda environment

# Run the program
python3 evaluation.py

