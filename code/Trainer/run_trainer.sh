#!/bin/bash

#!/bin/bash
#SBATCH -N 1
#SBATCH -t 07:00:00                      # wall time (D-HH:MM:SS)
#SBATCH -p general
#SBATCH -G a100:1
#SBATCH --mem 0                          # when you want to get all the memory on node (not recommended)

#SBATCH -o /scratch/bkandhar/DM-Project/CSE-572-Alignment-Reasoning/Trainer/logs/run_trainer.out                       # STDOUT (%j = JobId)
#SBATCH -e /scratch/bkandhar/DM-Project/CSE-572-Alignment-Reasoning/Trainer/logs/run_trainer.err                       # STDERR (%j = JobId)
##SBATCH -A bkandhar                       # Account hours will be pulled from (commented out with double # in front)
#SBATCH --mail-type=ALL                 # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=bkandhar@asu.edu       # send-to address
#module purge                           # Always purge modules to ensure a consistent environment
#conda activate dm-proj                      # Activate the conda environment

# Check if accelerate is installed
if ! command -v accelerate &>/dev/null; then
    echo "accelerate could not be found. Please install it with 'pip install accelerate'."
    exit
fi

# Number of GPUs to use
NUM_GPUS=1 # Adjust this based on available GPUs

# Launch training with accelerate
accelerate launch --multi_gpu --num_processes $NUM_GPUS train_dpo.py
