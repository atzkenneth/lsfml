#!/bin/bash

#SBATCH --job-name=minisci_train     # Job name
#SBATCH --ntasks=2                        # Number of CPU cores
#SBATCH --gpus=1                       # Number of CPU cores
#SBATCH --mem-per-cpu=16000          # Memory per CPU in MB
#SBATCH --time=04:00:00             # Maximum execution time (HH:MM:SS)
#SBATCH --output out_files/out_%A_%a.out      # Standard output
#SBATCH --error out_files/out_%A_%a.out       # Standard error

source ~/.bashrc
source activate gt2

export cmd=$(head -$SLURM_ARRAY_TASK_ID train_array.txt|tail -1)
echo "=========SLURM_COMMAND========="
echo $cmd
echo "=========SLURM_COMMAND========="
eval $cmd
