#!/bin/bash

#SBATCH -N 1                        # number of compute nodes
#SBATCH -n 1                        # number of tasks your job will spawn
#SBATCH --mem=64G                    # amount of RAM requested in GiB (2^40)
#SBATCH -p gpu                      # Use gpu partition
#SBATCH -q wildfire                 # Run job under wildfire QOS queue
#SBATCH --gres=gpu:1                # Request two GPUs previous  --gres=gpu:1 new added = =gpu:V100_32:1
#SBATCH -t 0-100:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when a job starts, stops, or fails
#SBATCH --mail-user=trawat2@asu.edu # send-to address

module purge
module load anaconda/py3
source activate /home/trawat2/.conda/envs/nlp

python3 /home/trawat2/LearningProject/Compression-BERT/src/finetuning.py