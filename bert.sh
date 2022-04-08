#!/bin/bash

#SBATCH -N 4                        # number of compute nodes --- Why are you calling for 4 nodes for gpu tasks?
#SBATCH -n 1                        # number of tasks your job will spawn
#SBATCH --mem=64G                    # amount of RAM requested in GiB (2^40) -- for GPU jobs the default 4.5G per core memeory is generally plenty
#SBATCH -p publicgpu                      # Use gpu partition   
#SBATCH -q publicgpu                 # Run job under wildfire QOS queue -- the correct QOS would be wildfire
#SBATCH --gres=gpu:1                # Request two GPUs
#SBATCH -t 0-100:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when a job starts, stops, or fails
#SBATCH --mail-user=trawat2@asu.edu # send-to address

python3 /src/pruning.py

## -- are you calling out GPU enabled modules in bert.py?   also what env are you running that python script in or does it not use any special packages?
