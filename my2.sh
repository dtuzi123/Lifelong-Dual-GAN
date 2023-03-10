#!/bin/bash
#SBATCH --job-name=TwinGenerative_Unsupervised_TaskBounders2            # Job name
#SBATCH --mail-type=END,FAIL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fy689@york.ac.uk          # Where to send mail 
#SBATCH --ntasks=1                            # Run a single task  
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --mem=128gb                            # Job memory request
#SBATCH --time=48:00:00                        # Time limit hrs:min:sec
#SBATCH --output=TwinGenerative_Unsupervised_TaskBounders2.log            # Standard output and error log
#SBATCH --partition=gpu                        # select the gpu nodes
#SBATCH --gres=gpu:1                          # select 1 gpu
 
echo "Running gaussian-test on $SLURM_CPUS_ON_NODE CPU cores"

python TwinGenerative_Unsupervised_TaskBounders2.py