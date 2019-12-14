#!/usr/bin/env bash
#
# This file:
#
#  -
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "ADULT_MONOTONICITY"
#SBATCH --output "ADULT_MONOTONICITY_%a.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --time="2-24:00:00"
#SBATCH --gres=gpu:1
#

source activate UMNN
python experiments/adult/AdultExperiment.py