#!/bin/bash

#SBATCH --job-name=hello_word
#SBATCH -n 1
#SBATCH --time=00:03:00
#SBATCH --output=first.out
#SBATCH --error=first.err
#SBATCH -p debug
#SBATCH -N 1






python hello_world.py
