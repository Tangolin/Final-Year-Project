#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name=test_run_1
#SBATCH --output=logs/output_%x_%j.out
#SBATCH --error=errors/error_%x_%j.err

module purge
module load anaconda
source /home/FYP/shuang033/.bashrc
python generator.py