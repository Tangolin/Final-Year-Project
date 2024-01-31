#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name=SAGAN
#SBATCH --output=logs/output_%x_%j.out
#SBATCH --error=errors/error_%x_%j.err

export PYTHONUNBUFFERED=1

module purge
module load anaconda
source /home/FYP/shuang033/.bashrc
python main.py --train --train_data_dir ./data/GEI_data --imsize 128 --save_step 100