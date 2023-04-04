#!/bin/bash
#SBATCH --gres=gpu:2080:1
#SBATCH --partition=students-prod
#SBATCH --time=24:00:00
##SBATCH --array=0-10%1
#SBATCH -e /homes/sprochilo/htr/jobs/fast_htr_lama_%j.err
#SBATCH -o /homes/sprochilo/htr/jobs/fast_htr_lama_%j.out
#SBATCH -J htr_lama_train_bce_adam_2018
#SBATCH --mail-user=242033@studenti.unimore.it
#SBATCH --mail-type=BEGIN


srun /homes/sprochilo/.conda/envs/lama/bin/python3  -u /homes/sprochilo/htr/train.py \
                                                    --experiment_name="bce_adam_2018" \
                                                    --configuration="binary_cross_entropy_adam_2018" \
                                                    --use_wandb=True --train=True
