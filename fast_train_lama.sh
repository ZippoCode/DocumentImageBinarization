#!/bin/bash
#SBATCH --gres=gpu:2080:1
#SBATCH --partition=students-prod
#SBATCH --time=24:00:00
##SBATCH --array=0-10%1
#SBATCH -e /homes/sprochilo/binarization/jobs/fast_htr_lama_%j.err
#SBATCH -o /homes/sprochilo/binarization/jobs/fast_htr_lama_%j.out
#SBATCH -J lama_train_bce_adam_2018_b18
#SBATCH --mail-user=242033@studenti.unimore.it
#SBATCH --mail-type=BEGIN


srun /homes/sprochilo/.conda/envs/lama/bin/python3  -u /homes/sprochilo/binarization/train.py \
                                                    -en="bce_adam_2018_256_b18" \
                                                    -cfg="configs/training/binary_cross_entropy_adam_2018_256.yaml" \
                                                    -net_cfg="configs/network/network_blocks_18.yaml" \
                                                    --use_wandb=True \
                                                    --train_network=True \
                                                    --patience_time=50 \
