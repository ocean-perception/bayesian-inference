#!/bin/bash

#SBATCH --mem=12G

#SBATCH --ntasks-per-node=10
#SBATCH --time=24:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cappelletto@gmail.com

ml restore blitz
conda activate blitz

LATENT_SIZE=16
LATENT_FILE="data/iridis/latent/latent_h16_TR_ALL.csv"
TARGET_FILE="data/iridis/target/landability/residuals-r020/residual-r020_TR00-04-36.csv"
SUFFIX_ESH="E"${BNN_EPOCHS}"s"${BNN_SAMPLES}"_h"${LATENT_SIZE}
OUT_FILE="residualM3_"${SUFFIX_ESH}".csv"
OUT_NET="trainM3-residual_"${SUFFIX_ESH}".csv"
LOG_FILE="train_residualM3_"${SUFFIX_ESH}".log"

python bnn_train.py --input=${LATENT_FILE} --target=${TARGET_FILE} --key=landability -o ${OUT_FILE} --uuid=uuid -g ${LOG_FILE} -n ${OUT_NET} -e ${BNN_EPOCHS} -s ${BNN_SAMPLES} -x 0.9
