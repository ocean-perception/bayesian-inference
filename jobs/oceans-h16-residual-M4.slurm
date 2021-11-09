#!/bin/bash

#SBATCH --mem=16G

#SBATCH --ntasks-per-node=5
#SBATCH --time=18:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cappelletto@gmail.com

ml restore blitz
conda activate blitz

BNN_EPOCHS=300
BNN_SAMPLES=5
LATENT_SIZE=16
OUT_KEY="measurability"

LATENT_FILE="data/iridis/latent/latent_h"${LATENT_SIZE}"_TR_ALL.csv"
TARGET_FILE="data/iridis/target/"${OUT_KEY}"/residual-r040/M4_residual-r040_TR00-06-36.csv"

SUFFIX_ESH="E"${BNN_EPOCHS}"s"${BNN_SAMPLES}"_h"${LATENT_SIZE}
OUT_FILE=${JOB_ID}"_"${SUFFIX_ESH}".csv"
OUT_NET="net-"${JOB_ID}"_"${SUFFIX_ESH}".pth"
LOG_FILE=${JOB_ID}"_"${SUFFIX_ESH}".log"

python bnn_train.py --input=${LATENT_FILE} --target=${TARGET_FILE} --key=${OUT_KEY} -g ${LOG_FILE} -o ${OUT_FILE} --uuid=uuid -n ${OUT_NET} -e ${BNN_EPOCHS} -s ${BNN_SAMPLES} -x 0.9
