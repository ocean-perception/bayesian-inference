#!/bin/bash

#SBATCH --mem=8G

#SBATCH --ntasks-per-node=5
#SBATCH --time=16:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cappelletto@gmail.com

ml restore blitz
conda activate blitz


# Script version 2
# JOB_ID must follow 8 character convention [t][LL][r][hh][e][k]
# [t]  type of data: (r) for residual or (d) for direct calculation
# [LL] type of target data by 2 character layer name: (M3) landability, (M4) measurability
# [r]  data spatial resolution: (u) ultrahigh res 10mm/px, (h) high res 20mm/px, (s) standard res 40mm/px, (l) low res 500mm/px
# [hh] latent vector dimension: 16~64
# [e]  number of training epochs x 100 (e.g. 3 -> 300 epochs) 
# [k]  number of MonteCarlo samples x 5 (e.g. 2 -> 10 samples) 

# Sample: dM4h6432 --> direct, measurability, 20mm/px, 64 latent, 300 epochs, 10 samples

# Let's verify it has 8 character as expected
if [ ${#JOB_ID} -lt 8 ]
    echo -e "Invalid JOB_ID="${JOB_ID}" definition, at least 8 character length expected"
    return -1
fi

_TYPE=${DD:0:1}
_LAYER=${DD:1:2}
_RESOL=${DD:3:1}
_LATEN=${DD:4:2}
_EPOCH=${DD:6:1}
_SAMPL=${DD:7:1}


# Easiest ones: Epochs, Samples and Latent
if [ _LATEN -lt 4 ]
    echo -e "Latent vector must have more than 4 dimensions. _LATEN = ["$_LATEN"]"
    return -1
fi
if [ _EPOCH -lt 1 ]
    echo -e "Training epochs must be positive. _EPOCH = ["$_EPOCH"]"
    return -1
fi
if [ _SAMPL -lt 1 ]
    echo -e "Monte Carlo samples must be positive. _SAMPL = ["$_SAMPL"]"
    return -1
fi
if [[ _TYPE -eq 'd']]
elif [[ _TYPE -eq 'r' ]]; then
    #statements



BNN_EPOCHS=300
BNN_SAMPLES=5
LATENT_SIZE=64
OUT_KEY="landability"

LATENT_FILE="data/iridis/latent/latent_h"${LATENT_SIZE}"_TR_ALL.csv"
TARGET_FILE="data/iridis/target/"${OUT_KEY}"/direct-r040/M3_landability-r040_TR00-06-36.csv"

# SUFFIX_ESH="E"${BNN_EPOCHS}"s"${BNN_SAMPLES}"h"${LATENT_SIZE}
OUT_FILE="pred_"${JOB_ID}"_"${SUFFIX_ESH}".csv"
OUT_NET="net_"${JOB_ID}"_"${SUFFIX_ESH}".pth"
LOG_FILE="log_"${JOB_ID}"_"${SUFFIX_ESH}".csv"

# python bnn_train.py --input=${LATENT_FILE} --target=${TARGET_FILE} --key=${OUT_KEY} -g ${LOG_FILE} -o ${OUT_FILE} --uuid=uuid -n ${OUT_NET} -e ${BNN_EPOCHS} -s ${BNN_SAMPLES} -x 0.9
