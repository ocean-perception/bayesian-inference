#!/bin/bash

#SBATCH --mem=8G
#SBATCH --ntasks-per-node=2
#SBATCH --time=4:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cappelletto@gmail.com

#ml restore blitz

#source /media/data/install/anaconda3/etc/profile.d/conda.sh
#conda activate blitz

# BNN prediction job. It calls bnn_predict.py to infer onto precalculated latent input vector and pretrained BNN networks
# Script version 2
# _JOB_ID must follow 8 character convention [t][LL][r][hh][e][k]
# [t]  type of data: (r) for residual or (d) for direct calculation
# [LL] type of target data by 2 character layer name: (M3) landability, (M4) measurability
# [r]  data spatial resolution: (u) ultrahigh res 10mm/px, (h) high res 20mm/px, (s) standard res 40mm/px, (l) low res 500mm/px
# [hh] latent vector dimension: 16~64
# [e]  number of training epochs x 100 (e.g. 3 -> 300 epochs) 
# [k]  number of MonteCarlo samples x 5 (e.g. 2 -> 10 samples) 

# Sample: dM4h6432 --> direct, measurability, 20mm/px, 64 latent, 300 epochs, 10 samples

_JOB_ID=$1

# Let's verify it has 8 character as expected
if [[ ${#_JOB_ID} -lt 8 ]]; then
    echo -e "Invalid _JOB_ID="${_JOB_ID}" definition, at least 8 character length expected"
    return -1
fi
# Now, we pull the substring for each parameter defined inside _JOB_ID string
_TYPE=${_JOB_ID:0:1}
_LAYER=${_JOB_ID:1:2}
_RESOL=${_JOB_ID:3:1}
_LATEN=${_JOB_ID:4:2}
_EPOCH=${_JOB_ID:6:1}
_SAMPL=${_JOB_ID:7:1}


# Easiest ones: Epochs, Samples and Latent
if (( _LATEN < 4 )); then
    echo -e "Latent vector must have more than 4 dimensions. _LATEN = ["$_LATEN"]"
    exit 1;
else
    LATENT_SIZE=${_LATEN}
    echo -e "Latent size: "${LATENT_SIZE}
fi

if (( _EPOCH < 1 )); then
    echo -e "Training epochs must be positive. _EPOCH = ["$_EPOCH"]"
    exit 1;
else
    BNN_EPOCHS=$((_EPOCH*100))
    echo -e "Epochs: "${BNN_EPOCHS}
fi

if ((_SAMPL < 1)); then
    echo -e "Monte Carlo samples must be positive. _SAMPL = ["$_SAMPL"]"
    exit 1;
else
    BNN_SAMPLES=$((_SAMPL*5))
    echo -e "Samples: "${BNN_SAMPLES}
fi

if [ "$_TYPE" == 'd' ]; then
    OUT_TYPE="direct"
    echo -e "Using ["${OUT_TYPE}"]"
elif [ "$_TYPE" == 'r' ]; then
    OUT_TYPE="residual"
    echo -e "Using ["${OUT_TYPE}"]"
else
    echo -e "Target type definition unkown. It must be either (d)irect or (r)esidual. Received: ["${_TYPE}"]"
    exit 1;
fi

if [ "$_LAYER" == 'M3' ]; then
    OUT_KEY="predicted_landability"
    echo -e "Training for ["${OUT_KEY}"]"
elif [ "$_LAYER" == 'M4' ]; then
    OUT_KEY="predicted_measurability"
    echo -e "Training for ["${OUT_KEY}"]"
else
    echo -e "Target unknown, expected (M3) landability or (M4) measurability. Received: ["${_LAYER}"]"
    exit 1;
fi

if [ "$_RESOL" == 's' ]; then
    RESOLUTION="r040"
    echo -e "Map resolution ["${RESOLUTION}"]"
elif [ "$_RESOL" == 'h' ]; then
    RESOLUTION="r020"
    echo -e "Map resolution ["${RESOLUTION}"mm/px]"
else
    echo -e "Unknown map resolution, expected (s)tandard 40mm/px or (h)igh 20mm/px. Received: ["${_RESOL}"]"
    exit 1;
fi

# This should pull the latent vector for all the transects overing the whole map (careful with potential overlaps)
LATENT_FILE="data/iridis/latent/latent_h"${LATENT_SIZE}"_TR_ALL.csv"
# The target file is no longer required as we will not train a new network
#TARGET_FILE="data/iridis/target/"${OUT_KEY}"/"${OUT_TYPE}"-"${RESOLUTION}"/"${_LAYER}"_"${OUT_TYPE}"_"${RESOLUTION}"_TR00-06-36.csv"

# to avoid replacing the original predictions (performed at the end of the training stage with the T:V data), we create a new output for the whole map
OUT_FILE="all_"${_JOB_ID}".csv"
TRAINED_NET="results/"${RESOLUTION}"/"${OUT_KEY}"/"${OUT_TYPE}"/"${_JOB_ID}"/net_"${_JOB_ID}".pth"

python bnn_predict.py --input ${LATENT_FILE} --network ${TRAINED_NET} --samples ${BNN_SAMPLES} --output ${OUT_FILE} --scale 0.1

# We could use a different number of MC samples for the MLE. It does not need to match the same MC samples used for training

# python bnn_predict.py --input=data/iridis/latent/latent_h16_TR_ALL.csv 
# --network=/media/cappelletto/ssd480/experiments/iridis5/bnn-analysis/iridis-results/r040/landability/direct/dM3s1631/net_dM3s1631.pth 
# -s 2 
# --output=out_dM3s1631_RENAME.csv
