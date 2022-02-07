#!/bin/bash

# _JOB_ID 8 character to full path generator. It can be used to speed-up BNN batch dispatching and sorting results

# Script version 1
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

# Expand _EPOCH range to admit single-digit hexadecimal (0-9,A-F)
_r=$((16#$_EPOCH))
if (( _r < 1 )); then
    echo -e "Invalid training epoch value, must be single digit positive hexadecimal (1-9,A-F). _EPOCH = ["$_EPOCH"]"
    exit 1;
else
    BNN_EPOCHS=$((_r*100))
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
    OUT_KEY="landability"
    echo -e "Training for ["${OUT_KEY}"]"
elif [ "$_LAYER" == 'M4' ]; then
    OUT_KEY="measurability"
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

export DATA_PATH="results/"${RESOLUTION}"/"${OUT_KEY}"/"${OUT_TYPE}"/"${_JOB_ID}"/"
# echo ${DATA_PATH}

# we can generate the full path to predicted CSV starting with all_
PREDICT_FILE=${DATA_PATH}"all_"${_JOB_ID}".csv"      # <-- predictions should be here
if [[ -f ${PREDICT_FILE} ]]; then
    echo "Prediction file found at ["$PREDICT_FILE"]"
else
    echo "Prediction file at ["$PREDICT_FILE"] not found!!!"
    exit 1;
fi

# now we need to contruct the full name to the target (ground truth) file
# canonical path is: data/iridis/target/landability/direct-r020/M3_landability-r020_TR_ALL.csv
# data/iridis/<score>/<type>-<resolution>/<layer>_<type>-<resolution>_TR_ALL.csv
TARGET_FILE="data/iridis/target/"${OUT_KEY}"/"${OUT_TYPE}"-"${RESOLUTION}"/"${_LAYER}"_"${OUT_TYPE}"_"${RESOLUTION}"_TR_ALL.csv"
if [[ -f ${TARGET_FILE} ]]; then
    echo "Target file found at ["$TARGET_FILE"]"
else
    echo "Target file at ["$TARGET_FILE"] not found!!!"
    exit 1;
fi

OUTPUT_FILE="map_"${_JOB_ID}".csv"

python join_predictions.py --input ${PREDICT_FILE} --target ${TARGET_FILE} --output ${OUTPUT_FILE} --key "predicted_"${OUT_KEY}
