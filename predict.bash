#!/bin/bash

# BNN prediction job. It calls bnn_predict.py to infer onto precalculated latent input vector and pretrained BNN networks
# Script version 2
# JOB_ID must follow 8 character convention [t][LL][r][hh][e][k]
# [t]  type of data: (r) for residual or (d) for direct calculation
# [LL] type of target data by 2 character layer name: (M3) landability, (M4) measurability
# [r]  data spatial resolution: (u) ultrahigh res 10mm/px, (h) high res 20mm/px, (s) standard res 40mm/px, (l) low res 500mm/px
# [hh] latent vector dimension: 16~64
# [e]  number of training epochs x 100 (e.g. 3 -> 300 epochs) 
# [k]  number of MonteCarlo samples x 5 (e.g. 2 -> 10 samples) 

# Sample: dM4h6432 --> direct, measurability, 20mm/px, 64 latent, 300 epochs, 10 samples

JOB_ID=$1
if [[ -z "$JOB_ID" ]]; then
    echo "Missing JOB_ID argument"
    exit 1
fi

export IFS=$(echo -e "\n")

{
    read -r OUT_TYPE
    read -r OUT_KEY
    read -r RESOLUTION
    read -r LATENT_SIZE
    read -r BNN_EPOCHS
    read -r BNN_SAMPLES
} <<< $(bash scripts/parse_id.bash $JOB_ID)   # multiline read for the 6 expected variables parsed from JOB_ID

if [[ "$?" -ne "0" ]]; then
    echo "Error parsing JOB_ID"
    echo $OUT_TYPE
    exit 1
fi

# Now, we pull the substring for each parameter defined inside JOB_ID string
_TYPE=${JOB_ID:0:1}
_LAYER=${JOB_ID:1:2}
_RESOL=${JOB_ID:3:1}
_LATEN=${JOB_ID:4:2}
_EPOCH=${JOB_ID:6:1}
_SAMPL=${JOB_ID:7:1}

# This should pull the latent vector for all the transects overing the whole map (careful with potential overlaps)
#LATENT_FILE="data/iridis/latent/latent_h"${LATENT_SIZE}"_TR_ALL.csv"
LATENT_FILE=$(bash scripts/id2latent.bash $JOB_ID)

# to avoid replacing the original predictions (performed at the end of the training stage with the T:V data), we create a new output for the whole map
OUT_FILE="all_"${JOB_ID}".csv"
TRAINED_NET="results/"${RESOLUTION}"/"${OUT_KEY}"/"${OUT_TYPE}"/"${JOB_ID}"/net_"${JOB_ID}".pth"

python bnn_predict.py --input ${LATENT_FILE} --network ${TRAINED_NET} --samples ${BNN_SAMPLES} --output ${OUT_FILE} --key "predicted_"${OUT_KEY} --scale 1.0
