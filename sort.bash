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

export JOB_ID=$1

# Let's verify it has 8 character as expected
if [[ ${#JOB_ID} -lt 8 ]]; then
    echo -e "Invalid JOB_ID="${JOB_ID}" definition, at least 8 character length expected"
    exit 1
fi

DATA_PATH=$(./scripts/id2path.bash ${JOB_ID})
_R=$?

if [[ _R -eq 0 ]]; then
    # let's check if the target directory exists
    if [[ ! -d "${DATA_PATH}" ]]; then
        echo "Target directory ["${DATA_PATH}"] not found."
        if [[ "$2" == "-c" ]]; then
            echo " Creating..."
            mkdir -p ${DATA_PATH}
        else
            echo "If you want to create it, use the option '-c' as second argument"
            echo "    $ sort.bash ${DATA_PATH} -c"
            exit 1
        fi     
    fi
    echo -e "Moving files to: ["${DATA_PATH}"]"
    mv *${JOB_ID}* ${DATA_PATH}
else
    echo "Invalid JOB_ID provided: [" ${JOB_ID} "]"
fi
