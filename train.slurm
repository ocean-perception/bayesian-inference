#!/bin/bash

#SBATCH --mem=8G
#sBATCH --ntasks-per-node=1
#SBATCH --time=28:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cappelletto@gmail.com

ml restore blitz

source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate blitz

# AT this point we could just call the train.bash script as it already contains the necessary commands to train the model
bash train.bash ${JOB_ID}
exit 0

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
if [[ ${#JOB_ID} -lt 8 ]]; then
    echo -e "Invalid JOB_ID="${JOB_ID}" definition, at least 8 character length expected"
    return -1
fi
# Now, we pull the substring for each parameter defined inside JOB_ID string
_TYPE=${JOB_ID:0:1}
_LAYER=${JOB_ID:1:2}
_RESOL=${JOB_ID:3:1}
_LATEN=${JOB_ID:4:2}
_EPOCH=${JOB_ID:6:1}
_SAMPL=${JOB_ID:7:1}

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

# if (( _EPOCH < 1 )); then
#     echo -e "Training epochs must be positive. _EPOCH = ["$_EPOCH"]"
#     exit 1;
# else
#     BNN_EPOCHS=$((_EPOCH*100))
#     echo -e "Epochs: "${BNN_EPOCHS}
# fi

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

LATENT_FILE="data/iridis/latent/latent_h"${LATENT_SIZE}"_TR_ALL.csv"

TARGET_FILE="data/iridis/target/"${OUT_KEY}"/"${OUT_TYPE}"-"${RESOLUTION}"/"${_LAYER}"_"${OUT_TYPE}"_"${RESOLUTION}"_TR00-06-36.csv"

# SUFFIX_ESH="E"${BNN_EPOCHS}"s"${BNN_SAMPLES}"h"${LATENT_SIZE}
OUT_FILE="prd_"${JOB_ID}".csv"
OUT_NET="net_"${JOB_ID}".pth"
LOG_FILE="log_"${JOB_ID}".csv"

python bnn_train.py --input=${LATENT_FILE} --target=${TARGET_FILE} --key=${OUT_KEY} -g ${LOG_FILE} -o ${OUT_FILE} --uuid=uuid -n ${OUT_NET} -e ${BNN_EPOCHS} -s ${BNN_SAMPLES} -x 0.9
