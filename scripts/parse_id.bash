#!/bin/bash

# Core JOB_ID parsing script. It verifies the validity of JOB_ID an returns the corresponding parsed output variables
# as a multi-line echo output that can be used in other scripts viar 'read'
# _JOB_ID must follow 8 character convention [t][LL][r][hh][e][k]
# [t]  type of data: (r) for residual or (d) for direct calculation
# [LL] type of target data by 2 character layer name: (M3) landability, (M4) measurability, (A1) hislope
# [r]  data spatial resolution: (u) ultrahigh res 10mm/px, (h) high res 20mm/px, (s) standard res 40mm/px, (l) low res 500mm/px
# [hh] latent vector dimension: 16~64
# [e]  number of training epochs x 100 (e.g. 3 -> 300 epochs) 
# [k]  number of MonteCarlo samples x 5 (e.g. 2 -> 10 samples) 

# Sample: dM4h6432 --> direct, measurability, 20mm/px, 64 latent, 300 epochs, 10 samples

if [[ -z "$1" ]]; then
    echo -e "No JOB_ID provided"
    exit 1
fi

_JOB_ID=$1

# Let's verify it has 8 character as expected
if [[ ${#_JOB_ID} -lt 8 ]]; then
    echo -e "Invalid _JOB_ID="${_JOB_ID}" definition, at least 8 character length expected"
    exit 1
fi
# Now, we pull the substring for each parameter defined inside _JOB_ID string
_TYPE=${_JOB_ID:0:1}
_LAYER=${_JOB_ID:1:2}
_RESOL=${_JOB_ID:3:1}
_LATEN=${_JOB_ID:4:2}
_EPOCH=${_JOB_ID:6:1}
_SAMPL=${_JOB_ID:7:1}

if [ "$_TYPE" == 'd' ]; then
    OUT_TYPE="direct"
#    echo -e "Using ["${OUT_TYPE}"]"
elif [ "$_TYPE" == 'r' ]; then
    OUT_TYPE="residual"
#    echo -e "Using ["${OUT_TYPE}"]"
else
    echo -e "Target type definition unkown. It must be either (d)irect or (r)esidual. Received: ["${_TYPE}"]"
    exit 1;
fi

if [ "$_LAYER" == 'M3' ]; then
    OUT_KEY="landability"
#    echo -e "Training for ["${OUT_KEY}"]"
elif [ "$_LAYER" == 'M4' ]; then
    OUT_KEY="measurability"
elif [ "$_LAYER" == 'A1' ]; then
    OUT_KEY="hislope_log"   # as it appears in the header of the CSV file
#    echo -e "Training for ["${OUT_KEY}"]"
else
    echo -e "Target unknown, expected (M3) landability, (M4) measurability or (A1) hislope. Received: ["${_LAYER}"]"
    exit 1;
fi

if [ "$_RESOL" == 's' ]; then
    RESOLUTION="r040"
#    echo -e "Map resolution ["${RESOLUTION}"]"
elif [ "$_RESOL" == 'h' ]; then
    RESOLUTION="r020"
#    echo -e "Map resolution ["${RESOLUTION}"mm/px]"
elif [ "$_RESOL" == 'b' ]; then
    RESOLUTION="r500"
#    echo -e "Map resolution ["${RESOLUTION}"mm/px]"
else
    echo -e "Unknown map resolution, expected (b)ase 500mm/px, (s)tandard 40mm/px or (h)igh 20mm/px. Received: ["${_RESOL}"]"
    exit 1;
fi

# Easiest ones: Epochs, Samples and Latent
if (( _LATEN < 4 )); then
    echo -e "Latent vector must have more than 4 dimensions. _LATEN = ["$_LATEN"]"
    exit 1;
else
    LATENT_SIZE=${_LATEN}
#    echo -e "Latent size: "${LATENT_SIZE}
fi

# Expand _EPOCH range to admit single-digit hexadecimal (0-9,A-F)
_r=$((16#$_EPOCH))
if (( _r < 1 )); then
    echo -e "Invalid training epoch value, must be single digit hexadecimal (0-9,A-F). _EPOCH = ["$_EPOCH"]"
    exit 1;
else
    BNN_EPOCHS=$((_r*100))
   # echo -e "Epochs: "${BNN_EPOCHS}
fi

if ((_SAMPL < 1)); then
    echo -e "Monte Carlo samples must be a valid single digit decimal [0-9]. Received _SAMPL = ["$_SAMPL"]"
    exit 1;
else
    BNN_SAMPLES=$((_SAMPL*5))
#    echo -e "Samples: "${BNN_SAMPLES}
fi

# export the parsed variables in the same order defined for the 8-character JOB_ID
# _TYPE=${_JOB_ID:0:1}
# _LAYER=${_JOB_ID:1:2}
# _RESOL=${_JOB_ID:3:1}
# _LATEN=${_JOB_ID:4:2}
# _EPOCH=${_JOB_ID:6:1}
# _SAMPL=${_JOB_ID:7:1}

echo $OUT_TYPE
echo $OUT_KEY
echo $RESOLUTION
echo $LATENT_SIZE
echo $BNN_EPOCHS
echo $BNN_SAMPLES
exit 0	# no errors detected
