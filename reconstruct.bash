#!/bin/bash

# _JOB_ID 8 character to full path generator. 
# Reconstructs the predicted map, including the required clip limit
# direct prediction --> [0.0, +1.0]
# residual --> [-1.0, +1.0]

# Sample: dM4h6432 --> direct, measurability, 20mm/px, 64 latent, 300 epochs, 10 samples
# We only need to parse the type of prediction, the rest is processed by the [findid] bash script
_JOB_ID=$1

# Let's verify it has 8 character as expected
if [[ ${#_JOB_ID} -lt 8 ]]; then
    echo -e "Invalid _JOB_ID="${_JOB_ID}" definition, at least 8 character length expected"
    exit 1
fi
# Now, we pull the substring for each parameter defined inside _JOB_ID string
_TYPE=${_JOB_ID:0:1}    # Direct or Residual

# let's retrieve the full path to the corresponding job folder
DATA_PATH=$(bash scripts/id2path.bash ${_JOB_ID})
if [[ $? -eq 1 ]]; then
    echo -e "Invalid _JOB_ID="${_JOB_ID}" definition provided"
    exit 1
fi

# Let's check if we have the predicted map all_$JOB_ID
if [[ ! -f ${DATA_PATH}/all_${_JOB_ID}.csv ]]; then
    echo -e "Missing predicted map ${DATA_PATH}/all_${_JOB_ID}.csv"
    exit 1
else
    echo -e "Found predicted map [${DATA_PATH}/all_${_JOB_ID}].csv"
fi

# If the predicted map type is 'residual' we need to calculate the reconstruction by adding the residual map to the low resolution map
if [[ $_TYPE == "r" ]]; then
    # This is only necessary when predicting residuals. We need to reconstruct the map using the low resolution version of the map
    # and the predicted residual map. Join (reconstruct) mut be performed on the uuid column
    _LOWRES_ID=$(sed 's/./b/4' <<< ${_JOB_ID})  # replace the 4th character, which defines the map resolution (b = 500mm/px)
    _LOWRES_ID=$(sed 's/./d/1' <<< ${_LOWRES_ID})  # replace the 1st character, as we need the direct map for (b = 500mm/px)
    LOWRES_FILE=$(bash scripts/id2target.bash ${_LOWRES_ID})
    if [[ $? -eq 1 ]]; then
        echo -e "Invalid _LOWRES_ID="${_LOWRES_ID}" definition provided"
        exit 1
    fi
    if [[ -f ${LOWRES_FILE} ]]; then
        echo "Base resolution map found at ["$LOWRES_FILE"]"
    else
        echo "Base resolution map ["$LOWRES_FILE"] not found!!!"
        exit 1;
    fi
fi

# TODO: Create script/Python that applies clip limit to dataset. This can be done as part of the join step that generates map_$JOB_ID
# TODO: Join base resolution map with predicted map (residual) using uuid and the calculate the reconstruction by sum both columns in the dataframe
exit 0

OUTPUT_FILE="rec_"${_JOB_ID}".csv"

#python join_predictions.py --input ${PREDICT_FILE} --target ${TARGET_FILE} --output ${OUTPUT_FILE} --key "predicted_"${OUT_KEY}
