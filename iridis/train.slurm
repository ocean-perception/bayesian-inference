#!/bin/bash

#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1
#SBATCH --time=28:00:00

#CH --mail-type=ALL
#CH --mail-user=cappelletto@gmail.com

ml restore blitz

source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate blitz

# AT this point we could just call the train.bash script as it already contains the necessary commands to train the model
bash train.bash ${JOB_ID}
exit 0
