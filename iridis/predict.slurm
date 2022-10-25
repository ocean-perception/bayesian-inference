#!/bin/bash

#SBATCH --mem=8G
#SBATCH --ntasks-per-node=2
#SBATCH --time=4:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cappelletto@gmail.com

ml restore blitz

source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate blitz

# We could just call the predict.bash script as it already contains the necessary commands to infer from the pretrained model
bash predict.bash ${JOB_ID}
exit 0
