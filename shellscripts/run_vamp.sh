#!/bin/bash

# assume root as working directory

# default params
set -a
source .env.experiments
set +a

# --- VAMPPRIOR SPECIFICS ---
export PRIOR=vampprior
export NUMBER_COMPONENTS=500
export PSEUDOINPUTS_MEAN="-0.05"
export PSEUDOINPUTS_STD=0.01
export USE_TRAINING_DATA_INIT=False

for SEED in $SEEDS
do
    echo "-------------------------------------------------------"
    echo "Starting Seed: $SEED"
    echo "-------------------------------------------------------"

    # 3. Export the current SEED so the next script can see it
    export SEED=$SEED

    # Execute your child script. 
    # Because of 'export' above, this script now has access to everything.
    bash shellscripts/run_env_experiments.sh
done