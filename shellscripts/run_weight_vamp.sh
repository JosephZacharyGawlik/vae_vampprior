#!/bin/bash

# assume root as working directory

# default params
set -a
source .env.experiments
set +a

# Weighted VampPrior
export PRIOR=vampprior
export WEIGHTED=True

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