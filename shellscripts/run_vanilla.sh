#!/bin/bash

# assume root as working directory

# default params
set -a
source .env.experiments
set +a

# Prior
export PRIOR=standard

for SEED in $SEEDS
do
    echo "-------------------------------------------------------"
    echo "Starting Standard VAE - Seed: $SEED"
    echo "-------------------------------------------------------"

    # Export the current SEED so the child script can see it
    export SEED=$SEED

    # Execute your child script. 
    bash shellscripts/run_env_experiments.sh
done