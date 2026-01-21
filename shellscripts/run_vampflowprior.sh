#!/bin/bash

# 1. Kill any hung processes to ensure the A40 is ready
pkill -9 python

# 2. Set your specific parameters
SEED=14
DATASET_NAME="freyfaces" 
MODEL_NAME="vae"
PRIOR="vampflowprior"
VAMPFLOW_K=50 
FLOW_H=64
FLOW_D=2
WEIGHTED="True"

echo "🚀 Launching Single VampFlowPrior Experiment..."
echo "Model: $MODEL_NAME | Prior: $PRIOR | Flows: $FLOW_D layers ($FLOW_H dim)"

# 3. Execute directly (No loops, no xargs)
# We export the variables so the sub-shell/python can see them
export SEED DATASET_NAME MODEL_NAME PRIOR
export NUMBER_COMPONENTS=$VAMPFLOW_K
export FLOW_HIDDEN_DIM=$FLOW_H
export FLOW_LAYERS=$FLOW_D
export WEIGHTED

bash shellscripts/run_env_experiments.sh

uv run utils/train_visualizations.py