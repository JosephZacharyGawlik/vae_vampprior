#!/bin/bash

# Load global defaults
set -a
source .env.experiments
set +a

# Ensure log directory exists so the run doesn't crash
mkdir -p logs

echo "📝 Starting experiment: Standard VAE + VampFlowPrior..."

# ----------------------------
# Configuration for VampFlowPrior
# ----------------------------
SEED=14
DATASET_NAME="static_mnist"
MODEL_NAME="vae"
PRIOR="vampflowprior"
WEIGHTED="True"

# ----------------------------
# Run the experiment
# ----------------------------
# We execute this directly rather than using xargs since it's just one job
SEED=$SEED \
DATASET_NAME=$DATASET_NAME \
MODEL_NAME=$MODEL_NAME \
PRIOR=$PRIOR \
WEIGHTED=$WEIGHTED \
bash shellscripts/run_env_experiments.sh

# ----------------------------
# Run visualizations after the experiment
# ----------------------------
echo "📝 Experiment finished. Running visualizations..."
uv run utils/train_visualizations.py

echo "✅ VampFlowPrior experiment and visualizations completed."