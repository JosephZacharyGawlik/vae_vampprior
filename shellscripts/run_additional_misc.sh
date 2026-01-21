#!/bin/bash

# Load global defaults
set -a
source .env.experiments
set +a

COMMANDS_FILE="experiment_queue.txt"
> $COMMANDS_FILE

echo "📝 Generating command queue for HVAE, convHVAE, pixelHVAE + VampFlowPrior..."

SEED=14
MODELS=("hvae_2level" "convhvae_2level" "pixelhvae_2level")
PRIORS=("standard" "vampprior" "flowprior")
WEIGHTEDS=("False" "True")   # only used for vampprior

# ----------------------------
# HVAE / convHVAE / pixelHVAE with 3 priors
# ----------------------------
for MODEL in "${MODELS[@]}"; do
    for PRIOR in "${PRIORS[@]}"; do
        if [ "$PRIOR" == "vampprior" ]; then
            for WEIGHTED in "${WEIGHTEDS[@]}"; do
                echo "SEED=$SEED MODEL_NAME=$MODEL PRIOR=$PRIOR WEIGHTED=$WEIGHTED bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FILE
            done
            continue
        fi

        echo "SEED=$SEED MODEL_NAME=$MODEL PRIOR=$PRIOR bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FILE
    done
done

# ----------------------------
# VampFlowPrior with standard VAE
# ----------------------------
VAMPFLOW_K=25 
FLOW_H=64
FLOW_D=2
DATASET_NAME="static_mnist"

echo "SEED=$SEED DATASET_NAME=$DATASET_NAME MODEL_NAME=vae PRIOR=vampflowprior NUMBER_COMPONENTS=$VAMPFLOW_K FLOW_HIDDEN_DIM=$FLOW_H FLOW_LAYERS=$FLOW_D WEIGHTED=True bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FILE

# ----------------------------
# Run experiments in parallel (2 jobs at a time)
# ----------------------------
TOTAL=$(wc -l < $COMMANDS_FILE)
echo "🚀 Starting $TOTAL experiments (Parallelism: 2)..."

cat $COMMANDS_FILE | xargs -I {} -P 2 sh -c "echo 'Running: {}'; {}"

# ----------------------------
# Run visualizations after all experiments
# ----------------------------
echo "📝 All experiments finished. Running visualizations..."
uv run utils/train_visualizations.py

echo "✅ All experiments and visualizations completed."
