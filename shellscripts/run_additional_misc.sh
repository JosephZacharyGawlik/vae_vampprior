#!/bin/bash

mkdir -p logs

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

COMMANDS_FAST="queue_fast.txt"
COMMANDS_PIXEL="queue_pixel.txt"
> $COMMANDS_FAST
> $COMMANDS_PIXEL

SEED=14

# ----------------------------
# 1. GENERATE FAST QUEUE (HVAE, ConvHVAE, and VampFlow)
# ----------------------------

# Blocks for HVAE and ConvHVAE
for MODEL in "hvae_2level" "convhvae_2level"; do
    for PRIOR in "${PRIORS[@]}"; do
        if [ "$PRIOR" == "vampprior" ]; then
            for WEIGHTED in "${WEIGHTEDS[@]}"; do
                echo "SEED=$SEED MODEL_NAME=$MODEL PRIOR=$PRIOR WEIGHTED=$WEIGHTED bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FAST
            done
        else
            echo "SEED=$SEED MODEL_NAME=$MODEL PRIOR=$PRIOR bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FAST
        fi
    done
done

# ----------------------------
# 2. GENERATE PIXEL QUEUE (PixelHVAE)
# ----------------------------
for PRIOR in "${PRIORS[@]}"; do
    if [ "$PRIOR" == "vampprior" ]; then
        for WEIGHTED in "${WEIGHTEDS[@]}"; do
            echo "SEED=$SEED MODEL_NAME=pixelhvae_2level PRIOR=$PRIOR WEIGHTED=$WEIGHTED S_SAMPLES=500 bash shellscripts/run_env_experiments.sh" >> $COMMANDS_PIXEL
        done
    else
        echo "SEED=$SEED MODEL_NAME=pixelhvae_2level PRIOR=$PRIOR S_SAMPLES=500 bash shellscripts/run_env_experiments.sh" >> $COMMANDS_PIXEL
    fi
done

# ----------------------------
# EXECUTION PHASE
# ----------------------------

# Run Fast Models (Parallel: 2) - This will finish the VAE/HVAE/ConvHVAE quickly
echo "🚀 Starting FAST experiments (Parallelism: 2)..."
cat $COMMANDS_FAST | xargs -I {} -P 2 sh -c "{} > logs/\$(date +%s%N).log 2>&1"

# Run Pixel Models (Parallel: 1) - This runs sequentially to avoid OOM
echo "🐢 Starting PIXEL experiments (Sequential/Parallelism: 1)..."
cat $COMMANDS_PIXEL | xargs -I {} -P 1 sh -c "echo 'Running: {}'; {}"

# Final Visualizations
echo "📝 All experiments finished. Running final visualizations..."
uv run utils/train_visualizations.py
