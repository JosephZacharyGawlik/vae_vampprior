#!/bin/bash

# Load global defaults
set -a
source .env.experiments
set +a

# Configuration Arrays
SEEDS=(14 69 420)
K_VALUES=(250 500 1000)
WEIGHTED_OPTS=("True" "False")
FLOW_L=(2 8 16)
FLOW_H=(64 128 256)

COMMANDS_FILE="experiment_queue.txt"
> $COMMANDS_FILE

echo "📝 Generating command queue using run_env_experiment.sh..."

# 1. VampPrior Commands
for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
        for W in "${WEIGHTED_OPTS[@]}"; do
            echo "SEED=$SEED PRIOR=vampprior NUMBER_COMPONENTS=$K WEIGHTED=$W bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FILE
        done
    done
done

# 2. FlowPrior Commands
for SEED in "${SEEDS[@]}"; do
    for L in "${FLOW_L[@]}"; do
        for H in "${FLOW_H[@]}"; do
            echo "SEED=$SEED PRIOR=flowprior FLOW_LAYERS=$L FLOW_HIDDEN_DIM=$H bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FILE
        done
    done
done

TOTAL=$(wc -l < $COMMANDS_FILE)
echo "🚀 Starting $TOTAL experiments on $DATASET_NAME (Parallelism: 2)..."

cat $COMMANDS_FILE | xargs -I {} -P 2 sh -c "echo 'Running: {}'; {}"

echo "✅ All experiments completed."
uv run utils/train_visualizations.py