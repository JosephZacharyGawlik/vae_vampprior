#!/bin/bash

# Load global defaults
set -a
source .env.experiments
set +a

# Configuration Arrays
SEEDS=(14 69 420)

COMMANDS_FILE="experiment_queue.txt"
> $COMMANDS_FILE

echo "📝 Generating command queue using run_env_experiment.sh..."

# 1. Vanilla Prior Commands
for SEED in "${SEEDS[@]}"; do
    echo "SEED=$SEED PRIOR=standard bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FILE
done

TOTAL=$(wc -l < $COMMANDS_FILE)
echo "🚀 Starting $TOTAL experiments on $DATASET_NAME (Parallelism: 2)..."

cat $COMMANDS_FILE | xargs -I {} -P 2 sh -c "echo 'Running: {}'; {}"

echo "✅ All experiments completed."
uv run utils/train_visualizations.py