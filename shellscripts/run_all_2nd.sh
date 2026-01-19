#!/bin/bash

# Load global defaults
set -a
[ -f .env.experiments ] && source .env.experiments
set +a

# Configuration Arrays
SEEDS=(14 69 420)
K_VALUES=(250 500 1000)
WEIGHTED_OPTS=("True" "False")
FLOW_L=(2 8 16)
FLOW_H=(64 128 256)

COMMANDS_FILE="experiment_queue.txt"
> $COMMANDS_FILE

echo "📝 Generating command queue..."

# 1. Generate VampPrior Commands (18 runs)
for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
        for W in "${WEIGHTED_OPTS[@]}"; do
            # We pass variables as ENV vars at the start of the command string
            echo "SEED=$SEED PRIOR=vampprior K=$K WEIGHTED=$W uv run experiment.py" >> $COMMANDS_FILE
        done
    done
done

# 2. Generate FlowPrior Commands (27 runs)
for SEED in "${SEEDS[@]}"; do
    for L in "${FLOW_L[@]}"; do
        for H in "${FLOW_H[@]}"; do
            echo "SEED=$SEED PRIOR=flowprior L=$L H=$H uv run experiment.py" >> $COMMANDS_FILE
        done
    done
done

TOTAL=$(wc -l < $COMMANDS_FILE)
echo "🚀 Starting $TOTAL experiments (Parallelism: 2 jobs max)..."

# Run with xargs. 
# sh -c is required so that the ENV=VALUE prefix is correctly interpreted by the shell.
cat $COMMANDS_FILE | xargs -I {} -P 2 sh -c "echo 'Running: {}'; {}"

echo "✅ All experiments completed. Running final visualizations..."
uv run utils/train_visualizations.py