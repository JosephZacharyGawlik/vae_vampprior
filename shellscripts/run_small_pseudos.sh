#!/bin/bash

# 1. Setup Environment
set -a
[ -f .env.experiments ] && source .env.experiments
set +a

# 2. Configuration for the K=25 Stress Test
SEEDS="14 69 420"
WEIGHTED_OPTS=("True" "False")
export K=25
export PRIOR=vampprior

COMMANDS_FILE="k25_stress_test.txt"
> $COMMANDS_FILE

echo "📝 Building command list for K=25 stress test..."

# 3. Build the queue
for SEED in $SEEDS; do
    for W in "${WEIGHTED_OPTS[@]}"; do
        # We pass the specific SEED and WEIGHTED flag as environment variables
        echo "SEED=$SEED WEIGHTED=$W NUMBER_COMPONENTS=$K PRIOR=$PRIOR bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FILE
    done
done

# 4. Run in Parallel (2 jobs at a time)
TOTAL=$(wc -l < $COMMANDS_FILE)
echo "🚀 Running $TOTAL experiments in parallel (Max 2 jobs)..."

cat $COMMANDS_FILE | xargs -I {} -P 2 sh -c "{}"

echo "✅ All K=25 experiments complete."
uv run utils/train_visualizations.py