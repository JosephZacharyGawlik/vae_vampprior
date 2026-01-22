#!/bin/bash

mkdir -p logs

# Load global defaults
set -a
[ -f .env.experiments ] && source .env.experiments
set +a

COMMANDS_FILE="hvae_remaining_seeds.txt"
> $COMMANDS_FILE

# Configuration for remaining HVAE runs
REMAINING_SEEDS=(69 420)
MODEL="hvae_2level"
PRIORS=("standard" "vampprior" "flowprior")
WEIGHTEDS=("False" "True")

echo "📝 Generating command queue for HVAE (Remaining Seeds: ${REMAINING_SEEDS[*]})..."

for SEED in "${REMAINING_SEEDS[@]}"; do
    for PRIOR in "${PRIORS[@]}"; do
        if [ "$PRIOR" == "vampprior" ]; then
            for WEIGHTED in "${WEIGHTEDS[@]}"; do
                echo "SEED=$SEED MODEL_NAME=$MODEL PRIOR=$PRIOR WEIGHTED=$WEIGHTED bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FILE
            done
        else
            echo "SEED=$SEED MODEL_NAME=$MODEL PRIOR=$PRIOR bash shellscripts/run_env_experiments.sh" >> $COMMANDS_FILE
        fi
    done
done

# ----------------------------
# EXECUTION PHASE
# ----------------------------

TOTAL=$(wc -l < $COMMANDS_FILE)
echo "🚀 Starting $TOTAL HVAE experiments (Parallelism: 2)..."

# Running with parallelism of 2 to finish faster on the remaining seeds
cat $COMMANDS_FILE | xargs -I {} -P 2 sh -c "echo 'Running: {}'; {} > logs/\$(date +%s%N).log 2>&1"

echo "✅ All HVAE experiments for seeds ${REMAINING_SEEDS[*]} finished."
uv run utils/train_visualizations.py
