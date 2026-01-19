# # run quick_eval.py for all model snapshots in a directory
# for model_dir in snapshots/*; do
#     echo "Evaluating model in directory: $model_dir"
#     python utils/quick_eval.py --model_dir "$model_dir"
# done

#!/bin/bash

# 1. Set parallel limit (adjust based on GPU memory and CPU data loading capacity)
MAX_JOBS=2 

# 2. Loop through your 20 snapshots
for model_dir in snapshots/*; do
    echo "Queueing evaluation: $model_dir"
    
    # Run in background and redirect output to a log file within that snapshot
    # This keeps your terminal clean so you don't see 4 models printing at once
    python utils/quick_eval.py --model_dir "$model_dir" > "$model_dir/eval_results/eval_log.txt" 2>&1 &

    # 3. Manage the job queue
    # If the number of running background jobs is >= MAX_JOBS, wait for one to finish
    while [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; do
        sleep 2  # Wait a moment before checking again
    done
done

# 4. Wait for the final batch to finish
wait
echo "All 20 models have been evaluated successfully."
