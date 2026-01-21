#!/bin/bash

# Here we run experiments with all parameters specified.

uv run experiment.py \
    --seed "$SEED" \
    --dataset_name "$DATASET_NAME" \
    --input_type "$INPUT_TYPE" \
    --model_name "$MODEL_NAME" \
    --prior "$PRIOR" \
    --lr "$LR" \
    --warmup "$WARMUP" \
    --batch_size "$BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --early_stopping_epochs "$EARLY_STOPPING_EPOCHS" \
    --z1_size "$Z1_SIZE" \
    --z2_size "$Z2_SIZE" \
    --input_size $INPUT_SIZE \
    --number_components "$NUMBER_COMPONENTS" \
    --pseudoinputs_mean "$PSEUDOINPUTS_MEAN" \
    --pseudoinputs_std "$PSEUDOINPUTS_STD" \
    --flow_layers "$FLOW_LAYERS" \
    --flow_hidden_dim "$FLOW_HIDDEN_DIM" \
    --S "$S_SAMPLES" \
    --MB "$MB_SIZE" \
    $( [ "$WEIGHTED" == "True" ] && echo "--weighted" ) \
    $( [ "$ACTIVATION" != "None" ] && echo "--activation" ) \
    $( [ "$NO_CUDA" == "True" ] && echo "--no-cuda" ) \
    $( [ "$DYNAMIC_BINARIZATION" == "True" ] && echo "--dynamic_binarization" ) \
    $( [ "$USE_TRAINING_DATA_INIT" == "True" ] && echo "--use_training_data_init" )