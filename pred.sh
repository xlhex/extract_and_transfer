#!/bin/bash - 
===============================================================================

set -o nounset                              # Treat unset variables as an error

TASK_NAME=$1
GLUE_DIR=data/$TASK_NAME/
SEED=$2
MODEL_dir=

python predict.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_predict \
    --predict_input_file $GLUE_DIR/${3} \
    --predict_output_file $GLUE_DIR/${4} \
    --do_lower_case \
    --data_dir $GLUE_DIR/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --defense $5 \
    --seed $2 \
    --output_dir   #> $output 2>&1
