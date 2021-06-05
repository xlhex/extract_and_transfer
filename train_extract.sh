#!/bin/bash - 
set -o nounset                              # Treat unset variables as an error
TASK_NAME=$1
SEED=$2
GLUE_DIR=data/$TASK_NAME/review
OUTPUT_DIR="expr/$TASK_NAME/review/seed$SEED"
MODEL_DIR=${TASK_NAME}/review/${SEED}

if [ ! -d $OUTPUT_DIR ];then
    mkdir -p $OUTPUT_DIR
fi

log="$OUTPUT_DIR/log_${SEED}.txt"

python main.py \
    --model_type dstilbert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --seed $2 \
    --output_dir $MODEL_DIR > $log 2>&1
