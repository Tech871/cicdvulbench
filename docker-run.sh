#!/bin/bash

cd /workspace/experiment || exit

source .env

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B}"
PROBLEM_TYPE="${PROBLEM_TYPE:-multi_label_classification}"
NUM_LABELS=${NUM_LABELS:-0}
CHECKPOINT=${CHECKPOINT:-0}

TARGET="${TARGET:-20,22,78,89,94,269,306,352,400,416,434,502,787,798,918}"
LANGUAGE="${LANGUAGE:-java}"

DEPTH=${DEPTH:-0}
TRAIN_AT_DEPTH="${TRAIN:-qwen_classified}_depth_${DEPTH}"
TEST_AT_DEPTH="${TEST:-osv_dev}_depth_${DEPTH}"

BUCKET="${BUCKET:-data-iaae1}"
RUN="${RUN:-main}"

if [ $RUN == "train" ]; then
  python3 dataset/loader/main.py --source $TRAIN_AT_DEPTH --s3_bucket $BUCKET

  torchrun --standalone --nnodes=1 --nproc-per-node=gpu classifier/qwen/train.py \
    --pretrained_model_path $MODEL_PATH \
    --problem_type $PROBLEM_TYPE \
    --num_labels $NUM_LABELS \
    --target $TARGET \
    --train $TRAIN_AT_DEPTH \
    --s3_bucket $BUCKET \
    --checkpoint $CHECKPOINT;

elif [ $RUN == "test" ]; then
  python3 dataset/loader/main.py --source $TEST_AT_DEPTH --s3_bucket $BUCKET

  python3 classifier/qwen/test.py \
    --pretrained_model_path $MODEL_PATH \
    --problem_type $PROBLEM_TYPE \
    --num_labels $NUM_LABELS \
    --target $TARGET \
    --train $TRAIN_AT_DEPTH \
    --test $TEST_AT_DEPTH \
    --s3_bucket $BUCKET;

elif [ $RUN == "main" ]; then
  echo "{
    \"classifiers\": [{
      \"pretrained_model_path\": \"$MODEL_PATH\",
      \"problem_type\": \"$PROBLEM_TYPE\"
    }],
    \"trains\": [{\"train\": \"$TRAIN_AT_DEPTH\"}],
    \"tests\": [\"$TEST_AT_DEPTH\"],
    \"languages\": [\"$LANGUAGE\"],
    \"targets\": [[$TARGET]],
    \"s3_bucket\": \"$BUCKET\"
  }" >> config.json

  python3 dataset/loader/main.py --source $TRAIN_AT_DEPTH --s3_bucket $BUCKET
  python3 dataset/loader/main.py --source $TEST_AT_DEPTH --s3_bucket $BUCKET

  torchrun --standalone --nnodes=1 --nproc-per-node=gpu classifier/qwen/main.py
fi