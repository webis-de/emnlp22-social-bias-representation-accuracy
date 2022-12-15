#! /bin/bash

ORIENTATION="center"
BATCH_SIZE=600
RUN_ID="${ORIENTATION}-frage-v1"

python lm/pointer.py \
    --save "../../data/models/frage-lstm/${RUN_ID}--b${BATCH_SIZE}.pt" \
    --cuda \
    --lambdasm 0.16 \
    --theta 1.4 \
    --window 4200 \
    --bptt 2000 \
    --data "../../data/processed/corpus-awd-lstm-format/${ORIENTATION}"
