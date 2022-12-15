#! /bin/bash

ORIENTATION="center"
BATCH_SIZE=600
RUN_ID="${ORIENTATION}-frage-v0"

# In case training needs to be resumed, include this flag in the python script call below
# --resume "../../data/models/frage-lstm/${RUN_ID}--b${BATCH_SIZE}.pt" \

python ./lm/main.py \
  --wandb_id "${RUN_ID}" \
  --epochs 500 \
  --batch_size "${BATCH_SIZE}" \
  --nonmono 5 \
  --data "../../data/processed/corpus-awd-lstm-format/${ORIENTATION}" \
  --cuda \
  --save "../../data/models/frage-lstm/${RUN_ID}--b${BATCH_SIZE}.pt" \
  --dropouti 0.5 \
  --dropouth 0.2 \
  --seed 1882 \
  --moment_split 8000 \
  --moment_lambda 0.02
