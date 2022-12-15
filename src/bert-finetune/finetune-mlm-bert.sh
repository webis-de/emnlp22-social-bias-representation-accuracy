#! /bin/bash

ORIENTATION="left"

python ./run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file "../../data/processed/corpus-awd-lstm-format/${ORIENTATION}/train.txt" \
    --validation_file "../../data/processed/corpus-awd-lstm-format/${ORIENTATION}/valid.txt" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --do_train \
    --do_eval \
    --report_to "wandb" \
    --cache_dir "./data-cache/${ORIENTATION}" \
    --output_dir "../../data/models/bert-finetuned/${ORIENTATION}"
