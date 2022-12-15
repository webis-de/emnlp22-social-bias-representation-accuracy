#! /bin/bash

echo -e "\e[1;31m============================================\e[0m"
echo -e "\e[1;31mStarting PMI score calculation script.\e[0m"

python src/compute_pmi_scores.py \
    --input "data/raw/all_articles.db" \
    --output_directory "data/processed/pmi-scores" \
    --word_sets "data/raw/word-sets.json" \
    --bigram_window_size 5 \
    --outlet_map "data/raw/allsides-ranking.csv"\
    --bias_type "ethnicity_bias" \
    --target_group "african_american" \
    --year "2020"

echo -e "\e[1;31m============================================\e[0m"
echo -e "\e[1;31mDone calculating PMI scores.\e[0m"
