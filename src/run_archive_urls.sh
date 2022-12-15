#! /bin/bash

echo -e "\e[1;31m============================================\e[0m"
echo -e "\e[1;31mStarting URL archiving script.\e[0m"

python src/archive_urls.py \
    --input "data/raw/all_urls.db" \
    --domain_outlet_map "data/raw/domain-outlet-map.json" \
    --batch_size 100 \
    --output_directory "data/processed"

echo -e "\e[1;31m============================================\e[0m"
echo -e "\e[1;31mDone archiving URLs.\e[0m"
