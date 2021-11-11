#!/bin/bash

PRETRAIN_CONFIG_FILE=("yelp_preprocess" "amazon_preprocess")
FINETUNE_CONFIG_FILE=("restaurants_preprocess" "laptops_preprocess" "mams_preprocess")

function preprocess() {
  echo "Preprocess $1"
  path="config/$1.yml"
  echo "Config file path: $path"
  python preprocess.py --config $path
}

if [ "$1" == '--pretrain' ]; then
  for file in ${PRETRAIN_CONFIG_FILE[*]}; do
    preprocess $file
  done
elif [ "$1" == '--finetune' ]; then
  for file in ${FINETUNE_CONFIG_FILE[*]}; do
    preprocess $file
  done
else
  echo "Please choose one of the following options: --pretrain, --finetune"
fi
