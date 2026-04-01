#!/bin/bash

declare -A dataset_config

dataset_config["Health"]="1024 16 20"

datasets=("Health")

for dataset in "${datasets[@]}"; do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Forecasting" ]; then
        mkdir logs/Forecasting
    fi
    if [ ! -d "logs/Forecasting/$dataset" ]; then
        mkdir logs/Forecasting/$dataset
    fi

    IFS=' ' read -r seq_len batch_size train_epochs <<< "${dataset_config[$dataset]}"

    if [ "$dataset" == "Energy" ] || [ "$dataset" == "Health" ]; then
        horizons=("12" "24" "36" "48")
    else
        horizons=("48" "96" "192" "336")
    fi


    for horizon in "${horizons[@]}"; do
        inference_token_len=${horizon}
        python run_longExp.py \
            --train_epochs ${train_epochs} \
            --seq_len ${seq_len} \
            --pred_len ${horizon} \
            --inference_token_len ${inference_token_len} \
            --data ${dataset} \
            --data_path "${dataset}.csv" \
            --batch_size ${batch_size} \
            >logs/Forecasting/$dataset/'_seq_len'$seq_len'_inference_token_len'$inference_token_len'_pred_len'$horizon.log
    done
done