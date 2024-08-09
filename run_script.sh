#!/bin/bash
DATASET_DIR="$MOUNT_PATH/dataset"
EXP_DIR="$MOUNT_PATH/experiments"
echo "Script start" 
if [ -z "$TRAIN_IN_TINY" ];then
    TRAIN_IN_TINY=false
fi
if [ -z "$TRAIN_IN_1K" ];then
    TRAIN_IN_1K=false
fi

if $TRAIN_IN_TINY; then
    if [ -d "$DATASET_DIR/tiny-imagenet-200" ]; then
        python3 ./imagenet_handle/org_dataset.py --path $DATASET_DIR/tiny-imagenet-200/val
    else
        echo "not find folder: $DATASET_DIR/tiny-imagenet-200"
        exit 999
    fi
elif $TRAIN_IN_1K; then
    if [ -d "$DATASET_DIR/imagenet-1k" ]; then
        echo "Handle ImageNet-1k"
        ./imagenet_handle/unzip.sh
        python3 ./imagenet_handle/unpack2.py --base_path $DATASET_DIR --target_path imagenet-1k-class
    else
        echo "not find folder: $DATASET_DIR/imagenet-1k"
        exit 999
    fi
else
    echo "Not use ImageNet-1k or tiny"
    exit 999
fi


if [ -z "$SPECIFIED_CONFIG" ];then 
    echo "Not SPECIFIED_CONFIG"
    for file in "$EXP_DIR"/*
    do
        if [ -f "$file" ]; then
            echo "run config:$file"
            logfolder=$(yq .logging.folder $file)
            train_vit_cls=$(yq '.train_type.vit_cls' $file)
            train_jepa_cls=$(yq '.train_type.jepa_cls' $file)
            if $train_vit_cls ;then
                echo "Training vit classification"
                python3 main_vit.py --fname "$file" --devices cuda:0  
            fi
            if $train_jepa_cls ;then
                echo "Training I-JEPA"
                python3 main.py --fname "$file" --devices cuda:0
                echo "Evaluate I-JEPA"
                python3 main_probing.py --fname $MOUNT_PATH/${logfolder//\"/} --devices cuda:0
            fi
        fi
    done
else #Specified
    echo "SPECIFIED_CONFIG"
    config_file=$EXP_DIR/$SPECIFIED_CONFIG
    if [ -f "$config_file" ]; then
        echo "run config:$config_file"
        logfolder=$(yq .logging.folder $config_file)
        train_vit_cls=$(yq '.train_type.vit_cls' $config_file)
        train_jepa_cls=$(yq '.train_type.jepa_cls' $config_file)
        if $train_vit_cls ;then
            echo "Training vit classification"
            python3 main_vit.py --fname "$config_file" --devices cuda:0  
        fi
        if $train_jepa_cls ;then
            echo "Training I-JEPA"
            python3 main.py --fname "$config_file" --devices cuda:0
            echo "Evaluate I-JEPA"
            python3 main_probing.py --fname $MOUNT_PATH/${logfolder//\"/} --devices cuda:0
        fi
    fi
fi
echo "Script end"