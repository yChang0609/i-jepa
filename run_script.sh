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
        python3 org_dataset.py --path $DATASET_DIR/tiny-imagenet-200/val
    else
        echo "not find folder: $DATASET_DIR/tiny-imagenet-200"
        exit 999
    fi
elif $TRAIN_IN_1K; then
    exit 999
else
    echo "Not use ImageNet-1k or tiny"
    exit 999
fi

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
echo "Script end"