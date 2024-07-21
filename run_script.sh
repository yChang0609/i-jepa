#!/bin/bash
DIRECTORY="$MOUNT_PATH/experiments"
echo "Script start"
nvidia-smi
for file in "$DIRECTORY"/*
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