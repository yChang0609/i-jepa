#!/bin/bash
DIRECTORY="$MOUNT_PATH/experiments"
echo "Script start"
nvidia-smi
for file in "$DIRECTORY"/*
do
    if [ -f "$file" ]; then
        echo "run config:$file"
        logfolder=$(yq .logging.folder $file)
        echo "Training vit classification"
        python3 main_vit.py --fname "$file" --devices cuda:0  
        echo "Training I-JEPA"
        python3 main.py --fname "$file" --devices cuda:0
        echo "Evaluate I-JEPA"
        python3 main_probing.py --fname $MOUNT_PATH/${logfolder//\"/} --devices cuda:0
    fi
done
echo "Script end"