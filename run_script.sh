#!/bin/bash
DIRECTORY="$MOUNT_PATH/experiment"

for file in "$DIRECTORY"/*
do
    if [ -f "$file" ]; then
        if ["$file" == *.yaml];then
            logfolder=$(yq e '.logging.folder' config.yaml)
            python3 main_vit.py --fname "$DIRECTORY/$file" --devices cuda:0  
            python3 main.py --fname "$DIRECTORY/$file" --devices cuda:0
            python3 main_probing.py --fname "$DIRECTORY/$logfolder" --devices cuda:0
        fi
    fi
done