#!/bin/bash

# 定义包含 .tar.gz 文件的目录
DIR="$MOUNT_PATH/dataset/imagenet-1k/data"
echo "Unzip...."
if [ -d "$DIR/imagenet_train/" ];then
    echo "imagenet_train exists"
else
    mkdir -p "$DIR/imagenet_train/"
    for file in "$DIR"/train_images_*.tar.gz
    do
        if [ -f "$file" ]; then
            echo "Extracting $file..."
            tar -xzf "$file" -C "$DIR/imagenet_train/"
        else
            echo "No .tar.gz files found in $DIR."
        fi
    done
fi

if [ -d "$DIR/imagenet_val/" ];then
    echo "imagenet_val exists"
else
    mkdir -p "$DIR/imagenet_val/"
    echo "Extracting val_images.tar.gz..."
    tar -xzf "$DIR/val_images.tar.gz" -C "$DIR/imagenet_val/"
fi
echo "Unzip end"