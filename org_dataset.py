import os
import shutil

# 定义路径
val_dir = '/home/mislab/Desktop/YungChang/i-jepa/dataset/tiny-imagenet-200/val/'
image_dir = os.path.join(val_dir, 'images')
val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

# 创建目标目录
target_dir = '/home/mislab/Desktop/YungChang/i-jepa/dataset/tiny-imagenet-200/organized_val/'
os.makedirs(target_dir, exist_ok=True)
i = 0
# 读取val_annotations.txt并重新组织验证集
with open(val_annotations_file, 'r') as f:
    for line in f:
        i += 1
        parts = line.split('\t')
        img_name = parts[0]
        class_name = parts[1]
        class_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(image_dir, img_name)
        if  i % 100 == 0:
            print(f"{img_path} to {class_dir}")
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(class_dir, img_name))

print('Validation set organized successfully.')