import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="The Python code used to organize Tiny-ImageNet validation data.")
    parser.add_argument('--path', type=str, required=True, help='validation folder path')
    
    args = parser.parse_args()
    val_dir = args.path #'./dataset/tiny-imagenet-200/val/'
    image_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    target_dir = os.path.join(os.path.dirname(val_dir),'organized_val/')#'./dataset/tiny-imagenet-200/organized_val/'
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        return
    os.makedirs(target_dir, exist_ok=True)
    i = 0
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
if __name__ == '__main__':
    main()