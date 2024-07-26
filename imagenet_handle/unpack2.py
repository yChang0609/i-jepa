import json, os, shutil
import argparse

copy_print_frq = 100
def unpack(base_dir="../", 
           target_dir=".", 
           train='imagenet-1k/data/imagenet_train', 
           val='imagenet-1k/data/imagenet_val',
           class_json='imagenet_handle/ImageNet_class_index.json',
           val_label='imagenet_handle/ImageNet_val_label.txt'):
    
    # path
    train_dir = os.path.join(base_dir, train)
    val_dir   = os.path.join(base_dir, val)
    json_dir  = class_json

    target_train_dir = os.path.join(target_dir, 'train')
    target_val_dir   = os.path.join(target_dir, 'organized_val')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(target_train_dir):
        os.mkdir(target_train_dir)
    if not os.path.exists(target_val_dir):
        os.mkdir(target_val_dir)

    # dictionary for class to num
    class2num = {}
    with open(json_dir) as json_file:
        json_data = json.load(json_file)
        for num in json_data:
            class2num[json_data[num][0]] = num
    
    # move training dataset
    i=1
    print("PROCESS TRAIN DATA")
    num_total = len(os.listdir(train_dir))
    for filename in sorted(os.listdir(train_dir)):
        class_id = class2num[filename.split('_')[0]]
        class_dir = os.path.join(target_train_dir, class_id)
        src = os.path.join(train_dir, filename)
        dst = os.path.join(class_dir, filename)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        if os.path.exists(dst):
            print(str(i)+"/"+str(num_total), src, dst, "SKIP")
        else:
            if (i % copy_print_frq) == 0:
                print(str(i)+"/"+str(num_total), src, dst, "COPY")
            shutil.copyfile(src, dst)
        i+=1

    # move val dataset
    i=1
    print("PROCESS VAL DATA")
    num_total = len(os.listdir(val_dir))
    for filename in sorted(os.listdir(val_dir)):
        class_id = class2num[filename.split('_')[-1].split('.')[0]]
        class_dir = os.path.join(target_val_dir, class_id)
        src = os.path.join(val_dir, filename)
        dst = os.path.join(class_dir, filename)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        if os.path.exists(dst):
            print(str(i)+"/"+str(num_total), src, dst, "SKIP")
        else:
            if (i % copy_print_frq) == 0:
                print(str(i)+"/"+str(num_total), src, dst, "COPY")
            shutil.copyfile(src, dst)
        i+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The Python code used to organize Tiny-ImageNet validation data.")
    parser.add_argument('--base_path', type=str, required=True, help='dataset folder path')
    parser.add_argument('--target_path', type=str, required=True, help='target folder path')
    args = parser.parse_args()
    target_dir = os.path.join(args.base_path, args.target_path)
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        exit()

    unpack(args.base_path, target_dir)