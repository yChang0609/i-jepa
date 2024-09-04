# Tiny
# python3 main_vit.py --fname configs/Experiment/tiny_model/in-tiny_vit_tiny-8_ep100.yaml --devices cuda:0  
# python3 main_vit.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v1.yaml --devices cuda:0 
# python3 main_vit.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v2.yaml --devices cuda:0 

# python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vit_tiny-8_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname logs/tiny/in-tiny_vit-t8_ep100/ --devices cuda:0
# python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v1.yaml --devices cuda:0 
# python3 main_probing.py --fname logs/tiny/in-tiny_vitc-t8_ep100-v1/ --devices cuda:0 
# python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v2.yaml --devices cuda:0 
# python3 main_probing.py --fname logs/tiny/in-tiny_vitc-t8_ep100-v2/ --devices cuda:0 

export TRAIN_IN_TINY=true
export MOUNT_PATH=/home/mislab/Desktop/YungChang/i-jepa/
#export SPECIFIED_CONFIG=config_name

# python3 main.py --fname experiments/in-tiny_vit_tiny-8_ep100-v0+1.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tuning/in-tiny_vit-t8_ep100-v0+1/ --devices cuda:0

### Tuning 
# ## base line
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_ep100/ --devices cuda:0

# ## A: prediector layer 
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_A6_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_A6_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_A9_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_A9_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_A15_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_A15_ep100/ --devices cuda:0

# ## B: prediector layer 
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_B48_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_B48_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_B96_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_B96_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_B192_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_B192_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_B768_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_B768_ep100/ --devices cuda:0

# ## A+B: prediector layer 
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_AB[6+96]_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_AB[6+96]_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_AB[6+192]_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_AB[6+192]_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_AB[9+96]_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_AB[9+96]_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_AB[9+192]_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_AB[9+192]_ep100/ --devices cuda:0

## C: Encoder attention layer
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_C6_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_C6_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_C9_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_C9_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_C15_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_C15_ep100/ --devices cuda:0

# ## D: Encoder Head
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_D6_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_D6_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_D8_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_D8_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-8_D12_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t8_D12_ep100/ --devices cuda:0

## E: Encoder patch size
# python3 main.py --fname experiments/in-tiny_vit_tiny-2_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t2_ep100/ --devices cuda:0
# python3 main.py --fname experiments/in-tiny_vit_tiny-4_ep100.yaml --devices cuda:0
# python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t4_ep100/ --devices cuda:0

## Best setting
python3 main.py --fname experiments/in-tiny_vit_tiny-4_Best_ep100.yaml --devices cuda:0
python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t4_Best_ep100/ --devices cuda:0

python3 main.py --fname experiments/in-tiny_vit_tiny-4_Best_ep300.yaml --devices cuda:0
python3 main_probing.py --fname $MOUNT_PATH/logs/tiny/in-tiny_vit-t4_Best_ep300/ --devices cuda:0


# ./run_script.sh

#unset SPECIFIED_CONFIG
unset TRAIN_IN_TINY
unset MOUNT_PATH

