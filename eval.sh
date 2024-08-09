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
export MOUNT_PATH=/path/to/folder
#export SPECIFIED_CONFIG=config_name

./run_script.sh

#unset SPECIFIED_CONFIG
unset TRAIN_IN_TINY
unset MOUNT_PATH