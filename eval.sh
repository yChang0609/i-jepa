# Tiny
python3 main_vit.py --fname configs/Experiment/tiny_model/in-tiny_vit_tiny-8_ep100.yaml --devices cuda:0  
# python3 main_vit.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v1.yaml --devices cuda:0 
# python3 main_vit.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v2.yaml --devices cuda:0 

python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vit_tiny-8_ep100.yaml --devices cuda:0
python3 main_probing.py --fname logs/tiny/in-tiny_vit-t8_ep100/ --devices cuda:0
# python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v1.yaml --devices cuda:0 
# python3 main_probing.py --fname logs/tiny/in-tiny_vitc-t8_ep100-v1/ --devices cuda:0 
# python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v2.yaml --devices cuda:0 
# python3 main_probing.py --fname logs/tiny/in-tiny_vitc-t8_ep100-v2/ --devices cuda:0 

# Extra
python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v3.yaml  --devices cuda:0
python3 main_probing.py --fname logs/tiny/in-tiny_vitc-t8_ep100-v3/ --devices cuda:0 
python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vitc_tiny-8_ep100-v4.yaml  --devices cuda:0
python3 main_probing.py --fname logs/tiny/in-tiny_vitc-t8_ep100-v4/ --devices cuda:0 
## sd
python3 main_vit.py --fname configs/Experiment/tiny_model/in-tiny_vit_sd-8_ep100.yaml --devices cuda:0
python3 main_vit.py --fname configs/Experiment/tiny_model/in-tiny_vitc_sd-8_ep100-v1.yaml --devices cuda:0

python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vit_sd-8_ep100.yaml  --devices cuda:0
python3 main_probing.py --fname logs/tiny/in-tiny_vit_sd-8_ep100/ --devices cuda:0 
python3 main.py --fname configs/Experiment/tiny_model/in-tiny_vitc_sd-8_ep100-v1.yaml  --devices cuda:0
python3 main_probing.py --fname logs/tiny/in-tiny_vitc_sd-8_ep100-v1/ --devices cuda:0 


# Base & Huge
# python3 main_probing.py --fname logs/in-tiny_vitc_tiny-8_ep20 --devices cuda:0 
# python3 main_vit.py --fname configs/Experiment/IN-tiny/in-tiny_vitc_base-8_ep20.yaml --devices cuda:0 
# python3 main_vit.py --fname configs/Experiment/IN-tiny/in-tiny_vit_base-8_ep20.yaml --devices cuda:0 
# python3 main_vit.py --fname configs/Experiment/IN-tiny/in-tiny_vitc_huge-8_ep20.yaml --devices cuda:0 
# python3 main_vit.py --fname configs/Experiment/IN-tiny/in-tiny_vit_huge-8_ep20.yaml --devices cuda:0 

# python3 main_probing.py --fname logs/in-tiny_vitc_tiny-8_ep20 --devices cuda:0 
# python3 main_probing.py --fname logs/in-tiny_vit_tiny-8_ep20 --devices cuda:0 

# python3 main_probing.py --fname logs/in-tiny_vit_base-8_ep20 --devices cuda:0 
# python3 main_probing.py --fname logs/in-tiny_vitc_base-8_ep20 --devices cuda:0 
# python3 main_probing.py --fname logs/in-tiny_vitc_huge-8_ep20-v2 --devices cuda:0  
# python3 main_probing.py --fname logs/in-tiny_vit_huge-8_ep20 --devices cuda:0 

