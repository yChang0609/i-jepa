# python3 main_probing.py --fname logs/in-tiny_vitc_tiny-8_ep20 --devices cuda:0 
python3 main_vit.py --fname configs/Experiment/IN-tiny/in-tiny_vit_base-8_ep20.yaml --devices cuda:0 
python3 main_vit.py --fname configs/Experiment/IN-tiny/in-tiny_vitc_base-8_ep20.yaml --devices cuda:0 
python3 main_vit.py --fname configs/Experiment/IN-tiny/in-tiny_vit_huge-8_ep20.yaml --devices cuda:0 
python3 main_vit.py --fname configs/Experiment/IN-tiny/in-tiny_vitc_huge-8_ep20.yaml --devices cuda:0 

python3 main_probing.py --fname logs/in-tiny_vit_base-8_ep20 --devices cuda:0 
python3 main_probing.py --fname logs/in-tiny_vitc_base-8_ep20 --devices cuda:0 
python3 main_probing.py --fname logs/in-tiny_vit_huge-8_ep20 --devices cuda:0 
python3 main_probing.py --fname logs/in-tiny_vitc_huge-8_ep20-v2 --devices cuda:0  

# ViT-Base	            
# ViTC-Base-v1	        
# ViT-Huge	          
# ViTC-Huge-v2	      

# JEPA-Base-LB	        
# JEPA(C)-Base-v1-LB	
# JEPA-Huge-LB	        
# JEPA(C)-Huge-v2-LB	