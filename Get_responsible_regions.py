import numpy as np
import os
import time

T1 = time.time()

feature_maps_and_saliency_maps_path = "output/feature_maps_and_saliency_maps_vgg19"
ID_groups = [[1082,1095,1845]]
layers = ["features.30"]
responisble_regions_save_path = "output/responsible_regions_vgg19"

layers_str = ""
for layer in layers:
    layers_str += (layer+" ")

ID_groups_str = ""
for ID_group in ID_groups:
    ID_groups_str += " --ID_groups "+(" ".join(map(str,ID_group)))

run_py_script = "python run/get_responsible_regions.py"+\
    " --feature_maps_and_saliency_maps_path "+str(feature_maps_and_saliency_maps_path)+\
        ID_groups_str+\
            " --layers "+str(layers_str)+\
                " --where_to_save "+str(responisble_regions_save_path)

print("run "+run_py_script)
os.system(run_py_script)

T2 = time.time()
print('time consumption: %s s' % ((T2 - T1)))
