import sys
import os
  
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
  
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)

print(sys.path)

# some util funcs
from func.utils import get_model_output_id_wnid_class_dict # get mapping: format: {"Model Ouput ID": ["WNID", "Class"]}
from func.utils import get_imagenet_id_wnid_class_dict # get mapping: format: {"ImageNet ID": ["WNID", "class"]}, e.g. {..."233": ['n02106382', 'Bouvier_des_Flandres'], ...}
from func.utils import map_model_id_to_imagenet_id, map_imagenet_id_to_model_id # mapping funcs
from func.utils import save_obj, load_obj, setup_logger

from func.responsible_regions import load_responsible_regions_from_given_path, X_y_preparation, process_cat_saliency_map

import numpy as np
import argparse
import gc

# get the dict of ImageNet ID, WNID and class name
# format: {"ImageNet ID": ["WNID", "class"]}, e.g. {..."233": ['n02106382', 'Bouvier_des_Flandres'], ...}
imagenet_id_label=get_imagenet_id_wnid_class_dict(matfilepath = "imagenet_info/ILSVRC2012_meta.mat")

# get the dict of model output ID, WNID and class name
# format: {"Model Ouput ID": ["WNID", "Class"]}
modeloutput_id_label=get_model_output_id_wnid_class_dict(jsonfilepath = "imagenet_info/imagenet_label_index.json")

# get dict map model output ID to ImageNet ID
map_dict_model2imagenet=map_model_id_to_imagenet_id(imagenet_id_label, modeloutput_id_label)

# get ImageNet ID to dict map model output ID
map_dict_imagenet2model=map_imagenet_id_to_model_id(imagenet_id_label, modeloutput_id_label)

# get parent children relationship of ImageNet classes
imagenet_class_parent_and_child_dict = load_obj("imagenet_info/imagenet_class_dict")

parser = argparse.ArgumentParser(description='Responsible Region Identification for Concepts')
parser.add_argument('--feature_maps_and_saliency_maps_path', type=str,
                        help='path to get feature maps and saliency maps')
parser.add_argument('--ID_groups', nargs='+', type=int, action='append',
                        help='ImageNet ID group(s)')
parser.add_argument('--layers', nargs='+', type=str,
                        help='which layer(s) to choose')
parser.add_argument('--where_to_save', default="responsible_regions", type=str,
                        help='the path used to save results')
args = parser.parse_args()

#
feature_maps_and_saliency_maps_path = args.feature_maps_and_saliency_maps_path
ID_groups = args.ID_groups
layers = args.layers
responisble_regions_save_path = args.where_to_save
#

if not os.path.exists(responisble_regions_save_path):
    os.makedirs(responisble_regions_save_path)

IDs_str = []
for ID_group in ID_groups:
    IDs_str.append(str(ID_group))
IDs_str = np.array(IDs_str)

for ID_group in ID_groups:
    for ID in ID_group:
        print(ID, imagenet_class_parent_and_child_dict[ID]['words'])

for idx_layer, layer in enumerate(layers):
    chosen_layer = layer
    
    save_layer_path = os.path.join(responisble_regions_save_path, chosen_layer)
    if not os.path.exists(save_layer_path):
        os.mkdir(save_layer_path)
    
    X_dict = {}
    for ID_group in ID_groups:
        for idx, ID in enumerate(ID_group):
            print("loading data", chosen_layer, ID, imagenet_class_parent_and_child_dict[ID]['words'], imagenet_class_parent_and_child_dict[ID]['gloss'])
            if ID not in X_dict.keys():
                X_pos_current, X_neg_current = load_responsible_regions_from_given_path(os.path.join(feature_maps_and_saliency_maps_path, "result_of_ID_"+str(ID)),
                                                                                        layer = chosen_layer, pic_size = 14)
                X_dict[ID] = {}
                X_dict[ID]["foreground"] = X_pos_current
                X_dict[ID]["background"] = X_neg_current
    
    for idx_ID, ID_group in enumerate(ID_groups):
        which_ID_group = idx_ID
        save_layer_ID_group_path = os.path.join(save_layer_path, IDs_str[which_ID_group])
        if not os.path.exists(save_layer_ID_group_path):
            os.mkdir(save_layer_ID_group_path)

        logger = setup_logger("readme", save_layer_ID_group_path)

        flag = True
        idx = 0

        logger.info("For "+save_layer_ID_group_path+"/X_y.npz, X is responsible regions and y is their labels\nFor the labels:")
        # print("\nFor "+save_layer_ID_group_path+"/X_y.npz, the labels: ",end="\n\n")
        for position_in_group, ID in enumerate(ID_group):
            X_pos_current = X_dict[ID]['foreground']
            X_neg_current = X_dict[ID]['background']
            
            logger.info(str(idx)+" presents concept "+str(ID)+" which is "+str(imagenet_class_parent_and_child_dict[ID]))
            # print(str(idx)+" presents concept "+str(ID)+" which is "+str(imagenet_class_parent_and_child_dict[ID]), end="\n\n")
            if flag:
                X = X_pos_current
                y = np.ones(X_pos_current.shape[0]) * idx
                X_neg = X_neg_current
                flag = False
                idx+=1
            else:
                X = np.concatenate((X, X_pos_current))
                y = np.concatenate((y, np.ones(X_pos_current.shape[0]) * idx))
                X_neg = np.concatenate((X_neg, X_neg_current))
                idx+=1
        
        logger.info(str(idx)+" presents background info \n")
        #print(str(idx)+" presents background info")
        X_neg = X_neg[np.random.randint(X_neg.shape[0], size=int(X.shape[0]/len(ID_group))),:]
        X = np.concatenate((X, X_neg))
        y = np.concatenate((y, np.ones(X_neg.shape[0]) * (idx)))
        
        np.savez_compressed(save_layer_ID_group_path+"/X_y.npz", X=X, y=y)