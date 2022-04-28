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
from func.utils import get_pretrained_model, load_and_preprocess_image, save_obj, load_obj

# saliency map method funcs
from func.saliency_maps import conduct_saliency_map_method, GuidedBackprop, VanillaBackprop, SmoothGrad, GradCAM, GuidedGradCAM, IntegratedGradients, GradientxInput

import numpy as np
import torch
import os
import random
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

parser = argparse.ArgumentParser(description='Prepare feature maps and saliency maps of the chosen layer of the chosen model used for "Responsible Region Identification for Concepts"')
parser.add_argument('--model_name', default="vgg19", type=str,
                        help='which model to use')
parser.add_argument('--image_path', default="/data/ImageNet_ILSVRC2012/ILSVRC2012_train", type=str,
                        help='path to the images of imagenet')
parser.add_argument('--chosen_IDs', default="1", nargs='+', type=int,
                        help='which ImageNet ID(s) to choose')
parser.add_argument('--layers', default="features.30", nargs='+', type=str,
                        help='which layer(s) to choose')
parser.add_argument('--saliency_method', default="GuidedBackprop", type=str,
                        help='saliency map method')
parser.add_argument('--device', default="cpu", type=str,
                        help='device for pytorch. you can input cpu or cuda or cuda:0 or cuda:1 ...')
parser.add_argument('--where_to_save', default="feature_maps_and_saliency_maps", type=str,
                        help='the path used to save results')
parser.add_argument('--max_img_samples', default=233, type=int,
                        help='max_img_samples')
parser.add_argument('--num_of_img_of_each_class', default=20, type=int,
                        help='num_of_img_of_each_class')
args = parser.parse_args()

#----------
chosen_IDs = args.chosen_IDs

model_name = args.model_name

img_path = args.image_path

which_layer_to_hook = args.layers

Saliency_Method_list = [GuidedBackprop, VanillaBackprop, SmoothGrad, IntegratedGradients, GradientxInput]
Saliency_Method_list_str = ["GuidedBackprop", "VanillaBackprop", "SmoothGrad", "IntegratedGradients", "GradientxInput"]

for i, Saliency_Method_name in enumerate(Saliency_Method_list_str):
    if Saliency_Method_name==args.saliency_method:
        salient_method = Saliency_Method_list[i]
        break

pretrained_model = get_pretrained_model(model_name, True)

device = torch.device(args.device)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("model structure:\n"+str(pretrained_model))

where_to_save = args.where_to_save

if not os.path.exists(where_to_save):
    os.makedirs(where_to_save)
#----------

def get_saliency_map_and_feature_map_cat(pretrained_model, which_layer_to_hook, preprocessed_img_tensor_and_id_list, device,
                                         cat_size, cat_size_for_show = [10,10],
                                         salient_method = GuidedBackprop, num_of_img_for_show = 100):
    saliency_map_list = []
    feature_map_list = []
    for idx, (preprocessed_img_tensor, correct_model_id, resized_img) in enumerate(preprocessed_img_tensor_and_id_list):
        print(idx)
        # GuidedBackprop, VanillaBackprop, SmoothGrad, IntegratedGradients, GradientxInput
        saliency_map, feature_map, _ = conduct_saliency_map_method(salient_method, preprocessed_img_tensor, correct_model_id,
                                                               pretrained_model, which_layer_to_hook=which_layer_to_hook,
                                                               device=device)
        saliency_map_list.append(saliency_map)
        feature_map_list.append(feature_map)

    assert len(saliency_map_list)==cat_size[0]*cat_size[1]
    assert len(feature_map_list)==cat_size[0]*cat_size[1]
    
    resized_img_cat = np.zeros((cat_size[0]*resized_img.shape[0],
                                cat_size[1]*resized_img.shape[1],
                                resized_img.shape[2]), dtype=np.int)
    saliency_map_cat = {}
    feature_map_cat = {}

    for layer in which_layer_to_hook:
        print("processing layer "+str(layer))
        saliency_map_cat[layer] = np.zeros((saliency_map_list[0][layer].shape[0],
                                            cat_size[0]*saliency_map_list[0][layer].shape[1],
                                            cat_size[1]*saliency_map_list[0][layer].shape[2]))
        feature_map_cat[layer] = np.zeros((saliency_map_list[0][layer].shape[0],
                                           cat_size[0]*feature_map_list[0][layer].shape[1],
                                           cat_size[1]*feature_map_list[0][layer].shape[2]))
        for i in range(cat_size[0]):
            for j in range(cat_size[1]):
                idx = i*cat_size[1]+j
                saliency_map = saliency_map_list[idx][layer]
                feature_map = feature_map_list[idx][layer]
                _, _, resized_img = preprocessed_img_tensor_and_id_list[idx]
                resized_img_cat[i*resized_img.shape[0]:(i+1)*resized_img.shape[0],
                                j*resized_img.shape[1]:(j+1)*resized_img.shape[1],:] = resized_img
                saliency_map_cat[layer][:,i*saliency_map.shape[1]:(i+1)*saliency_map.shape[1],
                                        j*saliency_map.shape[2]:(j+1)*saliency_map.shape[2]] = saliency_map
                feature_map_cat[layer][:,i*feature_map.shape[1]:(i+1)*feature_map.shape[1],
                                        j*feature_map.shape[2]:(j+1)*feature_map.shape[2]] = feature_map
    
    # get cat for show
    resized_img_list = []
    for _, _, resized_img in preprocessed_img_tensor_and_id_list:
        resized_img_list.append(resized_img)
    
    locs_for_show=np.random.choice(range(len(resized_img_list)), size=num_of_img_for_show, replace=False)
    
    saliency_map_list_for_show = []
    feature_map_list_for_show = []
    resized_img_list_for_show = []
    for loc_for_show in locs_for_show:
        saliency_map_list_for_show.append(saliency_map_list[loc_for_show])
        feature_map_list_for_show.append(feature_map_list[loc_for_show])
        resized_img_list_for_show.append(resized_img_list[loc_for_show])
    del resized_img_list
    
    assert len(saliency_map_list_for_show)==cat_size_for_show[0]*cat_size_for_show[1]
    assert len(feature_map_list_for_show)==cat_size_for_show[0]*cat_size_for_show[1]
    assert len(resized_img_list_for_show)==cat_size_for_show[0]*cat_size_for_show[1]

    resized_img_for_show_cat = np.zeros((cat_size_for_show[0]*resized_img.shape[0],
                                         cat_size_for_show[1]*resized_img.shape[1],
                                         resized_img.shape[2]), dtype=np.int)
    saliency_map_for_show_cat = {}
    feature_map_for_show_cat = {}

    for layer in which_layer_to_hook:
            saliency_map_for_show_cat[layer] = np.zeros((saliency_map_list_for_show[0][layer].shape[0],
                                                         cat_size_for_show[0]*saliency_map_list_for_show[0][layer].shape[1],
                                                         cat_size_for_show[1]*saliency_map_list_for_show[0][layer].shape[2]))
            feature_map_for_show_cat[layer] = np.zeros((feature_map_list_for_show[0][layer].shape[0],
                                                        cat_size_for_show[0]*feature_map_list_for_show[0][layer].shape[1],
                                                        cat_size_for_show[1]*feature_map_list_for_show[0][layer].shape[2]))
            for i in range(cat_size_for_show[0]):
                for j in range(cat_size_for_show[1]):
                    idx = i*cat_size_for_show[1]+j
                    saliency_map = saliency_map_list_for_show[idx][layer]
                    feature_map = feature_map_list_for_show[idx][layer]
                    resized_img = resized_img_list_for_show[idx]
                    resized_img_for_show_cat[i*resized_img.shape[0]:(i+1)*resized_img.shape[0],
                                             j*resized_img.shape[1]:(j+1)*resized_img.shape[1],:] = resized_img
                    saliency_map_for_show_cat[layer][:,i*saliency_map.shape[1]:(i+1)*saliency_map.shape[1],
                                                     j*saliency_map.shape[2]:(j+1)*saliency_map.shape[2]] = saliency_map
                    feature_map_for_show_cat[layer][:,i*feature_map.shape[1]:(i+1)*feature_map.shape[1],
                                                    j*feature_map.shape[2]:(j+1)*feature_map.shape[2]] = feature_map
    
    return feature_map_cat, saliency_map_cat, resized_img_cat,\
feature_map_for_show_cat, saliency_map_for_show_cat, resized_img_for_show_cat

for chosen_ID in chosen_IDs:
    children = []

    print("For (ImageNet ID: "+str(chosen_ID)+") "+imagenet_class_parent_and_child_dict[chosen_ID]["words"])

    def find_children(current_ID):
        if len(imagenet_class_parent_and_child_dict[current_ID]["children"])==0:
            print("find child (ImageNet ID: "+str(current_ID)+") "+imagenet_class_parent_and_child_dict[current_ID]["words"])
            children.append(current_ID)
            return
        else:
            for c in imagenet_class_parent_and_child_dict[current_ID]["children"]:
                find_children(c)

    find_children(chosen_ID)
    
    MAX_N = args.max_img_samples
    
    num_of_children = np.min([MAX_N, len(children)])

    children = np.random.choice(children, size=num_of_children, replace=False)
    print("num of children after random choice: "+str(len(children)))

    Num_of_img_of_each_child = args.num_of_img_of_each_class

    preprocessed_img_tensor_and_id_list = []
    for idx, child in enumerate(children):
        print("load: "+imagenet_class_parent_and_child_dict[child]["words"]+" "+str(idx/len(children)), end="\r")
        correct_model_id = int(map_dict_imagenet2model[str(child)])
        child_img_path = img_path+"/"+imagenet_class_parent_and_child_dict[child]["WNID"]
        try:
            img_names = os.listdir(child_img_path)
            if len(img_names)>=Num_of_img_of_each_child:
                chosen_img_names = np.random.choice(img_names, size=Num_of_img_of_each_child, replace=False)
                for chosen_img_name in chosen_img_names:
                    try:
                        preprocessed_img_tensor, resized_img = load_and_preprocess_image(child_img_path+"/"+chosen_img_name, resize_im=(224,224))
                        preprocessed_img_tensor_and_id_list.append((preprocessed_img_tensor, correct_model_id, resized_img))
                    except:
                        print("error of "+str(child))
                        pass
        except:
            print("Not found ", child_img_path)
            pass
    
    if len(preprocessed_img_tensor_and_id_list)>MAX_N:
        preprocessed_img_tensor_and_id_list = random.choices(preprocessed_img_tensor_and_id_list, k=MAX_N)
    
    cat_size = [len(preprocessed_img_tensor_and_id_list),1]
    assert len(preprocessed_img_tensor_and_id_list)==cat_size[0]*cat_size[1]
    
    cat_size_for_show = [10,10]
    num_of_img_for_show = 100
    if len(preprocessed_img_tensor_and_id_list)<100:
        num_of_img_for_show=len(preprocessed_img_tensor_and_id_list)
        cat_size_for_show[0] = int(np.ceil(num_of_img_for_show/10))
    
    feature_map_cat, saliency_map_cat, resized_img_cat,\
    feature_map_for_show_cat, saliency_map_for_show_cat, resized_img_for_show_cat = \
    get_saliency_map_and_feature_map_cat(pretrained_model, which_layer_to_hook,
                                         preprocessed_img_tensor_and_id_list, device,
                                         cat_size = cat_size, cat_size_for_show = cat_size_for_show, salient_method = salient_method, num_of_img_for_show = num_of_img_for_show)
    
    save_path = where_to_save+"/result_of_ID_"+str(chosen_ID)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for layer in feature_map_cat.keys():
        if not os.path.exists(save_path+"/"+layer):
            os.mkdir(save_path+"/"+layer)
        np.savez_compressed(save_path+"/"+layer+"/resized_imgs.npz", resized_img_cat=resized_img_cat, resized_img_for_show_cat=resized_img_for_show_cat)
        np.savez_compressed(save_path+"/"+layer+"/feature_map_and_saliency_map.npz", feature_map_cat=feature_map_cat[layer], saliency_map_cat=saliency_map_cat[layer],feature_map_for_show_cat=feature_map_for_show_cat[layer],saliency_map_for_show_cat=saliency_map_for_show_cat[layer])

gc.collect()
