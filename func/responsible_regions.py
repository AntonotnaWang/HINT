# some util funcs
from .utils import post_process_saliency_map, get_positive_negative_with_scale_control

import numpy as np
import gc
import os

def process_saliency_map(saliency_map, mode="norm"):
    print("process_saliency_map")
    grayscale_saliency_map = post_process_saliency_map(saliency_map, mode=mode)[0]
    pos_saliency_map, neg_saliency_map = get_positive_negative_with_scale_control(saliency_map)
    return grayscale_saliency_map, pos_saliency_map[0], neg_saliency_map[0]
# grayscale_saliency_map, pos_saliency_map, neg_saliency_map
# [pic_length, pic_width]

def process_cat_saliency_map(saliency_map, pic_size = None, num_of_pic_of_a_row = 10, mode = "norm"):
    print("process_saliency_map")
    grayscale_cat_saliency_map = np.zeros((saliency_map.shape[1], saliency_map.shape[2]))
    if pic_size is not None:
        size_of_one_saliency_map = [pic_size,pic_size]
    else:
        pic_size = int(saliency_map.shape[2]/num_of_pic_of_a_row)
        size_of_one_saliency_map = [pic_size,pic_size]
    cat_size = [int(saliency_map.shape[1]/pic_size),int(saliency_map.shape[2]/pic_size)]
    for i in range(cat_size[0]):
        for j in range(cat_size[1]):
            saliency_map_current = saliency_map[:,i*size_of_one_saliency_map[0]:np.clip((i+1)*size_of_one_saliency_map[0],0,saliency_map.shape[1]),
                                                j*size_of_one_saliency_map[1]:np.clip((j+1)*size_of_one_saliency_map[1],0,saliency_map.shape[2])]
            grayscale_saliency_map_current = post_process_saliency_map(saliency_map_current, mode=mode)[0]
            grayscale_cat_saliency_map[i*size_of_one_saliency_map[0]:np.clip((i+1)*size_of_one_saliency_map[0],0,saliency_map.shape[1]),
                                       j*size_of_one_saliency_map[1]:np.clip((j+1)*size_of_one_saliency_map[1],0,saliency_map.shape[2])] = grayscale_saliency_map_current
    return grayscale_cat_saliency_map

def X_y_preparation(feature_map_one, saliency_map_one, mode = "norm", t = 0.5, pic_size = None, num_of_pic_of_a_row = 10):
    grayscale_saliency_map = process_cat_saliency_map(saliency_map_one, pic_size=pic_size,
                                                      num_of_pic_of_a_row=num_of_pic_of_a_row, mode=mode)

    X_pos = feature_map_one[:, grayscale_saliency_map>t]
    X_neg = feature_map_one[:, grayscale_saliency_map<=t]

    X_pos = X_pos.transpose(1,0)
    X_neg = X_neg.transpose(1,0)

    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    X = np.concatenate((X_pos, X_neg))
    
    return X, y, X_pos, X_neg

def load_responsible_regions_from_given_path(path, layer=None, pic_size=None, num_of_pic_of_a_row=10, mode="norm", t=0.5):
    print("load path: "+path)
    print("focus on layer: "+str(layer))
    try:
        feature_map_and_saliency_map = np.load(path+"/"+layer+"/feature_map_and_saliency_map.npz")
    except:
        feature_map_and_saliency_map = np.load(path+"/feature_map_and_saliency_map.npz")
    
    feature_map_one = feature_map_and_saliency_map["feature_map_cat"]
    saliency_map_one = feature_map_and_saliency_map["saliency_map_cat"]
    
    _, _, X_pos, X_neg = X_y_preparation(feature_map_one, saliency_map_one, mode=mode, t=t,
                                         pic_size=pic_size,num_of_pic_of_a_row = num_of_pic_of_a_row)
    
    gc.collect()
    
    return X_pos, X_neg