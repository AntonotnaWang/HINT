import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import resize


def load_feature_saliency_map_and_resized_img_for_show(path, layer=None):
    print("load path: "+path)
    print("focus on layer: "+str(layer))
    
    if os.path.isfile(path+"/"+layer+"/resized_imgs.npz"):
        resized_imgs = np.load(path+"/"+layer+"/resized_imgs.npz")
    else:
        resized_imgs = np.load(path+"/resized_imgs.npz")
    
    try:
        feature_map_and_saliency_map = np.load(path+"/"+layer+"/feature_map_and_saliency_map.npz")
    except:
        feature_map_and_saliency_map = np.load(path+"/feature_map_and_saliency_map.npz")
    
    resized_img_for_show_cat = resized_imgs["resized_img_for_show_cat"]
    feature_map_for_show_cat = feature_map_and_saliency_map["feature_map_for_show_cat"]
    saliency_map_for_show_cat = feature_map_and_saliency_map["saliency_map_for_show_cat"]
    
    return resized_img_for_show_cat, feature_map_for_show_cat, saliency_map_for_show_cat

def show_concept_region_on_img(img_for_show, concept_map, \
    figsize=(10,10), alpha=0.5, is_save=False, save_name="pic.png", dpi=500):
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(img_for_show)
    concept_map_show = resize(np.array(concept_map, dtype=float), \
        (img_for_show.shape[0], img_for_show.shape[1]))
    if np.sum(concept_map_show) == 0:
        concept_map_show[0,0] = 1
    ax.matshow(-concept_map_show, cmap='Spectral', alpha=alpha)
    
    if is_save:
        plt.savefig(save_name, bbox_inches='tight', dpi=dpi, pad_inches=0.0)