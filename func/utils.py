import os
import numpy as np
from PIL import Image
import scipy.io
import pickle
import json
import logging
import sys

import torch
from torch.autograd import Variable
from torchvision import models

#def get_file_path():
#    return os.path.split(os.path.realpath(__file__))[0]


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# some funcs to process imagenet id

# modeloutput_id_wnid_class
def get_model_output_id_wnid_class_dict(jsonfilepath='imagenet_info/imagenet_label_index.json'):
    '''
    get the dict of model output ID, WNID and class name
    from the given json file
    format: {"Model Ouput ID": ["WNID", "Class"]}
    '''
    with open(jsonfilepath, 'r') as f:
        id_dict = json.load(f)
    
    return id_dict

# imagenet_id_wnid_class
def get_imagenet_id_wnid_class_dict(matfilepath='imagenet_info/ILSVRC2012_meta.mat'):
    '''
    get the dict of ImageNet ID, WNID and class name
    from the given mat file
    format: {"ImageNet ID": ["WNID", "class"]}, e.g. {..."233": ['n02106382', 'Bouvier_des_Flandres'], ...}
    '''
    meta=scipy.io.loadmat(matfilepath)['synsets']
    length=len(meta)
    id_dict={str(meta[0][0][0][0][0]):[meta[0][0][1][0], meta[0][0][2][0]]}
    for i in range(1, length):
        id_dict[str(meta[i][0][0][0][0])]=[meta[i][0][1][0], meta[i][0][2][0]]
        
    return id_dict

def map_model_id_to_imagenet_id(imagenet_id, modeloutput_id):
    '''
    return a dict mapping modeloutput id to imagenet id
    '''
    map_dict={}
    for imagenet_id_key in imagenet_id:
        for modeloutput_id_key in modeloutput_id:
            if modeloutput_id[modeloutput_id_key][0]==imagenet_id[imagenet_id_key][0]:
                map_dict[modeloutput_id_key]=imagenet_id_key
                break
    return map_dict

def map_imagenet_id_to_model_id(imagenet_id, modeloutput_id):
    '''
    return a dict mapping imagenet id to modeloutput id
    '''
    map_dict={}
    for imagenet_id_key in imagenet_id:
        for modeloutput_id_key in modeloutput_id:
            if modeloutput_id[modeloutput_id_key][0]==imagenet_id[imagenet_id_key][0]:
                map_dict[imagenet_id_key]=modeloutput_id_key
                break
    return map_dict



# to load a image from imagenet_sample with filepath_5000
# ----------------------------------------
filepath_5000 = "/data/imagenet_sample_5000" # you may change it

def get_img_names_and_labels_from_imagenet_sample(data_filepath=filepath_5000 + "/imagenet_sample_5000"):
    img_files=os.listdir(data_filepath)
    img_files.sort()
    
    labels = []
    for idx, file in enumerate(img_files):
        labels.append(int(file.split("_")[0]))
    
    print("There are "+str(len(img_files))+" imgs, and "+str(len(np.unique(np.array(labels))))+" classes.")
    
    return img_files, labels

def load_img_from_imagenet_sample_by_index(index, imagenet_labels=None,
                                           data_filepath=filepath_5000 + "/imagenet_sample_5000"):
    '''
    INPUT:
    index: (if data_filepath=filepath_5000 + "/imagenet_sample_5000") 0~4999, which image file to load
    imagenet_labels: a dict, len=1000, {"ImageNet ID": ["WNID", "class"]}, e.g. {..."233": ['n02106382', 'Bouvier_des_Flandres'], ...}
    data_filepath:
    (name format of imgs: [ImageNet_ID]_[WNID]_[N].JPEG, N: 0~4, e.g. "141_n02104029_3.JPEG" means ImageNet ID: 141, WNID: n02104029,
    3nd pic of this class, and its class is 'kuvasz'.)
    
    OUTPUT:
    the img file
    '''
    img_files, labels = get_img_names_and_labels_from_imagenet_sample(data_filepath)
    
    load_img = Image.open(data_filepath+"/"+img_files[index]).convert('RGB')
    
    if imagenet_labels is not None:
        print("load img "+data_filepath+"/"+img_files[index]+\
              "\nImageNet ID: "+str(labels[index])+\
              "\nWNID and class: "+str(imagenet_labels[str(labels[index])]))
    else:
        print("load img "+data_filepath+"/"+img_files[index]+\
              "\nImageNet ID: "+str(labels[index]))
    
    return load_img, index, labels[index]

def load_img_from_imagenet_sample_by_class(imagenet_id, imagenet_labels=None,
                                           data_filepath=filepath_5000 + "/imagenet_sample_5000"):
    '''
    INPUT:
    imagenet_id: 1~1000, indicating which class to load, we'll randomly choice one img in the class
    imagenet_labels: a dict, len=1000, {"ImageNet ID": ["WNID", "class"]}, e.g. {..."233": ['n02106382', 'Bouvier_des_Flandres'], ...}
    data_filepath:
    (name format of imgs: [ImageNet_ID]_[WNID]_[N].JPEG, N: 0~4, e.g. "141_n02104029_3.JPEG" means ImageNet ID: 141, WNID: n02104029,
    3nd pic of this class, and its class is 'kuvasz'.)
    
    OUTPUT:
    the img file
    '''
    img_files, labels = get_img_names_and_labels_from_imagenet_sample(data_filepath)
    
    idxs=np.where(np.array(labels)==imagenet_id)[0]
    chosen_idx=int(np.random.choice(idxs, 1))
    
    load_img = Image.open(data_filepath+"/"+img_files[chosen_idx]).convert('RGB')
    
    if imagenet_labels is not None:
        print("load img "+data_filepath+"/"+img_files[chosen_idx]+\
              "\nImageNet ID: "+str(labels[chosen_idx])+\
              "\nWNID and class: "+str(imagenet_labels[str(labels[chosen_idx])]))
    else:
        print("load img "+data_filepath+"/"+img_files[chosen_idx]+\
              "\nImageNet ID: "+str(labels[chosen_idx]))
    
    return load_img, chosen_idx, imagenet_id
# ----------------------------------------



# some functions for image processing, pre-trained model loading, and a unified way to conduct saliency map methods

# load pre-trained model
def get_pretrained_model(model_name, is_pretrained=True):
    if model_name == "alexnet":
        return models.alexnet(pretrained=is_pretrained)
    
    elif model_name == "vgg11":
        return models.vgg11(pretrained=is_pretrained)
    elif model_name == "vgg11_bn":
        return models.vgg11_bn(pretrained=is_pretrained)
    elif model_name == "vgg13":
        return models.vgg13(pretrained=is_pretrained)
    elif model_name == "vgg13_bn":
        return models.vgg13_bn(pretrained=is_pretrained)
    elif model_name == "vgg16":
        return models.vgg16(pretrained=is_pretrained)
    elif model_name == "vgg16_bn":
        return models.vgg16_bn(pretrained=is_pretrained)
    elif model_name == "vgg19":
        return models.vgg19(pretrained=is_pretrained)
    elif model_name == "vgg19_bn":
        return models.vgg19_bn(pretrained=is_pretrained)
    
    elif model_name == "resnet18":
        return models.resnet18(pretrained=is_pretrained)
    elif model_name == "resnet34":
        return models.resnet34(pretrained=is_pretrained)
    elif model_name == "resnet50":
        return models.resnet50(pretrained=is_pretrained)
    elif model_name == "resnet101":
        return models.resnet101(pretrained=is_pretrained)
    elif model_name == "resnet152":
        return models.resnet152(pretrained=is_pretrained)
    
    elif model_name == "inception_v3":
        return models.inception_v3(pretrained=is_pretrained)
    

# process an ImageNet raw image to input it into a CNN
def preprocess_image(pil_im, resize_im=(224,224)):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")
            
    # Resize image
    assert isinstance(resize_im,(int,tuple))
    if isinstance(resize_im, int):
        resize_im = (resize_im, resize_im)
    else:
        assert len(resize_im)==2
    pil_im = pil_im.resize(resize_im, Image.ANTIALIAS)
        
    im_as_arr = np.float32(pil_im)
    
    im_resize = np.array(im_as_arr, dtype = np.int)
    
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    
    return im_as_var, im_resize

# load an ImageNet raw image and process it
def load_and_preprocess_image(image_path, resize_im=(224,224)):
    pil_im = Image.open(image_path).convert('RGB')
    im_as_var, im_resize = preprocess_image(pil_im, resize_im=resize_im)
    return im_as_var, im_resize

def post_process_saliency_map(im_as_arr, mode="norm"):
    """
        Converts 3d image to saliency_map

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        saliency_map (numpy_arr): Grayscale image with shape (1,W,D)
    """
    assert mode in ["norm", "sum", "filter_norm", "max", "abs_sum", "abs_max"]
    if mode=="norm":
        im_as_arr_pro = np.linalg.norm(im_as_arr, axis=0)
    elif mode=="sum":
        im_as_arr_pro = np.sum(im_as_arr, axis=0)
    elif mode=="filter_norm":
        im_as_arr_pro = np.linalg.norm(np.clip(im_as_arr, a_min=0, a_max=None), axis=0)
    elif mode=="max":
        im_as_arr_pro = np.max(im_as_arr, axis=0)
    elif mode=="abs_sum":
        im_as_arr_pro = np.sum(np.abs(im_as_arr), axis=0)
    elif mode=="abs_max":
        im_as_arr_pro = np.max(np.abs(im_as_arr), axis=0)
    
    return normalization_of_saliency_map(im_as_arr_pro)

def normalization_of_saliency_map(im_as_arr):
    im_max = np.percentile(im_as_arr, 99)
    im_min = np.min(im_as_arr)
    if im_max>im_min:
        im_as_arr = (np.clip((im_as_arr - im_min) / (im_max - im_min), 0, 1))
    im_as_arr = np.expand_dims(im_as_arr, axis=0)
    return im_as_arr

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../grabcheck_results'):
        os.makedirs('../grabcheck_results')
    
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    
    # Save image
    path_to_file = os.path.join('../grabcheck_results', file_name + '.jpg')
    save_image(gradient, path_to_file)

def get_positive_negative_with_scale_control(input_img, is_one_channel_output = True):
    
    pos = np.maximum(0, input_img)
    neg = np.maximum(0, -input_img)
    
    pos = np.sum(pos, axis=0)
    neg = np.sum(neg, axis=0)
    
    pos_max = np.percentile(pos, 99)
    neg_max = np.percentile(neg, 99)
    max_of_pos_and_neg = np.max([pos_max, neg_max])
        
    pos = (np.clip(pos/max_of_pos_and_neg, 0, 1))
    neg = (np.clip(neg/max_of_pos_and_neg, 0, 1))
    
    if is_one_channel_output==True:
        pos = np.expand_dims(pos, axis=0)
        neg = np.expand_dims(neg, axis=0)
    
    return pos, neg

def setup_logger(name, save_dir, filename="readme.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
