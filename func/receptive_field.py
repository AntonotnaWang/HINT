import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import gc

def check_same(stride):
    if isinstance(stride, (list, tuple)):
        assert len(stride) == 2 and stride[0] == stride[1]
        stride = stride[0]
    return stride

# the most important one
def receptive_field(model, input_size, batch_size=-1, device="cuda"):
    '''
    :parameter
    'input_size': tuple of (Channel, Height, Width)

    :return  OrderedDict of `Layername`->OrderedDict of receptive field stats {'j':,'r':,'start':,'conv_stage':,'output_shape':,}
    'j' for "jump" denotes how many pixels do the receptive fields of spatially neighboring units in the feature tensor
        do not overlap in one direction.
        i.e. shift one unit in this feature map == how many pixels shift in the input image in one direction.
    'r' for "receptive_field" is the spatial range of the receptive field in one direction.
    'start' denotes the center of the receptive field for the first unit (start) in on direction of the feature tensor.
        Convention is to use half a pixel as the center for a range. center for `slice(0,5)` is 2.5.
    '''
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            
            print('processing '+class_name)
            
            module_idx = len(receptive_field)
            m_key = "%i" % module_idx
            p_key = "%i" % (module_idx - 1)
            receptive_field[m_key] = OrderedDict()

            if not receptive_field["0"]["conv_stage"]:
                print("Enter in deconv_stage")
                receptive_field[m_key]["j"] = 0
                receptive_field[m_key]["r"] = 0
                receptive_field[m_key]["start"] = 0
            else:
                p_j = receptive_field[p_key]["j"]
                p_r = receptive_field[p_key]["r"]
                p_start = receptive_field[p_key]["start"]
                
                if class_name == "Conv2d" or class_name == "MaxPool2d":
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    kernel_size, stride, padding = map(check_same, [kernel_size, stride, padding])
                    receptive_field[m_key]["j"] = p_j * stride
                    receptive_field[m_key]["r"] = p_r + (kernel_size - 1) * p_j
                    receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j
                elif class_name == "BatchNorm2d" or class_name == "ReLU" or class_name == "Bottleneck":
                    receptive_field[m_key]["j"] = p_j
                    receptive_field[m_key]["r"] = p_r
                    receptive_field[m_key]["start"] = p_start
                elif class_name == "ConvTranspose2d":
                    receptive_field["0"]["conv_stage"] = False
                    receptive_field[m_key]["j"] = 0
                    receptive_field[m_key]["r"] = 0
                    receptive_field[m_key]["start"] = 0
                else:
                    raise ValueError("module is not ok")
                    print("module is not ok")
                    pass
            receptive_field[m_key]["input_shape"] = list(input[0].size()) # only one
            receptive_field[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                # list/tuple
                receptive_field[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                # tensor
                receptive_field[m_key]["output_shape"] = list(output.size())
                receptive_field[m_key]["output_shape"][0] = batch_size

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(2, *input_size)).type(dtype)

    # create properties
    receptive_field = OrderedDict()
    receptive_field["0"] = OrderedDict()
    receptive_field["0"]["j"] = 1.0
    receptive_field["0"]["r"] = 1.0
    receptive_field["0"]["start"] = 0.5
    receptive_field["0"]["conv_stage"] = True
    receptive_field["0"]["output_shape"] = list(x.size())
    receptive_field["0"]["output_shape"][0] = batch_size
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    with torch.no_grad():
        model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>10} {:>10} {:>10} {:>15} ".format("Layer (type)", "map size", "start", "jump", "receptive_field")
    print(line_new)
    print("==============================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in receptive_field:
        # input_shape, output_shape, trainable, nb_params
        assert "start" in receptive_field[layer], layer
        assert len(receptive_field[layer]["output_shape"]) == 4
        line_new = "{:7} {:12}  {:>10} {:>10} {:>10} {:>15} ".format(
            "",
            layer,
            str(receptive_field[layer]["output_shape"][2:]),
            str(receptive_field[layer]["start"]),
            str(receptive_field[layer]["j"]),
            format(str(receptive_field[layer]["r"]))
        )
        print(line_new)

    print("==============================================================================")
    # add input_shape
    receptive_field["input_size"] = input_size
    
    del x, h, hooks
    torch.cuda.empty_cache()
    gc.collect()
        
    return receptive_field

def receptive_field_for_unit(receptive_field_dict, layer, unit_position):
    """Utility function to calculate the receptive field for a specific unit in a layer
        using the dictionary calculated above
    :parameter
        'layer': layer name, should be a key in the result dictionary
        'unit_position': spatial coordinate of the unit (H, W)

    ```
    alexnet = models.alexnet()
    model = alexnet.features.to('cuda')
    receptive_field_dict = receptive_field(model, (3, 224, 224))
    receptive_field_for_unit(receptive_field_dict, "8", (6,6))
    ```
    Out: [(62.0, 161.0), (62.0, 161.0)]
    """
    input_shape = receptive_field_dict["input_size"]
    if layer in receptive_field_dict:
        rf_stats = receptive_field_dict[layer]
        assert len(unit_position) == 2
        feat_map_lim = rf_stats['output_shape'][2:]
        if np.any([unit_position[idx] < 0 or
                   unit_position[idx] >= feat_map_lim[idx]
                   for idx in range(2)]):
            raise Exception("Unit position outside spatial extent of the feature tensor ((H, W) = (%d, %d)) " % tuple(feat_map_lim))
        # X, Y = tuple(unit_position)
        rf_range = [(rf_stats['start'] + idx * rf_stats['j'] - rf_stats['r'] / 2,
            rf_stats['start'] + idx * rf_stats['j'] + rf_stats['r'] / 2) for idx in unit_position]
        if len(input_shape) == 2:
            limit = input_shape
        else:  # input shape is (channel, H, W)
            limit = input_shape[1:3]
        rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(2)]
        print("Receptive field size for layer %s, unit_position %s,  is \n %s" % (layer, unit_position, rf_range))
        return rf_range
    else:
        raise KeyError("Layer name incorrect, or not included in the model.")

def get_rfs_from_a_mask_with_order(receptive_field_dict, layer, mask_with_probvalues):
    # receptive_field_dict is generated by receptive_field()
    # layer: a str of number indicating which layer
    # mask: numpy.array, same shape as the output of the layer, 0 & or probvalue >0 which is probability of being foreground, we find the receptive fields of >0
    try:
        rfs=[]

        mask_with_probvalues = np.array(mask_with_probvalues)
        mask_with_probvalues_reshape = -mask_with_probvalues.reshape(-1)
        sort_mask_with_probvalues_reshape = np.sort(mask_with_probvalues_reshape)
        sort_mask_with_probvalues = np.ones(mask_with_probvalues_reshape.shape[0])*(-1)
        for i in range(mask_with_probvalues_reshape.shape[0]):
            try:
                sort_mask_with_probvalues[i]=np.where(sort_mask_with_probvalues_reshape == mask_with_probvalues_reshape[i])[0][0]
            except:
                pass
        sort_mask_with_probvalues=sort_mask_with_probvalues.reshape(mask_with_probvalues.shape)
        sort_mask_with_probvalues=np.array(sort_mask_with_probvalues, dtype=np.int)

        total_num_of_rfs = np.sum(np.array(mask_with_probvalues>0, dtype=np.int))

        for i in range(total_num_of_rfs):
            loc = np.where(sort_mask_with_probvalues==i)
            loc_change_format = (loc[0][0], loc[1][0])
            rf = receptive_field_for_unit(receptive_field_dict, str(layer), loc_change_format)
            rfs.append(rf)
    except:
        print("something must be wrong with get_rfs_from_a_mask_with_order")
    return rfs

def get_rfs_from_a_mask(receptive_field_dict, layer, mask):
    # receptive_field_dict is generated by receptive_field()
    # layer: a str of number indicating which layer
    # mask: numpy.array, same shape as the output of the layer, 0 & 1, we find the receptive fields of the ones
    rfs=[]
    
    locs = np.where(mask>0)
    
    for i in range(len(locs[0])):
        loc = (locs[0][i], locs[1][i])
        rf = receptive_field_for_unit(receptive_field_dict, str(layer), loc)
        rfs.append(rf)
    
    return rfs

def show_rf_in_org_img(org_img, rfs, mask):
    # rfs is generated by get_rfs_from_a_mask()
    # org_img: [H, W, 3]
    # mask: numpy.array, same shape as the output of the layer, 0 & 1, we find the receptive fields of the ones
    
    mask_shape = mask.shape
    
    plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(mask_shape[0], mask_shape[1])
    
    locs = np.where(mask>0)
    
    assert len(locs[0])==len(rfs)
    
    for i in range(len(rfs)):
        ax = plt.subplot(gs[locs[0][i], locs[1][i]])
        rf = rfs[i]
        crop_org_img = org_img[int(rf[0][0]):int(rf[0][1]+1),
                               int(rf[1][0]):int(rf[1][1]+1), :]
        
        ax.imshow(crop_org_img)
        ax.axis('off')

def show_some_rfs_randomly(org_img, rfs, show_size=[5,5], fig_save_name="rfs.png", is_show=True):
    # rfs is generated by get_rfs_from_a_mask()
    # org_img: [H, W, 3]
    
    if not is_show:
        matplotlib.use('agg')
    
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(show_size[0], show_size[1])
    
    choice_rfs = []
    if len(rfs)<=show_size[0]*show_size[1]:
        choice_range = len(rfs)
        show_size[0] = np.int(np.floor(np.sqrt(choice_range)))
        show_size[1] = np.int(np.floor(np.sqrt(choice_range)))
    else:
        choice_range = show_size[0]*show_size[1]
        
    for i in np.random.choice(len(rfs), choice_range):
        try:
            choice_rfs.append(rfs[i])
        except:
            print("problem happens at i: "+str(i))
    
    for i in range(show_size[0]):
        for j in range(show_size[1]):
            ax = fig.add_subplot(gs[i,j])
            rf = choice_rfs[i*show_size[0]+j]
            crop_org_img = org_img[int(rf[0][0]):int(rf[0][1]),
                                   int(rf[1][0]):int(rf[1][1]), :]
            '''
            if i%30==0:
                plt.figure()
                plt.imshow(crop_org_img)
            '''
            ax.imshow(crop_org_img)
            ax.axis('off')
            
    plt.savefig(fig_save_name, bbox_inches='tight', dpi=fig.dpi,pad_inches=0.0)
    
    if not is_show:
        fig.clf()
        plt.close()
        del fig, gs
        gc.collect()

def show_some_rfs_order(org_img, rfs, show_size=[5,5], fig_save_name="rfs.png", is_show=True):
    # rfs is generated by get_rfs_from_a_mask()
    # org_img: [H, W, 3]
    # show show_size.shape[0]*show_size.shape[1] receptive fields with rising order of order_map pixel values
    
    if not is_show:
        matplotlib.use('agg')
    
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(show_size[0], show_size[1])
    
    choice_rfs = []
    if len(rfs)<=show_size[0]*show_size[1]:
        choice_range = len(rfs)
        show_size[0] = np.int(np.floor(np.sqrt(choice_range)))
        show_size[1] = np.int(np.floor(np.sqrt(choice_range)))
    else:
        choice_range = show_size[0]*show_size[1]
        
    for i in range(choice_range):
        try:
            choice_rfs.append(rfs[i])
        except:
            print("problem happens at i: "+str(i))
    
    for i in range(show_size[0]):
        for j in range(show_size[1]):
            ax = fig.add_subplot(gs[i,j])
            rf = choice_rfs[i*show_size[0]+j]
            crop_org_img = org_img[int(rf[0][0]):int(rf[0][1]),
                                   int(rf[1][0]):int(rf[1][1]), :]
            '''
            if i%30==0:
                plt.figure()
                plt.imshow(crop_org_img)
            '''
            ax.imshow(crop_org_img)
            ax.axis('off')
            
    plt.savefig(fig_save_name, bbox_inches='tight', dpi=fig.dpi,pad_inches=0.0)
    
    if not is_show:
        fig.clf()
        plt.close()
        del fig, gs
        gc.collect()