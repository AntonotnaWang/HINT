import numpy as np
from torch.nn import ReLU
from torch.autograd import Variable
import torch

# some util funcs
from .utils import post_process_saliency_map

# a unified way to conduct saliency map methods
def conduct_saliency_map_method(METHOD, processed_img_tensor, target_class_index, pretrained_model, which_layer_to_hook, device=torch.device('cpu')):
    saliency_map_method = METHOD(pretrained_model, which_layer_to_hook, device=device)
    saliency_map, feature_map = saliency_map_method.generate_explanation(processed_img_tensor, target_class_index)
    return saliency_map, feature_map, saliency_map_method
# format of saliency_map, feature_map
# [dimension, pic_length, pic_width]

class FeatureMapExtractor(object):
    def __init__(self, model, target_layers, device=torch.device('cpu')):
        self.device = device
        self.model = model.to(self.device)
        self.target_layers = target_layers
        self.gradients = []
        self.fmap_pool = dict()
        self.grad_pool = dict()
        self.handlers = []
        def forward_hook(key):
            def forward_hook_(module, input_im, output_im):
                #self.fmap_pool[key] = copy.deepcopy(output_im.detach().cpu().numpy())
                self.fmap_pool[key] = output_im.detach().clone().cpu().numpy()
            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                #self.grad_pool[key] = copy.deepcopy(grad_out[0].detach().cpu().numpy())
                self.grad_pool[key] = grad_out[0].detach().clone().cpu().numpy()
            return backward_hook_

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))
    
    def reset(self):
        self.gradients = []
        self.fmap_pool = dict()
        self.grad_pool = dict()
        self.handlers = []
        def forward_hook(key):
            def forward_hook_(module, input_im, output_im):
                #self.fmap_pool[key] = copy.deepcopy(output_im.detach().cpu().numpy())
                self.fmap_pool[key] = output_im.detach().clone().cpu().numpy()
            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                #self.grad_pool[key] = copy.deepcopy(grad_out[0].detach().cpu().numpy())
                self.grad_pool[key] = grad_out[0].detach().clone().cpu().numpy()
            return backward_hook_

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))
        
    def __call__(self, x):
        return self.model(x.to(self.device))

# Guided-Backpropagation
class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, target_layer_names, device=torch.device('cpu')):
        print("Guided Backpropagation")
        self.model = model
        self.device = device
        self.model.eval()
        self.forward_relu_outputs = []
        self.update_relus()
        self.target_layer_names = target_layer_names
        
        if isinstance(target_layer_names, list):
            self.target_layer_names = target_layer_names
        else:
            self.target_layer_names = [target_layer_names]
        
        self.extractor = FeatureMapExtractor(self.model, self.target_layer_names, self.device)
        
    def toString(self):
        return "Guided Backpropagation"
    
    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_explanation(self, input_image, target_class=None):
        # Forward pass
        model_output = self.extractor(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.cpu().data.numpy())
        
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(self.device)
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        
        feature_map_as_arr = {}
        gradients_as_arr = {}
        
        for num, target_layer_name in enumerate(self.target_layer_names):
            feature_map_as_arr[target_layer_name]=self.extractor.fmap_pool[target_layer_name][0]
            gradients_as_arr[target_layer_name]=self.extractor.grad_pool[target_layer_name][0]
        
        self.extractor.reset()
        
        return gradients_as_arr, feature_map_as_arr

# Vanilla Backpropagation
class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, target_layer_names, device=torch.device('cpu')):
        print("Vanilla Backpropagation")
        self.model = model
        self.device = device
        self.model.eval()
        self.target_layer_names = target_layer_names
        
        if isinstance(target_layer_names, list):
            self.target_layer_names = target_layer_names
        else:
            self.target_layer_names = [target_layer_names]
        
        self.extractor = FeatureMapExtractor(self.model, self.target_layer_names, self.device)
    
    def toString(self):
        return "Vanilla Backpropagation"
    
    def generate_explanation(self, input_image, target_class=None):
        # Forward pass
        model_output = self.extractor(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.cpu().data.numpy())
        
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(self.device)
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        
        feature_map_as_arr = {}
        gradients_as_arr = {}
        
        for num, target_layer_name in enumerate(self.target_layer_names):
            feature_map_as_arr[target_layer_name]=self.extractor.fmap_pool[target_layer_name][0]
            gradients_as_arr[target_layer_name]=self.extractor.grad_pool[target_layer_name][0]
        
        self.extractor.reset()
        
        return gradients_as_arr, feature_map_as_arr

# SmoothGrad
class SmoothGrad():
    
    def __init__(self, pretrained_model, target_layer_names, is_guidedbackprop=False, device=torch.device('cpu')):
        print("SmoothGrad")
        if is_guidedbackprop==False:
            self.backprop = VanillaBackprop(pretrained_model, target_layer_names, device=device)
        else:
            self.backprop = GuidedBackprop(pretrained_model, target_layer_names, device=device)
    
    def toString(self):
        return "SmoothGrad"
    
    def generate_explanation(self, input_image, target_class, param_n = 50, param_sigma_multiplier = 4):
        smooth_grad, smooth_feature_map = self.get_smooth_grad(self.backprop, input_image, target_class, param_n, param_sigma_multiplier)
        return smooth_grad, smooth_feature_map
    
    def get_smooth_grad(self, Backprop, prep_img, target_class, param_n, param_sigma_multiplier):
        """
            Generates smooth gradients of given Backprop type. You can use this with both vanilla
            and guided backprop
        Args:
            Backprop (class): Backprop type
            prep_img (torch Variable): preprocessed image
            target_class (int): target class of imagenet
            param_n (int): Amount of images used to smooth gradient
            param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
        """
        # Calculate gradients
        grads, feature_map = Backprop.generate_explanation(prep_img, target_class)
        
        assert list(grads.keys())==list(feature_map.keys())
        
        smooth_grad = {}
        smooth_feature_map = {}
        
        # Generate an empty image/matrix
        for layer in grads.keys():
            smooth_grad[layer]=0
            smooth_feature_map[layer]=0
            #smooth_grad[layer]=np.zeros(grads[layer].shape)
            #smooth_feature_map[layer]=np.zeros(smooth_feature_map[layer].shape)

        mean = 0
        sigma = param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).item()
        for x in range(param_n):
            print("progress: "+str(x/param_n), end='\r')
            # Generate noise
            noise = Variable(prep_img.data.new(prep_img.size()).normal_(mean, sigma**2))
            # Add noise to the image
            noisy_img = prep_img + noise
            # Calculate gradients
            grads, feature_map = Backprop.generate_explanation(noisy_img, target_class)
            for layer in grads.keys():
                # Add gradients to smooth_grad
                smooth_grad[layer] = smooth_grad[layer] + grads[layer]
                smooth_feature_map[layer] = smooth_feature_map[layer] + feature_map[layer]
                
        # Average it out
        for layer in grads.keys():
            # Add gradients to smooth_grad
            smooth_grad[layer] = smooth_grad[layer] / param_n
            smooth_feature_map[layer] = smooth_feature_map[layer] / param_n
        return smooth_grad, smooth_feature_map

# GradCAM
class GradCAM():
    """
        Produces class activation map
    """
    
    def __init__(self, pretrained_model, target_layer_names, device=torch.device('cpu')):
        print("GradCAM")
        self.backprop = VanillaBackprop(pretrained_model, target_layer_names, device=device)
    
    def toString(self):
        return "GradCAM"
    
    def generate_explanation(self, input_image, target_class=None):
        
        grads, feature_map = self.backprop.generate_explanation(input_image, target_class)
        
        cams = {}
        
        for layer in grads.keys():
        
            # Get weights from gradients
            weights = np.mean(grads[layer], axis=(1, 2))  # Take averages for each gradient
            # Create empty numpy array for cam
            cam = np.ones(feature_map[layer].shape[1:], dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(weights):
                cam += w * feature_map[layer][i, :, :]
            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.expand_dims(cam, axis=0)
            
            cams[layer] = cam
        
        return cams, feature_map

# Guided Grad-CAM
class GuidedGradCAM():
    def __init__(self, pretrained_model, target_layer_names, device=torch.device('cpu')):
        print("Guided GradCAM")
        self.gradcam=GradCAM(pretrained_model, target_layer_names, device=device)
        self.gbp=GuidedBackprop(pretrained_model, target_layer_names, device=device)
    
    def toString(self):
        return "Guided GradCAM"
    
    def generate_explanation(self, input_image, target_class=None, mode="norm"):
        cams, feature_map = self.gradcam.generate_explanation(input_image, target_class)
        guided_grads, _ = self.gbp.generate_explanation(input_image, target_class)
        
        assert list(cams.keys()) == list(feature_map.keys())
        assert list(guided_grads.keys()) == list(feature_map.keys())
        
        cam_gb = {}
        
        for layer in cams.keys():
            grayscale_guided_grads = post_process_saliency_map(guided_grads[layer], mode=mode)[0]
            cam_gb[layer] = np.multiply(cams[layer], grayscale_guided_grads)
        return cam_gb, feature_map

# IntegratedGradients
class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, pretrained_model, target_layer_names, device=torch.device('cpu')):
        print("IntegratedGradients")
        self.backprop = VanillaBackprop(pretrained_model, target_layer_names, device=device)
        self.device = device
    
    def toString(self):
        return "IntegratedGradients"
    
    def generate_explanation(self, input_image, target_class=None, baseline_img=None, steps = 25):
        if baseline_img is None:
            baseline_img=torch.zeros(input_image.shape)
        
        diff_img = input_image - baseline_img
        
        baseline_img.to(self.device)
        diff_img.to(self.device)
        
        grad_init, feature_map = self.backprop.generate_explanation(input_image, target_class)
        
        assert list(grad_init.keys())==list(feature_map.keys())
        
        integrated_grads = {}
        
        for layer in grad_init.keys():
            integrated_grads[layer] = np.zeros(grad_init[layer].shape)
        
        for alpha in np.linspace(0, 1, steps):
            print("Progress: "+str(int(alpha*100))+"%", end="\r")
            img_step = baseline_img + alpha * (input_image - baseline_img)
            grads, _ = self.backprop.generate_explanation(img_step, target_class)
            assert list(grad_init.keys())==list(grads.keys())
            for layer in grad_init.keys():
                integrated_grads[layer] = integrated_grads[layer] + grads[layer]
        
        _, diff_feature_map = self.backprop.generate_explanation(diff_img, target_class)
        
        assert list(grad_init.keys())==list(diff_feature_map.keys())
        
        for layer in grad_init.keys():
            integrated_grads[layer] = integrated_grads[layer] * diff_feature_map[layer] / steps
        
        return integrated_grads, feature_map

# Gradient * Input
class GradientxInput(VanillaBackprop):
    def __init__(self, pretrained_model, target_layer_names, device=torch.device('cpu')):
        print("Gradient * Input")
        super(GradientxInput, self).__init__(pretrained_model, target_layer_names, device=device)
    
    def toString(self):
        return "Gradient * Input"
    
    def generate_explanation(self, input_img, target_class_index):
        vanilla_grads, feature_map = super(GradientxInput, self).generate_explanation(input_img, target_class_index)
        
        assert list(vanilla_grads.keys())==list(feature_map.keys())
        
        gradientxinput = {}
        
        for layer in vanilla_grads.keys():
            gradientxinput[layer] = np.multiply(vanilla_grads[layer], feature_map[layer])
            
        return gradientxinput, feature_map