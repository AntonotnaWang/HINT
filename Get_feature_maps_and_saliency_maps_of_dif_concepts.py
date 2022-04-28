import numpy as np
import os
import time

T1 = time.time()

image_path = "/data/ImageNet_ILSVRC2012/ILSVRC2012_train"

result_save_path = "output/feature_maps_and_saliency_maps" # you may change the save path

chosen_IDs = [1082,1095,1845]
# chosen_IDs = [3, 8, 10, 1549, 15, 30, 34, 1072, 1073, 1076, 1078, 55, 1081, 1082, 1083, 1084, 1089, 1090, 1091, 1095, \
#     76, 80, 1109, 1110, 85, 1115, 1117, 1119, 1120, 95, 1121, 1148, 1152, 1153, 1156, 1157, 1158, 1159, 1160, 153, 1182, \
#         1187, 1190, 1192, 174, 1203, 1204, 1205, 1206, 1209, 1210, 1211, 190, 1215, 1218, 1220, 1221, 1222, 1223, 1224, 1225, \
#             1226, 1227, 1228, 1229, 206, 1230, 1232, 1233, 1231, 214, 1243, 1263, 1264, 1268, 1270, 1274, 1275, 1276, 1284, \
#                 1288, 1289, 266, 1292, 1294, 1295, 1296, 1297, 1322, 1323, 1327, 1845, 1358, 1359, 1361, 1364, 342, 343, 357, \
#                     359, 1388, 1391, 383, 384, 389]

device = "cuda"

saliency_method = "GuidedBackprop"
assert saliency_method in ["GuidedBackprop", "VanillaBackprop", "SmoothGrad", "IntegratedGradients", "GradientxInput"]

### for diff layers of diff models

model = "vgg19"
layers = ["features.30"]
# layers = ["features.7", "features.10", "features.14", "features.20", "features.25", "features.30"]
for chosen_ID in chosen_IDs:
    for layer in layers:
        run_py_script = "python run/get_feature_maps_and_saliency_maps_for_responsible_region_identification.py"+\
            " --image_path "+str(image_path)+\
            " --chosen_IDs "+str(chosen_ID)+\
                " --layers "+str(layer)+\
                    " --model_name "+str(model)+\
                        " --device "+str(device)+\
                            " --where_to_save "+str(result_save_path)+"_"+str(model)+\
                                " --saliency_method "+str(saliency_method)
        print("run "+run_py_script)
        os.system(run_py_script)



model = "resnet50"
layers = ["layer3.5"]
for chosen_ID in chosen_IDs:
    for layer in layers:
        run_py_script = "python run/get_feature_maps_and_saliency_maps_for_responsible_region_identification.py"+\
            " --image_path "+str(image_path)+\
            " --chosen_IDs "+str(chosen_ID)+\
                " --layers "+str(layer)+\
                    " --model_name "+str(model)+\
                        " --device "+str(device)+\
                            " --where_to_save "+str(result_save_path)+"_"+str(model)+\
                                " --saliency_method "+str(saliency_method)
        print("run "+run_py_script)
        os.system(run_py_script)



model = "inception_v3"
layers = ["Mixed_6b"]
for chosen_ID in chosen_IDs:
    for layer in layers:
        run_py_script = "python run/get_feature_maps_and_saliency_maps_for_responsible_region_identification.py"+\
            " --image_path "+str(image_path)+\
            " --chosen_IDs "+str(chosen_ID)+\
                " --layers "+str(layer)+\
                    " --model_name "+str(model)+\
                        " --device "+str(device)+\
                            " --where_to_save "+str(result_save_path)+"_"+str(model)+\
                                " --saliency_method "+str(saliency_method)
        print("run "+run_py_script)
        os.system(run_py_script)

T2 = time.time()
print('time consumption: %s s' % ((T2 - T1)))
