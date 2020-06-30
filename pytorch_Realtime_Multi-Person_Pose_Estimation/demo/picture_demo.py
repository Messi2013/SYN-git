#3.25
import ipdb
import os
import re #2.28 assign multi split-symbol in the char string
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)



model = get_model('vgg19')
# ipdb.set_trace()     
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()





# 3.4 version
# modify the directory to overall dataset, to get the heatmap .npy result.
foulder = '/scratch/meil3/SYN_DB/MPV/train/pose2/'
g = os.walk(r"/scratch/meil3/SYN_DB/MPV/train/img/") 
for path,dir_list,file_list in g:  
    for file_name in file_list: 
        fullname = os.path.join(path, file_name) #full filename includes file foulder
        name_split_string = re.split('[/.]', fullname)
        cam_id = name_split_string[-4]
        group_id = name_split_string[-3]
        img_id = name_split_string[-2]
        oriImg = cv2.imread(fullname) # B,G,R order
        shape_dst = np.min(oriImg.shape[0:2])

        # Get results of original image

        with torch.no_grad():
            # print('oriImg',oriImg.shape)
            paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
            #helge:
            str1 = '/'
            new_name_split = [cam_id, group_id, img_id]
            save_foulder = foulder + cam_id + '/' + group_id
            if not os.path.exists(save_foulder):
                os.makedirs(save_foulder)
            npy_name = foulder + str1.join(new_name_split) + '.npy'            
            np.save(npy_name, heatmap)
            print(npy_name)
            
            # output each channel's visualized image
            # image = np.load(npy_name)                        
            # for i in range(0,image.shape[2]):
            #     plt.imshow(image[i,:,:])
            #     cv2.imwrite(npy_name[0:-4]+'_'+str(i)+".png",image[:,:,i])
            #     plt.show()



            # ipdb.set_trace()
            # hp = np.load("heatmap.npy")
            # print(hp)

        # print('scale', im_scale, 'heatmap.shape',heatmap.shape)
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
                
        out = draw_humans(oriImg, humans)
        # ipdb.set_trace()
        cv2.imwrite('result.png',out)  



# # 2.28 version
# # modify the directory to overall dataset, to get the heatmap .npy result.

# # for filename in os.listdir(r"/scratch/meil3/SYN_DB/cam1"):
# #     print (filename)
# g = os.walk(r"/scratch/meil3/SYN_DB/cam2/0") 
# for path,dir_list,file_list in g:  
#     for file_name in file_list:  
#         fullname = os.path.join(path, file_name) #full filename includes file foulder
#         name_split_string = re.split('[/.]', fullname)
#         cam_id = name_split_string[4]
#         group_id = name_split_string[5]
#         img_id = name_split_string[6]
#         test_image = fullname
#         oriImg = cv2.imread(test_image) # B,G,R order
#         shape_dst = np.min(oriImg.shape[0:2])

#         # Get results of original image

#         with torch.no_grad():
#             paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
#             #helge:
#             str1 = '_'
#             ipdb.set_trace()
#             new_name_split = [name_split_string[4], name_split_string[5], name_split_string[6]]
#             foulder = '/scratch/meil3/Result/SYN_heatmap_npy/'
#             np.save(foulder + str1.join(new_name_split) + '.npy', heatmap)
#             # hp = np.load("heatmap.npy")
#             # print(hp)

#         print('scale', im_scale, 'heatmap.shape',heatmap.shape)
#         humans = paf_to_pose_cpp(heatmap, paf, cfg)
                
#         out = draw_humans(oriImg, humans)
#         cv2.imwrite('result.png',out)   





# modify the directory to overall dataset, to get the heatmap .npy result.
# test_image = './readme/ski.jpg'
# oriImg = cv2.imread(test_image) # B,G,R order
# shape_dst = np.min(oriImg.shape[0:2])

# # Get results of original image

# with torch.no_grad():
#     paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
#     #helge:
#     np.save("heatmap.npy", heatmap)
#     hp = np.load("heatmap.npy")
#     print(hp)

# print('scale', im_scale, 'heatmap.shape',heatmap.shape)
# humans = paf_to_pose_cpp(heatmap, paf, cfg)
        
# out = draw_humans(oriImg, humans)
# cv2.imwrite('result.png',out)   
