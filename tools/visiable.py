import cv2
import torch
from torch.nn.functional import normalize
import numpy as np
from joblib.numpy_pickle_utils import xrange



# def maxminnorm(array):
#     maxcols=array.max(axis=0)
#     mincols=array.min(axis=0)
#     data_shape = array.shape
#     data_rows = data_shape[0]
#     data_cols = data_shape[1]
#     t=np.empty((data_rows,data_cols))
#     for i in xrange(data_cols):
#         t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
#     return t


def visualize_feature_map(item, img_name):

    f_split = torch.chunk(item, 8)  # 8 change to batch size
    c = -1
    for feature in f_split:
        feature_map = feature.squeeze(0)
        feature_map = torch.clamp(feature_map, 0, 1)
        c = c+1
        print(feature_map)
        feature_map = feature_map.detach().cpu().numpy()

        feature_map = feature_map[:, :, ::-1].transpose((2, 1, 0))*255.0

        imgFile = "/home/caoyiwen/slns/MyBES/feature_map_save/" + img_name[c] +".png"

        cv2.imwrite(imgFile, feature_map)



# def visualize_feature_map(item, img_name):
#
#     f_split = torch.chunk(item, 8)  # 8 change to batch size
#     c = 0
#     for feature in f_split:
#         feature_map = feature.squeeze(0)
#
#         feature_map = torch.clamp(feature_map, 0, 1)
#         f = feature_map[0]
#         f = f.unsqueeze(0)
#         unp = torch.cat([f, f, f], 0)
#         unp = unp.detach().cpu().numpy()
#         unp = unp[:, :, ::-1].transpose((2, 1, 0))*255.0
#         # c = c+1
#         # print(feature_map)
#         # feature_map = feature_map.detach().cpu().numpy()
#         # print(feature_map)
#         #
#         # feature_map = feature_map[:, :, ::-1].transpose((2, 1, 0))*255.0
#
#         imgFile = "/home/caoyiwen/slns/MyBES/feature_map_save/" + img_name[0] +".png"
#
#         cv2.imwrite(imgFile, unp)