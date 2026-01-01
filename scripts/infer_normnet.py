import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
print(FILE_PATH)
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append('/opt/data/private/code/NormNet')
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import time
import random
from pprnet.NormNet_2 import NormNet, load_checkpoint, save_checkpoint
from pprnet.object_type import ObjectType
import pprnet.utils.eulerangles as eulerangles
import pprnet.utils.eval_util as eval_util
import pprnet.utils.visualize_util as visualize_util
from pprnet.data.IPA_pose_dataset import IPAPoseDataset
from pprnet.data.pointcloud_transforms import PointCloudShuffle, ToTensor, PointCloudJitter
import pprnet.utils.show3d_balls as show3d_balls

from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.cluster import MeanShift
from collections import Counter

import open3d as o3d
from utils.utils import FPS_sample, read_ply

NUM_POINT = 16384
transforms = transforms.Compose(
    [
        # PointCloudShuffle(NUM_POINT),
        ToTensor()
    ]
)

OBJECT_TYPE = [
        ObjectType(type_name='apple', class_idx=0, symmetry_type='finite',
                    lambda_p=[[0.02256567482, 0.0, 0.0], [0.0, 0.02178021144, 0.0], [-0.0, 0.0, 0.02187101965]],
                    G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        ObjectType(type_name='banana', class_idx=1, symmetry_type='finite',
                    lambda_p=[[0.04862649581, 0.0, 0.0], [0.0, 0.01316621683, 0.0], [-0.0, 0.0, 0.01974384423]],
                    G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        ObjectType(type_name='Hex_nut', class_idx=2, symmetry_type='finite',
                    lambda_p=[[0.0138583038, 0.0, 0.0], [0.0, 0.0138583039, 0.0], [-0.0, 0.0, 0.00603320115]],
                    G=[[[0.5, -0.8661, 0], [0.8661, 0.5, 0], [0, 0, 1]],
                            [[-0.5, -0.8661, 0], [0.8661, -0.5, 0], [0, 0, 1]],
                            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                            [[-0.5, 0.8661, 0], [-0.8661, -0.5, 0], [0, 0, 1]],
                            [[0.5, 0.8661, 0], [-0.8661, 0.5, 0], [0, 0, 1]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[1,0,0], [0,-1,0], [0,0,-1]],
                            [[0.5, 0.8661,0], [0.8661, -0.5, 0], [0,0,-1]],
                            [[-0.5, 0.8661, 0], [0.8661, 0.5, 0], [0,0,-1]],
                            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                            [[-0.5, -0.8661,0], [-0.8661,0.5,0], [0,0,-1]],
                            [[0.5, -0.8661, 0], [-0.8661, -0.5, 0], [0,0,-1]],
                            ]), # FIXME:补充旋转轴G
        ObjectType(type_name='trapezoidal_block', class_idx=3, symmetry_type='finite',
                    lambda_p=[[0.01406964991, 0.0, 0.0], [0.0, 0.03412923116, 0.0], [-0.0, 0.0, 0.01930919199]],
                    G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        ObjectType(type_name='wooden_cylinder', class_idx=4, symmetry_type='revolution',
                    lambda_p=0.029382386695575267,
                    retoreflection=True),
        ]

# def _rot_loss_revolution(rot_matrix, rot_label, lambda_p, retoreflection, weight, return_pointwise_loss=False):
#     """ 
#     Calculate weighted rotation loss for revolution objects, privite helper function.
#     Args:
#         rot_matrix: torch.Tensor (M, 3, 3) 
#         rot_label: torch.Tensor (M, 3, 3) 
#         lambda_p: scalar
#         retoreflection: bool
#         weight: torch.Tensor (M,)  
#     Returns:
#         l: scalar , weight loss sum of all M samples
#         w: sum of weight
#         *Note* weighted average loss is l/w
#     """
#     dtype, device = rot_matrix.dtype, rot_matrix.device
#     M = rot_matrix.shape[0]
#     if self.ez_m_3_1 is None or self.ez_m_3_1.shape[0] != M:
#         self.ez_m_3_1 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device, requires_grad=False).view(1,3,1) # (1, 3, 1)
#         self.ez_m_3_1 = self.ez_m_3_1.repeat(M, 1, 1) # (M, 3, 1)
#     ez = self.ez_m_3_1
#     if retoreflection==False:
#         loss = lambda_p * torch.abs(torch.matmul(rot_matrix, ez) - torch.matmul(rot_label, ez)) # (M, 3, 1)
#         loss = loss.squeeze(2)  # (M, 3)
#         loss = torch.sum(loss, dim=1)  # (M,) 
#     else:
#         loss1 = lambda_p * torch.abs(torch.matmul(rot_matrix, ez) - torch.matmul(rot_label, ez)) # (M, 3, 1)
#         loss2 = lambda_p * torch.abs(torch.matmul(rot_matrix, ez) + torch.matmul(rot_label, ez)) # (M, 3, 1)
#         loss = torch.cat([loss1, loss2], dim=-1)    # (M, 3, 2)
#         loss = loss.transpose(1, 2) # (M, 2, 3)
#         loss = torch.sum(loss, dim=-1)  # (M, 2)
#         loss = torch.min(loss, dim=-1)[0]   #(M,)
#     x = loss    #(M,)

#     if weight is not None:
#         loss = torch.mul(loss, weight)    # (M,) 
#         l = torch.sum(loss)    # scalar
#         w = torch.sum(weight)   # scalar
#     else:
#         l = torch.sum(loss)    # scalar
#         w = 1.0*M 

#     if not return_pointwise_loss:
#         return l, w
#     else:
#         return l, w, x

# def _rot_loss_finite(rot_matrix, rot_label, lambda_p, G, weight=None, return_pointwise_loss=False):
#     """ 
#     Calculate weighted rotation loss for objects finite symmetry, privite helper function.
#     Args:
#         rot_matrix: torch.Tensor (M, 3, 3) 
#         rot_label: torch.Tensor (M, 3, 3) 
#         lambda_p: List[List[float]] (3, 3)
#         G: List[ List[List[float]] (3, 3)  ], len(G)==K, objects with K equal poses
#         weight: torch.Tensor (M,)  
#     Returns:
#         l: scalar , weight loss sum of all M samples
#         w: sum of weight
#         *Note* weighted average loss is l/w
#     """
#     dtype, device = rot_matrix.dtype, rot_matrix.device
#     M = rot_matrix.shape[0]
#     K = len(G)
#     if self.G_list_m_3_3 is None or self.G_list_m_3_3[0].shape[0]!=M:
#         G = [  torch.tensor(g, dtype=dtype, device=device, requires_grad=False) for g in G ]
#         self.G_list_m_3_3 = [ g.unsqueeze(0).repeat(M,1,1) for g in G ] # list of (M, 3, 3)
#     G = self.G_list_m_3_3
#     if self.lambda_p_m_3_3 is None or self.lambda_p_m_3_3.shape[0]!=M:
#         lambda_p = torch.tensor(lambda_p, dtype=dtype, device=device, requires_grad=False).view(1, 3, 3)    # (1, 3, 3)
#         self.lambda_p_m_3_3 = lambda_p.repeat(M, 1, 1)    # (M, 3, 3)
#     lambda_p = self.lambda_p_m_3_3  # (M, 3, 3)

#     P = torch.matmul(rot_matrix, lambda_p)  # (M, 3, 3)
#     P = torch.unsqueeze(P, -1)  # (M, 3, 3, 1)
#     P = P.repeat(1,1,1,K)   # (M, 3, 3, K)

#     L_list = []
#     for i in range(K):
#         l = torch.matmul(torch.matmul(rot_label, G[i]), lambda_p) # (M, 3, 3)
#         L_list.append(torch.unsqueeze(l, -1)) # (M, 3, 3, 1)
#     L = torch.cat(L_list, dim=-1) # (M, 3, 3, K)

#     sub = torch.abs(P - L).permute([0,3,1,2]).contiguous() # (M, K, 3, 3)
#     sub = sub.view(M,K,9) # (M, K, 9)
#     dist = torch.sum(sub, dim=-1) # (M, K)
#     min_dist = torch.min(dist, dim=-1)[0] # (M,)
#     x = min_dist

#     if weight is not None:
#         min_dist = torch.mul(min_dist, weight)    # (M,) 
#         l = torch.sum(min_dist)    # scalar
#         w = torch.sum(weight)   # scalar
#     else:
#         l = torch.sum(min_dist)    # scalar
#         w = 1.0*M   # scalar

#     if not return_pointwise_loss:
#         return l, w
#     else:
#         return l, w, x

def show_points(point_array, color_array=None, radius=3):
    assert isinstance(point_array, list)
    all_color = None
    if color_array is not None:
        assert len(point_array) == len(color_array)
        all_color = [ np.zeros( [ pnts.shape[0] ,3] ) for pnts in point_array]
        for i, c in enumerate(color_array):
            all_color[i][:] = [c[1],c[0],c[2]]
        all_color = np.concatenate(all_color, axis=0)
    all_points = np.concatenate(point_array, axis=0)
    show3d_balls.showpoints(all_points, c_gt=all_color, ballradius=radius, normalizecolor=False)

def extract_vertexes_from_obj(file_name):
    with open(file_name, 'r') as f:
        vertexes = []
        for line in f.readlines():
            line = line.strip()
            if line.startswith('v'):
                words = line.split()[1:]
                xyz = [float(w) for w in words]
                vertexes.append(xyz)
        ori_model_pc = np.array(vertexes)
        # center = ( np.max(ori_model_pc, axis=0) + np.min(ori_model_pc, axis=0) ) / 2.0
        # ori_model_pc = ori_model_pc - center
    return ori_model_pc

def inference_single(net, point_clouds, model_dict, device, is_show=False):
    input_point_ori = point_clouds

    inputs = {
        'point_clouds': torch.from_numpy(point_clouds).to(device).unsqueeze(0),
    }

    # Forward pass
    with torch.no_grad():
        time_start = time.time()
        pred_results, _ = net.inference(inputs)
        print("Forward time:", time.time()-time_start)

    # pred_trans_val = pred_results[0][0].cpu().numpy()
    # pred_mat_val = pred_results[1][0].cpu().numpy()
    # pred_vis_val = pred_results[2][0].cpu().numpy()
    # pred_cls_logits_val = pred_results[3][0].cpu().numpy()
    pred_trans_val = torch.cat([x for x in pred_results[0]], dim=0).cpu().numpy()
    pred_mat_val = torch.cat([x for x in pred_results[1]], dim=0).cpu().numpy()
    pred_vis_val = torch.cat([x for x in pred_results[2]], dim=0).cpu().numpy()
    pred_cls_logits_val = torch.cat([x for x in pred_results[3]], dim=0).cpu().numpy()


    vs_picked_idx = pred_vis_val > 0.60

    input_point = input_point_ori
    pred_trans_val = pred_trans_val[vs_picked_idx]
    pred_mat_val = pred_mat_val[vs_picked_idx]
    pred_cls_logits_val = pred_cls_logits_val[vs_picked_idx]

    print(input_point_ori.shape)
    print(pred_trans_val.shape)
    print(pred_mat_val.shape)
    print(pred_cls_logits_val.shape)

    # print('pred_trans_val', pred_trans_val.shape)
    # print('pred_mat_val', pred_mat_val.shape)
    # pred_trans_val = pred_trans_val[0]
    # pred_mat_val = pred_mat_val
    
    ms = MeanShift(bandwidth=20, bin_seeding=True, cluster_all=False, min_bin_freq=200)
    # ms = MeanShift(bandwidth=20, bin_seeding=True, cluster_all=False, min_bin_freq=20)
    ms.fit(pred_trans_val)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_


    # # Number of clusters in labels, ignoring noise if present. 
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # print(n_clusters)


    color_cluster = [np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]) for i in range(n_clusters)]
    color_per_point = np.ones([pred_trans_val.shape[0], pred_trans_val.shape[1]]) * 255
    for idx in range(color_per_point.shape[0]):
        if labels[idx] != -1:
            color_per_point[idx, :] = color_cluster[labels[idx]]
    

    pred_trans_cluster = [[] for _ in range(n_clusters)]
    pred_mat_cluster = [[] for _ in range(n_clusters)]
    pred_cls_cluster = [[] for _ in range(n_clusters)]
    for idx in range(pred_trans_val.shape[0]):
        if labels[idx] != -1:
            pred_trans_cluster[labels[idx]].append(np.reshape(pred_trans_val[idx], [1, 3]))
            pred_mat_cluster[labels[idx]].append(np.reshape(pred_mat_val[idx], [1, 3, 3]))
            pred_cls_cluster[labels[idx]].append(np.argmax(pred_cls_logits_val[idx]))
    pred_trans_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_trans_cluster]
    pred_mat_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_mat_cluster]
    pred_cls_cluster = [Counter(cluster).most_common(1)[0][0] for cluster in pred_cls_cluster]

    cluster_center_pred = [ np.mean(cluster, axis=0) for cluster in pred_trans_cluster]


    cluster_mat_pred = []
    # ms_rotation = MeanShift(bandwidth=10, bin_seeding=True, cluster_all=False, min_bin_freq=20)
    for i, mat_cluster in enumerate(pred_mat_cluster):
        all_quat = np.zeros([mat_cluster.shape[0], 4])
        for idx in range(mat_cluster.shape[0]):
            all_quat[idx] = eulerangles.mat2quat(mat_cluster[idx])

        # ms_rotation.fit(all_quat * 1000)
        # ms_rotation.fit(mat_cluster[:, :3, 2] * 1000)
        # rot_labels_ = ms_rotation.labels_
        # rot_cluster_centers = ms_rotation.cluster_centers_
        # n_rot_clusters = len(set(rot_labels_)) - (1 if -1 in labels else 0)
        # print(n_rot_clusters)

        quat = eulerangles.average_quat(all_quat)
        # quat = all_quat[500]
        cluster_mat_pred.append(eulerangles.quat2mat(quat))
    # exit(0)

    if is_show:
        all_model_point = []
        all_model_color = []
        for cluster_idx in range(n_clusters):
            model_pointcloud = model_dict[int(pred_cls_cluster[cluster_idx])]
            all_model_color.append(np.tile(color_cluster[cluster_idx], [model_pointcloud.shape[0], 1]))
            all_model_point.append(np.dot(model_pointcloud, cluster_mat_pred[cluster_idx].T) + \
                                                    np.tile(np.reshape(cluster_center_pred[cluster_idx], [1, 3]), [model_pointcloud.shape[0], 1]))

        all_model_color = np.concatenate(all_model_color, axis=0)
        all_model_point = np.concatenate(all_model_point, axis=0)

        show3d_balls.showpoints(input_point_ori, ballradius=5)
        # show3d_balls.showpoints(input_point_ori, c_gt=color_per_point, ballradius=5)
        show3d_balls.showpoints(pred_trans_val, c_gt=color_per_point, ballradius=5)
        show3d_balls.showpoints(input_point, c_gt=color_per_point, ballradius=5)
        show3d_balls.showpoints(all_model_point, c_gt=all_model_color, ballradius=5, normalizecolor=False)
        show_points([all_model_point, input_point_ori], [[255,0,0], [255,255,255]], radius=5) #

    # print(len(cluster_mat_pred))
    # print(len(cluster_center_pred))
    # print(len(pred_cls_cluster))
    # print(pred_cls_cluster)
    return cluster_mat_pred, cluster_center_pred, pred_cls_cluster


def inference_batch():
    pass


def init_model(net_cls, checkpoint_path, device):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    object_types = [
        ObjectType(type_name='apple', class_idx=0, symmetry_type='finite',
                    lambda_p=[[0.02256567482, 0.0, 0.0], [0.0, 0.02178021144, 0.0], [-0.0, 0.0, 0.02187101965]],
                    G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        ObjectType(type_name='banana', class_idx=1, symmetry_type='finite',
                    lambda_p=[[0.04862649581, 0.0, 0.0], [0.0, 0.01316621683, 0.0], [-0.0, 0.0, 0.01974384423]],
                    G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        ObjectType(type_name='Hex_nut', class_idx=2, symmetry_type='finite',
                    lambda_p=[[0.0138583038, 0.0, 0.0], [0.0, 0.0138583039, 0.0], [-0.0, 0.0, 0.00603320115]],
                    G=[[[0.5, -0.8661, 0], [0.8661, 0.5, 0], [0, 0, 1]],
                            [[-0.5, -0.8661, 0], [0.8661, -0.5, 0], [0, 0, 1]],
                            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                            [[-0.5, 0.8661, 0], [-0.8661, -0.5, 0], [0, 0, 1]],
                            [[0.5, 0.8661, 0], [-0.8661, 0.5, 0], [0, 0, 1]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[1,0,0], [0,-1,0], [0,0,-1]],
                            [[0.5, 0.8661,0], [0.8661, -0.5, 0], [0,0,-1]],
                            [[-0.5, 0.8661, 0], [0.8661, 0.5, 0], [0,0,-1]],
                            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                            [[-0.5, -0.8661,0], [-0.8661,0.5,0], [0,0,-1]],
                            [[0.5, -0.8661, 0], [-0.8661, -0.5, 0], [0,0,-1]],
                            ]), # FIXME:补充旋转轴G
        ObjectType(type_name='trapezoidal_block', class_idx=3, symmetry_type='finite',
                    lambda_p=[[0.01406964991, 0.0, 0.0], [0.0, 0.03412923116, 0.0], [-0.0, 0.0, 0.01930919199]],
                    G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        ObjectType(type_name='wooden_cylinder', class_idx=4, symmetry_type='revolution',
                    lambda_p=0.029382386695575267,
                    retoreflection=True),
        ]
    backbone_config = {
        'npoint_per_layer': [4096,1024,256,64],
        'radius_per_layer': [[10,20,30],[30,45,60],[60,80,120],[120,160,240]]
    }
    net = net_cls(object_types, 'pointnet2msg', backbone_config, True, None, False)
    net.to(device)
    net, _, _ = load_checkpoint(checkpoint_path, net)
    net.eval()  #
    return net


if __name__ == '__main__':

    # load object model
    apple_model_path = '/opt/data/private/code/NormNet/generate_dataset/models/sampled_model/apple.obj'
    apple_model = extract_vertexes_from_obj(apple_model_path)
    banana_model_path = '/opt/data/private/code/NormNet/generate_dataset/models/sampled_model/banana.obj'
    banana_model = extract_vertexes_from_obj(banana_model_path)
    hex_nut_model_path = '/opt/data/private/code/NormNet/generate_dataset/models/sampled_model/hex_nut.obj'
    hex_nut_model = extract_vertexes_from_obj(hex_nut_model_path)
    trapezoidal_block_model_path = '/opt/data/private/code/NormNet/generate_dataset/models/sampled_model/trapezoidal_block.obj'
    trapezoidal_block_model = extract_vertexes_from_obj(trapezoidal_block_model_path)
    wooden_cylinder_model_path = '/opt/data/private/code/NormNet/generate_dataset/models/sampled_model/wooden_cylinder.obj'
    wooden_cylinder_model = extract_vertexes_from_obj(wooden_cylinder_model_path)
    model_dict = {0:apple_model, 1:banana_model, 2:hex_nut_model, 3:trapezoidal_block_model, 4:wooden_cylinder_model}

    # build net
    checkpoint_path = "/opt/data/private/code/NormNet/logs/scripts/NormNet2_batch4_noise_scale_range_0.5_1.5_init_pretrained_ppr/epoch389/checkpoint.tar"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    object_types = [
        ObjectType(type_name='apple', class_idx=0, symmetry_type='finite',
                    lambda_p=[[0.02256567482, 0.0, 0.0], [0.0, 0.02178021144, 0.0], [-0.0, 0.0, 0.02187101965]],
                    G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        ObjectType(type_name='banana', class_idx=1, symmetry_type='finite',
                    lambda_p=[[0.04862649581, 0.0, 0.0], [0.0, 0.01316621683, 0.0], [-0.0, 0.0, 0.01974384423]],
                    G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        ObjectType(type_name='Hex_nut', class_idx=2, symmetry_type='finite',
                    lambda_p=[[0.0138583038, 0.0, 0.0], [0.0, 0.0138583039, 0.0], [-0.0, 0.0, 0.00603320115]],
                    G=[[[0.5, -0.8661, 0], [0.8661, 0.5, 0], [0, 0, 1]],
                            [[-0.5, -0.8661, 0], [0.8661, -0.5, 0], [0, 0, 1]],
                            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                            [[-0.5, 0.8661, 0], [-0.8661, -0.5, 0], [0, 0, 1]],
                            [[0.5, 0.8661, 0], [-0.8661, 0.5, 0], [0, 0, 1]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[1,0,0], [0,-1,0], [0,0,-1]],
                            [[0.5, 0.8661,0], [0.8661, -0.5, 0], [0,0,-1]],
                            [[-0.5, 0.8661, 0], [0.8661, 0.5, 0], [0,0,-1]],
                            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                            [[-0.5, -0.8661,0], [-0.8661,0.5,0], [0,0,-1]],
                            [[0.5, -0.8661, 0], [-0.8661, -0.5, 0], [0,0,-1]],
                            ]), # FIXME:补充旋转轴G
        ObjectType(type_name='trapezoidal_block', class_idx=3, symmetry_type='finite',
                    lambda_p=[[0.01406964991, 0.0, 0.0], [0.0, 0.03412923116, 0.0], [-0.0, 0.0, 0.01930919199]],
                    G=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        ObjectType(type_name='wooden_cylinder', class_idx=4, symmetry_type='revolution',
                    lambda_p=0.029382386695575267,
                    retoreflection=True),
        ]
    backbone_config = {
        'npoint_per_layer': [4096,1024,256,64],
        'radius_per_layer': [[10,20,30],[30,45,60],[60,80,120],[120,160,240]]
    }
    net = NormNet(object_types, 'pointnet2msg', backbone_config, True, None, False)
    net.to(device)
    net, _, _ = load_checkpoint(checkpoint_path, net)
    net.eval() #

    # inference with h5
    # h5_file = '/opt/data/private/data/dataset/realgrasp/h5_dataset/val/cycle_0049/028.h5'
    # f = h5py.File(h5_file)
    # point_clouds = f['data'][:].reshape(NUM_POINT, 3).astype(np.float32) * 1000
    # inference_single(net=net, point_clouds=point_clouds, model_dict=model_dict, device=device)

    # inference with ply
    ply_file_path = "/opt/data/private/code/NormNet/real_result/result_point_cloud.ply"
    point_clouds = read_ply(ply_file_path)
    point_clouds = FPS_sample(point_clouds, NUM_POINT)
    point_clouds = point_clouds.reshape(NUM_POINT, 3).astype(np.float32)
    cluster_mat_pred, cluster_center_pred, pred_cls_cluster = inference_single(net=net,
                                                                        point_clouds=point_clouds,
                                                                        model_dict=model_dict,
                                                                        device=device,
                                                                        is_show=True)