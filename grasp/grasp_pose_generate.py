import sys
sys.path.append('/opt/data/private/code/NormNet')

from grasp.constant import GRASP_POSE_DICT, ID2OBJ, MECH_EYE_TRANSFORMATION_MATRIX, THREE_FINGER_GRASP_OFFSET,GRASP_OFFSET_DICT
from grasp.pose_transform import camera_to_robot_pose, rotation_matrix_to_euler, euler_to_rotation_matrix,\
                                    pose_to_transform_matrix, transform_matrix_to_pose
# from scripts.infer_pprnet import inference_single_PPRNet, init_model, extract_vertexes_from_obj
from scripts.infer_normnet import inference_single, init_model, extract_vertexes_from_obj
from utils.utils import FPS_sample, read_ply
import numpy as np
import torch
import open3d as o3d
from scipy.spatial import KDTree
from multiprocessing import Pool
from pprnet.NormNet_2 import NormNet
from pprnet.pprnet import PPRNet
import time

from utils.pcd_process_tools import *

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
MODEL_DICT = {0: apple_model, 1: banana_model, 2: hex_nut_model, 3: trapezoidal_block_model, 4: wooden_cylinder_model}

grasper_model_path = '/opt/data/private/code/NormNet/grasp/grasper_model/three_finger_grasper_Zrotated180.obj'
GRASPER_MODEL = extract_vertexes_from_obj(grasper_model_path)

NUM_POINT = 16384
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_grasp_pose(pred_center_offset, pred_mat, pred_cls):
    object_init_pose = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = pred_center_offset
    transformation_matrix[:3, :3] = pred_mat

    candidate_pose = []
    candidate_grasp_pose = GRASP_POSE_DICT[ID2OBJ[int(pred_cls)]]
    candidate_grasp_offset = GRASP_OFFSET_DICT[ID2OBJ[int(pred_cls)]]
    for grasp_pose in candidate_grasp_pose:
        init_grasp_pose = np.copy(object_init_pose)
        init_grasp_pose[:3,:3] = grasp_pose
        init_grasp_pose[0, 3] = candidate_grasp_offset[0]
        init_grasp_pose[1, 3] = candidate_grasp_offset[1]
        init_grasp_pose[2, 3] = candidate_grasp_offset[2]
        # print('transformation_matrix:', transformation_matrix)
        # print('init_grasp_pose:', init_grasp_pose)
        # print('candidate_pose:', np.dot(transformation_matrix, init_grasp_pose))
        candidate_pose.append(np.dot(transformation_matrix, init_grasp_pose))
    return candidate_pose

def voxel_grid_downsample(point_cloud, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    downpcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downpcd.points)

def check_grasp_pose(candidate_poses, scene_point_cloud, GRASPER_MODEL, voxel_size=0.001,threshold=0.001):
    '''
    作用：根据夹爪检查抓取位姿是否和场景点云存在干涉，若存在干涉则去掉
    '''
    scene_point_cloud = voxel_grid_downsample(scene_point_cloud, voxel_size)
    scene_kdtree = KDTree(scene_point_cloud)
    valid_poses = []
    
    for pose in candidate_poses:
        transformed_grasper = (pose[:3, :3] @ GRASPER_MODEL.T).T + pose[:3, 3]
        distances, _ = scene_kdtree.query(transformed_grasper)
        if np.all(distances > threshold) and pose[2,3] > 0.05:  # 假定0.001为容许的最小距离,>0.10是为了保证夹爪在t台子上方
            valid_poses.append(pose)
    
    return valid_poses

def sample_grasp_pose(candidate_pose_list):
    '''
    作用：从candidate_pose选择一个位姿，选择标准是机械臂关键点离桌面最远
    '''
    if not candidate_pose_list:
        return None
    candidate_pose_list.sort(key=lambda pose: pose[2, 3], reverse=True)  # Sort by z-axis value
    return candidate_pose_list[0]

def transform_pose_camera2base(pose_camera_frame, is_tool=False):
    '''
    作用：将抓取位姿从相机坐标系变换到基坐标系下
    param is_tools:判断是不是工具位姿，如果是工具位姿，则需要乘上工具坐标系的变换矩阵
    '''
    if is_tool:
        return camera_to_robot_pose(pose_camera_frame, MECH_EYE_TRANSFORMATION_MATRIX, THREE_FINGER_GRASP_OFFSET, is_matrix=True, return_matrix=True)
    else:
        return np.dot(MECH_EYE_TRANSFORMATION_MATRIX, pose_camera_frame)

def transform_pcd_camera2base(pcd_camera_frame):
    return np.dot(pcd_camera_frame, MECH_EYE_TRANSFORMATION_MATRIX[:3,:3].T) + \
                                    np.tile(np.reshape(MECH_EYE_TRANSFORMATION_MATRIX[:3, 3], [1, 3]), [pcd_camera_frame.shape[0], 1])

def visulize_grasp_pose(grasper_pose_list, scene_point_cloud):
    '''
    作用:将夹爪在场景点云中进行可视化
    '''
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_point_cloud)
    # o3d.visualization.draw_geometries([scene_pcd])

    for pose in grasper_pose_list:
        print(pose)
        grasper_points = (GRASPER_MODEL @ pose[:3, :3].T) + pose[:3, 3]
        grasper_pcd = o3d.geometry.PointCloud()
        grasper_pcd.points = o3d.utility.Vector3dVector(grasper_points)
        o3d.visualization.draw_geometries([scene_pcd, grasper_pcd])

if __name__ == '__main__':
    # time_1 = time.time()
    net_cls = NormNet
    net = init_model(net_cls, '/opt/data/private/code/NormNet/logs/scripts/NormNet2_batch4_noise_scale_range_0.5_1.5_init_pretrained_ppr/epoch389/checkpoint.tar', DEVICE)

    scene_ply_file_path = '/opt/data/private/code/NormNet/real_result/9/pcd_scene.ply'  # 场景点云
    scene_point_cloud = read_ply(scene_ply_file_path)
    background_ply_file_path = '/opt/data/private/code/NormNet/real_result/background_box/pcd_scene.ply'  # 背景点云
    background_point_cloud = read_ply(background_ply_file_path)
    
    point_clouds = remove_overlap(scene_point_cloud, background_point_cloud)
    point_clouds = np.asarray(point_clouds.points)
    # foreground_ply_file_path = "/opt/data/private/code/NormNet/real_result/result_point_cloud_apple.ply" # 前景点云
    # point_clouds = read_ply(foreground_ply_file_path)

    point_clouds = FPS_sample(point_clouds, NUM_POINT)
    point_clouds = point_clouds.reshape(NUM_POINT, 3).astype(np.float32)
    time_1 = time.time()
    cluster_mat_pred, cluster_center_pred, pred_cls_cluster = inference_single(net=net,
                                                        point_clouds=point_clouds,
                                                        model_dict=MODEL_DICT,
                                                        device=DEVICE,
                                                        is_show=False,
                                                        )

    time_2 = time.time()
    print('位姿估计消耗时长：', time_2 - time_1)
    print('物体个数：', len(cluster_center_pred))

    # convert unit from mm to m
    cluster_center_pred = [x / 1000 for x in cluster_center_pred]
    scene_point_cloud /= 1000
    GRASPER_MODEL /= 1000

    # generate grasp pose
    scene_point_cloud_base_frame = transform_pcd_camera2base(scene_point_cloud)
    print(scene_point_cloud_base_frame)
    grasper_pose_list_camera_frame = []
    for mat_pred, center_pred, pred_cls in zip(cluster_mat_pred, cluster_center_pred, pred_cls_cluster):
        candidate_pose = generate_grasp_pose(center_pred, mat_pred, pred_cls)
        for i in range(len(candidate_pose)):
            candidate_pose[i] = transform_pose_camera2base(candidate_pose[i], True)
            # if ID2OBJ[int(pred_cls)] == 'trapezoidal_block':
            # visulize_grasp_pose([candidate_pose[i]], scene_point_cloud_base_frame)
        candidate_pose = check_grasp_pose(candidate_pose, scene_point_cloud_base_frame, GRASPER_MODEL)
        # visulize_grasp_pose(candidate_pose, scene_point_cloud_base_frame)
        candidate_pose = sample_grasp_pose(candidate_pose)
        if candidate_pose is not None:
            grasper_pose_list_camera_frame.append(candidate_pose)
        # visulize_grasp_pose([candidate_pose], scene_point_cloud_base_frame)
    # visulize_grasp_pose(grasper_pose_list_camera_frame, scene_point_cloud_base_frame)
    # convert matrix to 1x6 pose
    grasp_pose_list = []
    for grasp_pose in grasper_pose_list_camera_frame:
        # print('grasp_pose:', grasp_pose)
        # print('transform_matrix_to_pose(grasp_pose):', transform_matrix_to_pose(grasp_pose))
        # print('pose_to_transform_matrix(transform_matrix_to_pose(grasp_pose)):', pose_to_transform_matrix(transform_matrix_to_pose(grasp_pose)))
        grasp_pose_list.append(transform_matrix_to_pose(grasp_pose).tolist())
    print(grasp_pose_list)
    time_3 = time.time()
    print('碰撞检测消耗时长：', time_3 - time_2)
    print('总消耗时长：', time_3 - time_1)
    # time_end = time.time()
    # print('消耗时长：', time_end-time_start)
    # visulize_grasp_pose(grasper_pose_list_camera_frame, scene_point_cloud)