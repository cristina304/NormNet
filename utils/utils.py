import h5py
import open3d as o3d
import numpy as np
import torch

def read_h5(file_path):
     f = h5py.File(file_path)
     return f

def read_ply(ply_path):
     # 读取 PLY 文件
     ply_file_path = ply_path
     pcd = o3d.io.read_point_cloud(ply_file_path)

     # 将点云数据转换为 NumPy 数组
     points = np.asarray(pcd.points)
     return points

def FPS_sample(points, target_num_point):
     from pointnet2_ops.pointnet2_utils import furthest_point_sample
     points_transpose = torch.from_numpy(points.reshape(1, points.shape[0], points.shape[1])).float().cuda()
     sampled_idx = furthest_point_sample(points_transpose, target_num_point).cpu().numpy().reshape(target_num_point)
     points = points[sampled_idx]
     return points


if __name__ == '__main__':
     ply_path = "/opt/data/private/code/NormNet/real_result/result_point_cloud.ply"
     point_cloud = read_ply(ply_path)
     point_cloud = FPS_sample(point_cloud, 16384)
     print(point_cloud.shape)
     