import open3d as o3d
import numpy as np
from scipy.stats import norm

import open3d as o3d
import numpy as np

def remove_overlap(point_cloud_a, point_cloud_b, voxel_size=0.05, distance_threshold=10):
    if isinstance(point_cloud_a, np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_a)
        point_cloud_a = point_cloud
    if isinstance(point_cloud_b, np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_b)
        point_cloud_b = point_cloud

    # o3d.visualization.draw_geometries([point_cloud_a])
    # o3d.visualization.draw_geometries([point_cloud_b])

    # Downsample both point clouds
    pcd_a_down = point_cloud_a.voxel_down_sample(voxel_size)
    pcd_b_down = point_cloud_b.voxel_down_sample(voxel_size)
    
    # Compute the distance from each point in A to its nearest neighbor in B
    distances = pcd_a_down.compute_point_cloud_distance(pcd_b_down)
    
    # Create a boolean mask for points in A that are farther than the threshold
    mask = np.asarray(distances) > distance_threshold
    
    # Select points that are not overlapping
    non_overlapping_points = np.asarray(pcd_a_down.points)[mask]
    
    # Create a new point cloud with non-overlapping points
    pcd_non_overlapping = o3d.geometry.PointCloud()
    pcd_non_overlapping.points = o3d.utility.Vector3dVector(non_overlapping_points)

    # o3d.visualization.draw_geometries([pcd_non_overlapping])
    
    return pcd_non_overlapping

def outier_remove(point_cloud):
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=50,
                                                    std_ratio=3.0)
    inlier_cloud = point_cloud.select_by_index(ind)
    return inlier_cloud

def crop_point_cloud(input_file, output_file):
    # 读取点云
    pcd = o3d.io.read_point_cloud(input_file)

    # 可视化点云并手动选择感兴趣的区域
    print("请在窗口中选择要裁剪的点云区域。")
    print("按 'Y' 键选择，按 'K' 键确认选择。")
    print("按shift+鼠标左键查看选中点的坐标。")
    print("或者使用ctrl+左键单击进行多边形选择")
    print("按“C”获取选定的几何图形并保存")
    print("按“F”切换到自由视图模式")
    print("按control+s保存裁减后的点云文件")
    o3d.visualization.draw_geometries_with_editing([pcd])

    # 保存裁剪后的点云
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"裁剪后的点云已保存到 {output_file}")

def compute_pcd_center_and_noise(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 获取点的坐标
    points = np.asarray(pcd.points)

    # 计算点云的中心位置
    center = np.mean(points, axis=0)
    print("点云中心位置:", center)

    # 计算相对中心的z方向上的偏移
    z_offsets = points[:, 2] - center[2]

    # 拟合高斯分布
    mu, std = norm.fit(z_offsets)

    # 打印拟合结果
    return center, mu, std

if __name__ == '__main__':
    # # 裁剪点云
    # input_file = "/home/liberty/Desktop/grasp/background.ply"  # 输入点云文件路径
    # output_file = "/home/liberty/Desktop/grasp/CroppedPointCloud.ply"  # 输出点云文件路径
    # crop_point_cloud(input_file, output_file)

    # 计算点云中心位置和噪声
    # pcd_file = '/home/liberty/Desktop/grasp/cropped_1.ply'
    # center, mu, std = compute_pcd_center_and_noise(pcd_file)
    # print("点云中心位置:", center)
    # print(f"高斯分布拟合结果: 均值 = {mu}, 标准差 = {std}")

    # 背景裁减,a为前景点云,b为背景点云
    pcd_a = o3d.io.read_point_cloud("/opt/data/private/code/NormNet/real_result/2/pcd_scene.ply")
    pcd_b = o3d.io.read_point_cloud("/opt/data/private/code/NormNet/real_result/background/background.ply")
    result_pcd = remove_overlap(pcd_a, pcd_b)
    o3d.io.write_point_cloud("result_point_cloud.ply", result_pcd)
    o3d.visualization.draw_geometries([result_pcd])

    # 移除异常值
    outier_remove(result_pcd)
    o3d.visualization.draw_geometries([result_pcd])