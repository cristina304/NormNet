import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix_to_euler(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    singular = sy < 1e-6  # 判断是否为奇异情况

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])

    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    print(x,y,z)
    return x, y, z


# 将欧拉角转换为旋转矩阵
def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return R_z @ R_y @ R_x

def pose_to_transform_matrix(pose):
    """
    将 6D 位姿转换为 4x4 齐次变换矩阵
    :param pose: 6D 位姿 [x, y, z, rx, ry, rz]
    :return: 4x4 齐次变换矩阵
    """
    # 提取平移分量
    x, y, z = pose[:3]

    # 提取欧拉角（假设顺序为 XYZ）
    rx, ry, rz = pose[3:]

    # 将欧拉角转换为旋转矩阵
    rotation = euler_to_rotation_matrix(rx, ry, rz)

    # 构建 4x4 齐次变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = [x, y, z]

    return transform_matrix

def transform_matrix_to_pose(transform_matrix):
    """
    将 4x4 齐次变换矩阵转换为 6D 位姿
    :param transform_matrix: 4x4 齐次变换矩阵
    :return: 6D 位姿 [x, y, z, rx, ry, rz]
    """
    # 提取平移分量
    x, y, z = transform_matrix[:3, 3]

    # 提取旋转矩阵并转换为欧拉角（假设顺序为 XYZ）
    rotation = transform_matrix[:3,:3]
    # print('rotation:', rotation)
    rx, ry, rz = rotation_matrix_to_euler(rotation)

    return np.array([x, y, z, rx, ry, rz])


def adjust_for_end_effector_offset(transform_matrix, offset):
    """
    调整变换矩阵以考虑末端执行器的偏移
    :param transform_matrix: 4x4 齐次变换矩阵
    :param offset: 沿 Z 轴的偏移量（单位：米）
    :return: 调整后的 4x4 齐次变换矩阵
    """
    # 创建偏移变换矩阵
    offset_matrix = np.eye(4)
    offset_matrix[2, 3] = offset  # 沿 Z 轴移动

    # 将偏移应用到原始变换矩阵
    adjusted_matrix = transform_matrix @ offset_matrix

    return adjusted_matrix


def camera_to_robot_pose(camera_pose, transformation_matrix, offset, is_matrix=False, return_matrix=False):
    """
    将相机坐标系下的 6D 位姿转换为机械臂坐标系下的 6D 位姿，并考虑末端执行器的偏移
    :param camera_pose: 相机坐标系下的 6D 位姿 [x, y, z, rx, ry, rz]
    :param transformation_matrix: 相机到机械臂的 4x4 齐次变换矩阵
    :param offset: 末端执行器的偏移量（单位：米）
    :param is_matrix: 是否是齐次矩阵
    :return: 机械臂坐标系下的 6D 位姿
    """
    # 将相机坐标系下的 6D 位姿转换为齐次变换矩阵
    if not is_matrix:
        camera_pose = pose_to_transform_matrix(camera_pose)

    # 将相机坐标系下的位姿变换到机械臂坐标系
    robot_transform = transformation_matrix @ camera_pose

    # 考虑末端执行器的偏移
    adjusted_robot_transform = adjust_for_end_effector_offset(robot_transform, offset)

    # print('camera_pose:', camera_pose)
    # print('transformation_matrix:', transformation_matrix)
    # print('robot_transform:', robot_transform)
    # print('adjusted_robot_transform:', adjusted_robot_transform)
    # print('----------------------------------------------------')

    # 将最终的齐次变换矩阵转换回 6D 位姿
    if not return_matrix:
        robot_pose = transform_matrix_to_pose(adjusted_robot_transform)
    else:
        robot_pose = adjusted_robot_transform

    return robot_pose


if __name__ == "__main__":
    # 输入数据
    camera_pose = np.array([0.073875, -0.065959, 1.071883, 1.047198, -1.429257, 0.741629])
    transformation_matrix = np.array([
        [0.9979146513598224, 0.0221580157807878, -0.06062483763330029, 0.5852961463611065],
        [0.019624530039612065, -0.9989218916606913, -0.04207056193767156, 0.04200634764514499],
        [-0.06149167766560171, 0.04079309620126187, -0.9972736318985804, 1.1400974681691178],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # # 计算逆矩阵
    # inverse_matrix = np.linalg.inv(transformation_matrix)

    # print("Inverse Matrix:")
    # print(inverse_matrix)
    # exit()

    # end_effector_offset = -0.18835  # 10 厘米（可变量）

    # 计算机械臂坐标系下的抓取点 6D 位姿
    robot_pose = camera_to_robot_pose(camera_pose, transformation_matrix, end_effector_offset)

    # 输出结果
    print("机械臂坐标系下的抓取点 6D 位姿：")
    print(robot_pose)