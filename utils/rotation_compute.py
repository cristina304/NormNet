import numpy as np
import json

# 定义绕 y 轴旋转的函数
def rotation_matrix_y(theta):
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))
    return np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

# 定义绕 x 轴旋转的函数
def rotation_matrix_x(theta):
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))
    return np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])

# 定义绕 z 轴旋转的函数
def rotation_matrix_z(theta):
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

# 计算组合旋转矩阵
def combined_rotation_matrix(theta, rot_func):
    R_ = rot_func(theta)
    return np.dot(R_x_90, R_)

# 检查矩阵是否是单位矩阵
def is_unitary(matrix):
    identity = np.identity(3)
    return np.allclose(np.dot(matrix.T, matrix), identity)

if __name__ == '__main__':
    # 定义绕 x 轴旋转 90 度的旋转矩阵
    R_x_90 = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    # 生成 12 个均匀采样的角度
    angles = np.linspace(0, 330, 12)

    output_json_path = '/opt/data/private/output.json'
    output = []

    # 计算并检查每个旋转矩阵
    for angle in angles:
        R = combined_rotation_matrix(angle, rotation_matrix_x)
        output.append(R.tolist())
        print(f"Rotation matrix for {angle} degrees:")
        print(R)
        print("Is unitary:", is_unitary(R))
    
    with open(output_json_path, 'w') as f:
        json.dump(output, f)