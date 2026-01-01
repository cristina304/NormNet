# -*- coding:utf-8 -*-
import csv
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import math
import numpy as np
import os
import json
import nibabel.quaternions as nq



CONFIG_PATH = "./config_multi.json"
JSON_LOAD = open(CONFIG_PATH).read()
CONFIG_FILE = json.loads(JSON_LOAD)

CAMERA_LOCATION = CONFIG_FILE["cam_location_x_y_z"]
CAMERA_ROTATION = CONFIG_FILE["cam_rotation_qw_qx_qy_qz"]
PHYSICS_PATH = os.path.join(CONFIG_FILE["output_dir"], 'physics_result')
GT_PATH = os.path.join(CONFIG_FILE["output_dir"], 'gt')
SEGMENT_PATH = os.path.join(CONFIG_FILE["output_dir"], 'segment_images')
if not os.path.exists(GT_PATH):
    os.mkdir(GT_PATH)

def read_parameter(txt_path):
    with open(txt_path, 'r') as f:
        parameter = f.readlines()
    return parameter



def read_csv(csv_path):
    with open(csv_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader, None)
        obj_id = []
        obj_name = []
        obj_class = []
        for row in spamreader:
            obj_id.append(row[1])
            obj_name.append(row[0])
            obj_class.append(row[-1])
    pose = np.loadtxt(csv_path,
                      delimiter=",",
                      skiprows=1,
                      usecols=(2, 3, 4, 5, 6, 7, 8))
    # guarantee the pose in the form of matrix
    if pose.ndim == 1:
        pose = pose.reshape(1, pose.shape[0])
    return obj_name, obj_id, pose, obj_class


def generate_gt(obj_name, obj_index,pose,segment_path):
    # camera external parameter
    T_old = np.array([CAMERA_LOCATION[0], CAMERA_LOCATION[1], CAMERA_LOCATION[2]])
    T = T_old.reshape(3, 1)
    Q_camera = np.array([CAMERA_ROTATION[0], CAMERA_ROTATION[1], CAMERA_ROTATION[2], CAMERA_ROTATION[3]])
    R_camera =nq.quat2mat(Q_camera)

    # generate t and R
    pose_c = np.zeros((len(obj_index),12))
    for index in range(len(obj_index)):
        w_cent=np.array([pose[index][0],pose[index][1],pose[index][2]])
        w_Q = np.array([pose[index][3], pose[index][4], pose[index][5], pose[index][6]])
        w_R = nq.quat2mat(w_Q)
        matrix_c2w=np.zeros([4,4])
        matrix_w=np.zeros([4,4])
        matrix_c2w[:3,3]=T_old
        matrix_c2w[:3,:3]=R_camera
        matrix_w[:3,3]=w_cent
        matrix_w[:3,:3]=w_R
        matrix_c2w[3,3]=1
        matrix_w[3,3]=1
        matrix_c=np.dot(np.linalg.pinv(matrix_c2w),matrix_w)
        c_R = matrix_c[:3,:3]
        c_cent = matrix_c[:3,3]  
        pose_c[index][:3] = c_cent.reshape(1,-1)
        pose_c[index][3:] = c_R.reshape(1,-1)

    # generate vs label
    files_segment = os.listdir(segment_path)
    segment = cv2.imread('' + segment_path + '/' + files_segment[0] + '', cv2.IMREAD_UNCHANGED)
    obj_id = []
    # for v in range(segment.shape[0]):
    #     for u in range(segment.shape[1]):
    #         if (segment[v][u][0] == 0.0 and segment[v][u][2] == 1.0):
    #             obj_id.append(int(round(segment[v][u][1] * (len(obj_index) - 1))))
    # obj_id = np.array(obj_id)
    obj_id = np.round(segment[:,:,1][segment[:,:,2]==1]*(len(obj_index)-1))

    # print('obj_id:',obj_id,'obj_id1:',obj_id1)
    # if (obj_id.all()==obj_id1.all()):
    #     print(True)
    # exit()
    d = {}
    for index_n in obj_id:
        if index_n not in d:
            # print(index_n)
            d[index_n] = 1
        else:
            d[index_n] += 1
    print('d:', d)
    max_index_num = {}
    for index in range(len(obj_index)):
        key_name = obj_name[index]
        try:
            point_num = d[index]
        except:
            continue

        if key_name in max_index_num:
            max_index_num[key_name] = point_num if point_num > max_index_num[key_name] else max_index_num[key_name]
        else:
            max_index_num.update({key_name: point_num})

    # max_index_num = d[max(d, key=d.get)]
    # print('len(obj_index):', len(obj_index))
    # print('max_index_num:', max_index_num)
    vs_label = np.zeros([len(obj_index), 1])
    # print('len(vs_label):', len(vs_label))
    for i in range(len(obj_index)):
        try:
            key_name = obj_name[i]
            vs_label[i] = d[i] / max_index_num[key_name]
        except:
            vs_label[i] = 0.0
    return pose_c , vs_label

if __name__ == "__main__":
    cycle_names = os.listdir(PHYSICS_PATH)
    for cycle_id, c_name in enumerate(cycle_names):
        if 0<=int(c_name.split('_')[-1]) <=100:
            scene_path = os.path.join(PHYSICS_PATH, c_name)
            scene_names = os.listdir(scene_path)
            for scene_id, s_name in enumerate(scene_names):
                # if s_name.split('.')[0].split('_')[-1] == '000':
                    # continue
                csv_path = os.path.join(PHYSICS_PATH, c_name, s_name)
                segment_path = os.path.join(SEGMENT_PATH, c_name, s_name.split('.')[0])
                obj_name, obj_index, pose, obj_class = read_csv(csv_path)
                try:
                    pose_c , vs = generate_gt(obj_name, obj_index, pose, segment_path)
                except:
                    print('Wrong! Please check:',c_name,s_name)
                    pass
                headers = ["class_name","ID","x", "y", "z", "R1", "R2", "R3", "R4","R5", "R6", "R7", "R8","R9", "vs", "class"]
                rows = []
                gt_path = os.path.join(GT_PATH, c_name)
                if not os.path.exists(gt_path):
                    os.makedirs(gt_path)

                print('len(obj_index):', len(obj_index))
                print('len(vs):', len(vs))
                for i in range(len(obj_index)):
                    rows.append((obj_name[i], obj_index[i], pose_c[i][0],pose_c[i][1],pose_c[i][2],pose_c[i][3],pose_c[i][4],pose_c[i][5],pose_c[i][6],\
                    pose_c[i][7],pose_c[i][8],pose_c[i][9],pose_c[i][10],pose_c[i][11],vs[i][0], obj_class[i]))
                    #print(rows)
                file_loc = gt_path + '/' + s_name
                with open(file_loc, 'w') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow(headers)
                        f_csv.writerows(rows)
                print('--------------------------------------------------------')

