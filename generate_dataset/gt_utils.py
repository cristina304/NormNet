"""
generate ground truth from physics simulation for render results

@author: Lv Weijie
@checked: Lv Weijie
@example usage: python gt_utils.py
@time: 2022/1/21

load physics results -> define camera frames -> 
pose of world frames, name, index, layer

return: name, pose of camera frames, index, fg_prob

"""
import csv
import cv2
import math
import numpy as np
import os
import json
import nibabel.quaternions as nq
from easydict import EasyDict
import yaml


CONFIG_PATH = "./config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = EasyDict(config)

CAMERA_LOCATION = config.cam_location_x_y_z
CAMERA_ROTATION = config.cam_rotation_qw_qx_qy_qz

PHYSICS_PATH = os.path.join(config.output_dir, 'physics_result')
GT_PATH = os.path.join(config.output_dir, 'gt')


''' read csv data
Args:
    csv_path:the path of csv file
Returns:None
'''   
def read_csv(csv_path):        
    with open(csv_path,'r') as csv_file:  
        all_lines=csv.reader(csv_file) 
        list_file = [i for i in all_lines]  
    array_file = np.array(list_file)[1:] # exclude titles
    obj_name = array_file[:,0]
    obj_index = array_file[:,1].astype('int')
    pose = array_file[:,2:9].astype('float32')
    return obj_name, obj_index, pose


def generate_gt(pose_world):
    ''' generate pose in camera coodinate system

    Args:
        pose_world: spartparts' pose in world coordinate system 

    Returns:
        pose_camera: spartparts' pose in camera coordinate system 
    '''  

    # camera external parameter
    t_c2w = np.array(CAMERA_LOCATION).reshape(1, 3)
    quat_c2w = np.array(CAMERA_ROTATION)
    R_c2w  = nq.quat2mat(quat_c2w).reshape(3,3)

    #t and R in world coordinate system
    t_world = pose_world[:,:3]
    quat_world = pose_world[:,3:]
    R_world = [nq.quat2mat(quat) for quat in quat_world]

    # generate t and R in camera coordinate system
    t_camera = np.array([(np.dot(R_c2w, t) + t_c2w).reshape(3) \
                      for t in t_world])
    R_camera = np.array([np.dot(R_c2w, R.reshape(3,3)).reshape(9) \
                      for R in R_world])
    pose_camera = np.concatenate((t_camera, R_camera),axis=-1)
    return pose_camera

if __name__ == "__main__":
    # cycle_range, scene_range = [300,600],[1,40]
    cycle_range, scene_range = [0,300],[1,40]
    # cycle_range, scene_range = [1200,1400],[1,40]
    # cycle_range, scene_range = [1400,1600],[1,40]
    
    for cycle_id in range(cycle_range[0],cycle_range[1]):
        for scene_id in range(scene_range[0],scene_range[1]):
            scene_name = '{:0>3}'.format(scene_id)
            csv_path = os.path.join(PHYSICS_PATH, \
                       'cycle_{:0>4}'.format(cycle_id),\
                        "{:0>3}.csv".format(scene_id))
            name_temp, index, pose_world = read_csv(csv_path)
            # name = []
            # for i in name_temp:
            #     if 'part' in i :
            #         if '1' in i :
            #             name.append(1)
            #         if '2' in i :
            #             name.append(2)
            #         if '3' in i :
            #             name.append(3)
            #         if '4' in i :
            #             name.append(4)
            #         if '5' in i :
            #             name.append(5)
            #         if '6' in i :
            #             name.append(6)
            #         if '7' in i :
            #             name.append(7)
            #     else:
            #         name.append(0)
            # name = np.array(name).astype('float32')





           
            # print(name_temp)

        
            # exit()
            # fg_prob = np.zeros(layer_index.shape)
            # fg_prob[layer_index==layer_dict[layer_name]] = 1
            pose_camera = generate_gt(pose_world)

            headers = ["class_name","id","x", "y", "z", "R1", "R2", "R3", "R4","R5", "R6", "R7", "R8","R9"]
            temp = np.concatenate((name_temp.reshape(-1,1), index.reshape(-1,1)),axis=-1)
            temp = np.concatenate((temp, pose_camera),axis=-1)
            # result = np.concatenate((temp, fg_prob.reshape(-1,1)),axis=-1).tolist()
            result = temp.tolist()
            
            assert len(result[0]) == len(headers)
            save_path = os.path.join(GT_PATH, 'cycle_{:0>4}'.format(cycle_id))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_loc = save_path + '/' + scene_name + '.csv'
            with open(file_loc, 'w') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(headers)
                    f_csv.writerows(result)
        print(f'The {cycle_id} cycle is completed')
