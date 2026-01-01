"""
Code to simulate random falling and stacking of objects based on pybullet.
Example usage:
    python run_simulation.py  --num_rollouts 1000 --obj_names cube sylinder --num_obj 10, 10 --render
Aboving code will output 1000 csv files of different stacking layouts using pybullet.GUI server.

Author: Xin Fu
"""
import pybullet as p
import time
import math
import pybullet_data
import csv
import argparse
import numpy as np
import json
import os
# from cal_box import get_l
CONFIG_PATH = "./config_multi.json"
JSON_LOAD = open(CONFIG_PATH, encoding='utf-8').read()
CONFIG_FILE = json.loads(JSON_LOAD)
OBJ_NUM = CONFIG_FILE["obj_num"]
CAMERA_RESOLUTION = CONFIG_FILE["resolution"]
CAMERA_FOCAL_LEN = CONFIG_FILE["focal_length"]
CAMERA_SENSOR_SIZE = CONFIG_FILE["sensor_size"]
CAMERA_LOCATION = CONFIG_FILE["cam_location_x_y_z"]
CAMERA_ROTATION = CONFIG_FILE["cam_rotation_qw_qx_qy_qz"]

# OBJ2ID = {"apple":0, "banana":1, "Hex_nut":2, "milk_box":3, "trapezoidal_block":4, "wooden_cylinder":5}
# ID2OBJ = {0:"apple", 1:"banana", 2:"Hex_nut", 3:"milk_box", 4:"trapezoidal_block", 5:"wooden_cylinder"}
OBJ2ID = {"apple":0, "banana":1, "Hex_nut":2, "trapezoidal_block":3, "wooden_cylinder":4}
ID2OBJ = {0:"apple", 1:"banana", 2:"Hex_nut", 3:"trapezoidal_block", 4:"wooden_cylinder"}
OBJ_NUM = {'apple':1, 'banana':1, 'Hex_nut':4, "trapezoidal_block":4, 'wooden_cylinder':5}

NUM_CLASS = len(OBJ2ID)
CHOOSE_CLASS = 5

print(CONFIG_FILE["object_names"])

def main():
        with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)

        # choose meshScales
        if config["unit_of_obj"] == 'mm':
                meshScale = [0.001, 0.001, 0.001]
        elif config["unit_of_obj"] == 'm':
                meshScale = [1, 1, 1]

        # define range of total objects number
        low = config["number_of_instance_per_scene"][0]
        high = config["number_of_instance_per_scene"][1]
        path = config["output_dir"] + '/' + 'physics_result'
        data_folder = os.path.exists(path)
        if not data_folder:
                os.makedirs(path)

        # file index in file name
        file_index = 0
        # scenes generated so far
        total_num = 0
        cycle_id = 0

        for num_rollout in range(low, high):
                for k in range(config['number_of_repeat']):
                        file_index += 1
                        cycle_id=k
                        scene_id = num_rollout
                        if config["show_GUI"]:
                                cid = p.connect(p.GUI)
                        else:
                                cid = p.connect(p.DIRECT)

                        p.setPhysicsEngineParameter(numSolverIterations=10)
                        p.setTimeStep(1./120.)
                        p.setAdditionalSearchPath(pybullet_data.getDataPath())
                        logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
                        #useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
                        p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
                        #disable rendering during creation.
                        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
                        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
                        #disable tinyrenderer, software (CPU) renderer, we don't use it here
                        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER,0)
                        # p.resetDebugVisualizerCamera(cameraDistance=0.1, cameraYaw=0, cameraPitch=0, cameraTargetPosition=(0,0,l))
                        l = 0
                        # lists of different shapes
                        visualShapedIds = []
                        collisionShapedIds = []
                        object_names = config["object_names"]
                        choose_ids = sorted(np.random.choice(range(0, NUM_CLASS), CHOOSE_CLASS, replace=False))
                        choose_object_name = [ID2OBJ[obj_id] for obj_id in choose_ids]
                        print(choose_object_name)

                        for obj_name in choose_object_name: 
                                obj_name = obj_name
                                filename = config["obj_dir"] + '/' + obj_name + ".obj"
                                visualShapedIds.append(p.createVisualShape(shapeType=p.GEOM_MESH,
                                                                        fileName=filename,
                                                                        meshScale=meshScale))

                                collisionShapedIds.append(p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                                        fileName=filename,
                                                                        meshScale=meshScale))
                        length = config["box"]["length"]
                        width = config["box"]["width"]
                        height = config["box"]["height"]
                        bottom = config["box"]["bottom_thickness"]
                        random_range = config["random_range"]
                        num_classes = len(config["object_names"])

                        # generate numbers of different objects randomly, and sum of these numbers are defined in config["total_number"]
                        mask = [0 for x in range(CHOOSE_CLASS)]
                        for i in range(len(choose_object_name)):
                                object_name = choose_object_name[i]
                                object_total_num = OBJ_NUM[object_name]
                                mask[i] = np.random.choice(range(0, object_total_num+1))

                        mask = np.array(mask)
                        print('mask:', mask)

                        # two dimensional list to save entities of different shapes
                        mb = []
                        i = 0
                        for vShapedId, cShapedId, num in zip(visualShapedIds, collisionShapedIds, mask):
                                print(i)
                                mb.append([])
                                for _ in range(num):
                                        position = []
                                        position.append(np.random.uniform(-random_range[0], random_range[0]))
                                        position.append(np.random.uniform(-random_range[1], random_range[1]))
                                        position.append(np.random.uniform(random_range[2] + config["box"]["bottom_thickness"],
                                                                        random_range[3] + config["box"]["bottom_thickness"]))
                                        rand_euler_angle = np.random.uniform(-2.0*math.pi, 2.0*math.pi, [3] )
                                        rand_quat = p.getQuaternionFromEuler(rand_euler_angle)
                                        mb[i].append(p.createMultiBody(baseMass=100000,
                                                                       baseInertialFramePosition=config["box"]["position"],
                                                                       baseCollisionShapeIndex=cShapedId,
                                                                       baseVisualShapeIndex=vShapedId,
                                                                       basePosition = position,
                                                                       baseOrientation = rand_quat,
                                                                       useMaximalCoordinates=False))
                                        p.changeVisualShape(mb[i][-1], -1, rgbaColor=[1,0,0,1])
                                i += 1

                        box_position = config["box"]["position"]

                        visualShapedId1 = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                              halfExtents=[length/2, width/2, height/2])

                        collisionShapedId1 = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                                    halfExtents=[length/2, width/2, height/2])

                        visualShapedId2 = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                              halfExtents=[length/2, width/2, bottom/2])

                        collisionShapedId2 = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                                    halfExtents=[length/2, width/2, bottom/2])
                        block_positions = [[0, width, height/2],
                                          [0, -width, height/2],
                                          [length, 0, height/2],
                                          [-length, 0, height/2]]

                        for bposition in block_positions:
                                mb1 = p.createMultiBody(baseMass=0,
                                                #   baseInertialFramePosition=[0,0,0],
                                                  baseCollisionShapeIndex=collisionShapedId1,
                                                  baseVisualShapeIndex = visualShapedId1,
                                                  basePosition = bposition,
                                                  useMaximalCoordinates=False)
                                p.changeVisualShape(mb1, -1, rgbaColor=[0,1,0,1])
                        mb1 = p.createMultiBody(baseMass=0,
                                                # baseInertialFramePosition=[0,0,0],
                                                baseCollisionShapeIndex=collisionShapedId2,
                                                baseVisualShapeIndex = visualShapedId2,
                                                basePosition = [0, 0, bottom/2],
                                                useMaximalCoordinates=False)
                        #####用于查看是否有穿模#############
                        # p.resetDebugVisualizerCamera(cameraDistance=3*l, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=(0,0,l))
                        ##################################
                        p.changeVisualShape(mb1, -1, rgbaColor=[0,0,1,1])

                        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
                        p.setGravity(0,0,-10)

                        if config["show_GUI"]:
                                headers = ["Type", "Index",  "x", "y", "z", "w", "i", "j", "k", "class"]
                                rows = []
                                No = -1
                                time.sleep(1)
                                for _ in range(720):
                                        time.sleep(0.03)
                                        p.stepSimulation()
                                for i, name in enumerate(choose_object_name):
                                        for Id in mb[i]:
                                                No += 1
                                                final_position, angle = p.getBasePositionAndOrientation(Id) 
                                                rows.append((name, No,
                                                             final_position[0], final_position[1], final_position[2]-config["box"]["bottom_thickness"],
                                                             angle[3], angle[0], angle[1], angle[2], OBJ2ID[name]))

                                out_cycle_dir = os.path.join(path, 'cycle_{:0>4}'.format(cycle_id))
                                if not os.path.exists(out_cycle_dir):
                                        os.mkdir(out_cycle_dir)
                                file_loc = out_cycle_dir + '/' + "{:0>3}.csv".format(scene_id)
                                with open(file_loc, 'w') as f:
                                        f_csv = csv.writer(f)
                                        f_csv.writerow(headers)
                                        f_csv.writerows(rows)
                                print("next?")
                                next = input()

                        else:
                                headers = ["Type", "Index",  "x", "y", "z", "w", "i", "j", "k",  "class"]
                                rows = []
                                No = -1
                                for _ in range(720):
                                        p.stepSimulation()
                                for i, name in enumerate(choose_object_name):
                                        print(name)
                                        print(i)
                                        for Id in mb[i]:
                                                No += 1
                                                final_position, angle = p.getBasePositionAndOrientation(Id)
                                                rows.append((name, No,
                                                             final_position[0], final_position[1], final_position[2]-config["box"]["bottom_thickness"],
                                                             angle[3], angle[0], angle[1], angle[2], OBJ2ID[name]))
                                out_cycle_dir = os.path.join(path, 'cycle_{:0>4}'.format(cycle_id))
                                if not os.path.exists(out_cycle_dir):
                                        os.mkdir(out_cycle_dir)
                                file_loc = out_cycle_dir + '/' + "{:0>3}.csv".format(scene_id)
                                with open(file_loc, 'w') as f:
                                        f_csv = csv.writer(f)
                                        f_csv.writerow(headers)
                                        f_csv.writerows(rows)
                                p.disconnect()
                        total_num += config["number_of_repeat"]
                        print("{0} done".format(total_num))
        print('**************************************')
        print("{0} done".format(config["object_names"]))


if __name__ == '__main__':
        #CONFIG_PATH = "./config.json"
        #JSON_LOAD = open(CONFIG_PATH, encoding='utf-8').read()
        #CONFIG_FILE = json.loads(JSON_LOAD)
        #OBJ_NUM = CONFIG_FILE["obj_num"]
        main()
