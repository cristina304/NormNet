"""
Code to read csv from physics_simulation, then generate depth images and RGB label images.
Example usage:
   "xxxx path" blender --background --python render.py  
/opt/data/private/blender-2.79b-linux-glibc219-x86_64/blender --background --python render.py

Author: Yangbo Lin
Checked: Zhikai Dong
"""
import bpy
import csv
import json
import mathutils
import os
import numpy as np
import shutil
# from math import radians
import math
import re

CONFIG_PATH = "./config_multi.json"
JSON_LOAD = open(CONFIG_PATH, encoding='utf-8').read()
CONFIG_FILE = json.loads(JSON_LOAD)

CAMERA_RESOLUTION = CONFIG_FILE["resolution"]
CAMERA_FOCAL_LEN = CONFIG_FILE["focal_length"]
CAMERA_SENSOR_SIZE = CONFIG_FILE["sensor_size"]
CAMERA_LOCATION = CONFIG_FILE["cam_location_x_y_z"]
CAMERA_ROTATION = CONFIG_FILE["cam_rotation_qw_qx_qy_qz"]
if CONFIG_FILE["unit_of_obj"] == 'mm':
    OBJ_SCALE = 0.001
elif CONFIG_FILE["unit_of_obj"] == 'm':
    OBJ_SCALE = 1.0
DEPTH_DIVIDE = CONFIG_FILE["depth_graph_divide"]
DEPTH_LESS = CONFIG_FILE["depth_graph_less"]
PHYSICS_PATH = os.path.join(CONFIG_FILE["output_dir"], 'physics_result')
OBJ_PATH = CONFIG_FILE["obj_dir"]
DEPTH_PATH = os.path.join(CONFIG_FILE["output_dir"], 'depth_images')
SEGMENT_PATH = os.path.join(CONFIG_FILE["output_dir"], 'segment_images')
RGB_PATH = os.path.join(CONFIG_FILE["output_dir"], 'rgb_images')

# RGB_PATH = "C:/Users/win10/Desktop/test/rgb_images"
if not os.path.exists(DEPTH_PATH):
    os.mkdir(DEPTH_PATH)
if not os.path.exists(SEGMENT_PATH):
    os.mkdir(SEGMENT_PATH)
CLASS_NAME = CONFIG_FILE["object_names"]


# set the parameters of the camera
def camera_set():
    # write the engine kind
    bpy.data.scenes["Scene"].render.engine = "CYCLES"

    # write the camera Internal parameter
    bpy.data.scenes["Scene"].render.resolution_x = CAMERA_RESOLUTION[0]
    bpy.data.scenes["Scene"].render.resolution_y = CAMERA_RESOLUTION[1]
    bpy.data.scenes["Scene"].render.resolution_percentage = 100

    # write the camera focal legth and sensor size. unit is mm
    bpy.data.cameras["Camera"].type = "PERSP"
    bpy.data.cameras["Camera"].lens = CAMERA_FOCAL_LEN
    bpy.data.cameras["Camera"].lens_unit = "MILLIMETERS"
    bpy.data.cameras["Camera"].sensor_width = CAMERA_SENSOR_SIZE[0]
    bpy.data.cameras["Camera"].sensor_height = CAMERA_SENSOR_SIZE[1]
    bpy.data.cameras["Camera"].sensor_fit = "HORIZONTAL"

    bpy.data.scenes["Scene"].render.pixel_aspect_x = 1.0
    bpy.data.scenes["Scene"].render.pixel_aspect_y = CAMERA_SENSOR_SIZE[1] * CAMERA_RESOLUTION[0] / CAMERA_RESOLUTION[
        1] / CAMERA_SENSOR_SIZE[0]
    bpy.data.scenes["Scene"].cycles.progressive = "BRANCHED_PATH"
    bpy.data.scenes["Scene"].cycles.aa_samples = 1
    bpy.data.scenes["Scene"].cycles.preview_aa_samples = 1

    # write the camera external parameter euler is hudu value, unit is m
    bpy.data.objects["Camera"].location = [CAMERA_LOCATION[0],
                                           CAMERA_LOCATION[1],
                                           CAMERA_LOCATION[2]]
    bpy.data.objects["Camera"].rotation_mode = 'QUATERNION'
    bpy.data.objects["Camera"].rotation_quaternion = [CAMERA_ROTATION[0],
                                                      CAMERA_ROTATION[1],
                                                      CAMERA_ROTATION[2],
                                                      CAMERA_ROTATION[3]]

    # let the camera coordinate rotate 180 degree around X axis
    bpy.data.objects["Camera"].rotation_mode = 'XYZ'
    bpy.data.objects["Camera"].rotation_euler[0] = bpy.data.objects["Camera"].rotation_euler[0] + math.pi


# read the csv document
def read_csv(csv_path):
    with open(csv_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader, None)
        obj_name = []
        for row in spamreader:
            # print(row)
            obj_name.append(row[0])
            # print(obj_name)

    pose = np.loadtxt(csv_path,
                      delimiter=",",
                      skiprows=1,
                      usecols=(2, 3, 4, 5, 6, 7, 8))

    # guarantee the pose in the form of matrix
    if pose.ndim==1:
        pose = pose.reshape(1,pose.shape[0])
    return obj_name, pose


# import the obj document and pose values
def import_obj(pose, instance_index):
    # delete the objects in scene at the beginning
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.select = True
        else:
            o.select = False
    bpy.ops.object.delete()
    print(os.path.join(OBJ_PATH, CLASS_NAME[0] + '.obj'))

    # import the corresponding obj model
    scn = bpy.context.scene
    for i, e in enumerate(instance_index):
        if len(e) > 0:
            # print(filepath=os.path.join(OBJ_PATH, CLASS_NAME[i]+'_{}'.format(obj_index) + '.obj'))
            # exit()
            bpy.ops.import_scene.obj(filepath=os.path.join(OBJ_PATH, CLASS_NAME[i] + '.obj'))
            # copy the obj
            objs = [ o for o in bpy.context.scene.objects if o.select]
            obj = objs[0]
            #obj = bpy.context.selected_objects[0]
            for instance_index in range(1, len(e)):
                new_obj = obj.copy()
                new_obj.data = obj.data.copy()
                new_obj.animation_data_clear()
                scn.objects.link(new_obj)

            transfer(i, e, pose)


# transfer the csv data to corresponding instance
def transfer(class_id, instance_ids, pose):
    count = 0
    for instance in bpy.data.objects:
        # identify the name whatever it's uppercase or lowercase
        if CLASS_NAME[class_id] in instance.name:
            instance = bpy.data.objects[instance.name]
            i = instance_ids[count]
            instance.pass_index = i
            instance.scale = [OBJ_SCALE, OBJ_SCALE, OBJ_SCALE]
            instance.location = [pose[i][0], pose[i][1], pose[i][2]]
            instance.rotation_mode = 'QUATERNION'
            instance.rotation_quaternion = [pose[i][3], pose[i][4], pose[i][5], pose[i][6]]
            count = count + 1


# define the compositing nodes and link them
def depth_graph(depth_path,segment_path):
    # edit the node compositing to use node
    bpy.data.scenes["Scene"].use_nodes = 1

    # define the compositing node
    scene = bpy.context.scene
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    for node in nodes:
        nodes.remove(node)

    render_layers = nodes.new("CompositorNodeRLayers")
    divide = nodes.new("CompositorNodeMath")
    divide.operation = "DIVIDE"
    divide.inputs[1].default_value = DEPTH_DIVIDE
    less_than = nodes.new("CompositorNodeMath")
    less_than.operation = "LESS_THAN"
    less_than.inputs[1].default_value = DEPTH_LESS
    multiply = nodes.new("CompositorNodeMath")
    multiply.operation = "MULTIPLY"

    # create the new menu of corresponding scenes
    isExist_depth = os.path.exists(depth_path)
    if isExist_depth:
        shutil.rmtree(depth_path)
    isExist_segment = os.path.exists(segment_path)
    if isExist_segment:
        shutil.rmtree(segment_path)

    # one output is for depth, the other is for label
    output_file_depth = nodes.new("CompositorNodeOutputFile")
    output_file_depth.base_path = depth_path
    output_file_depth.format.file_format = "PNG"
    output_file_depth.format.color_mode = "BW"
    output_file_depth.format.color_depth = '16'

    output_file_label = nodes.new("CompositorNodeOutputFile")
    output_file_label.base_path = segment_path
    output_file_label.format.file_format = "OPEN_EXR"
    output_file_label.format.color_mode = "RGB"
    output_file_label.format.color_depth = '32'

    composite = nodes.new("CompositorNodeComposite")
    viewer = nodes.new("CompositorNodeViewer")

    links.new(render_layers.outputs['Image'], composite.inputs['Image'])
    links.new(render_layers.outputs['Depth'], less_than.inputs[0])
    links.new(render_layers.outputs['Depth'], multiply.inputs[0])

    links.new(less_than.outputs[0], multiply.inputs[1])
    links.new(multiply.outputs[0], divide.inputs[0])

    links.new(divide.outputs[0], output_file_depth.inputs['Image'])
    links.new(divide.outputs[0], viewer.inputs['Image'])
    links.new(render_layers.outputs['Image'], output_file_label.inputs['Image'])


# define the materials(such as color) of the object, and make the objects point at the same materials
def label_graph(label_number):
    mymat = bpy.data.materials.get('mymat')
    if not mymat:
        mymat = bpy.data.materials.new('mymat')
        mymat.use_nodes = True

    # delete the initial nodes
    nodes = mymat.node_tree.nodes
    links = mymat.node_tree.links
    for node in nodes:
        nodes.remove(node)

    # change the color of ColorRamp
    ColorRamp = nodes.new(type="ShaderNodeValToRGB")
    ColorRamp.color_ramp.interpolation = 'LINEAR'
    ColorRamp.color_ramp.color_mode = 'RGB'

    ColorRamp.color_ramp.elements[0].color[:3] = [1.0, 0.0, 0.0]
    ColorRamp.color_ramp.elements[1].color[:3] = [1.0, 1.0, 0.0]

    # add the stop button according to the number of objeccts
    ObjectInfo = nodes.new(type="ShaderNodeObjectInfo")
    OutputMat = nodes.new(type="ShaderNodeOutputMaterial")
    Emission = nodes.new(type="ShaderNodeEmission")

    Math = nodes.new(type="ShaderNodeMath")
    Math.operation = "DIVIDE"
    Math.inputs[1].default_value = label_number

    links.new(ObjectInfo.outputs[1], Math.inputs[0])
    links.new(Math.outputs[0], ColorRamp.inputs[0])
    links.new(ColorRamp.outputs[0], Emission.inputs[0])
    links.new(Emission.outputs[0], OutputMat.inputs[0])

    # let the obj document point at the same materials
    objects = bpy.data.objects
    for obj in objects:
        if obj.type == 'MESH':
            if not 'mymat' in obj.data.materials:
                obj.data.materials.append(mymat)

# set the light
def light_set():
    # denoise the output rgb picture 
    bpy.context.scene.render.layers[0].cycles.use_denoising = True

    bpy.data.lamps["Lamp"].energy = 0.5
    bpy.data.lamps["Lamp"].type = 'POINT'
    bpy.data.objects["Lamp"].location = [CAMERA_LOCATION[0],
                                         CAMERA_LOCATION[1],
                                         CAMERA_LOCATION[2]+2.5]
    bpy.data.objects["Lamp"].rotation_mode = 'QUATERNION'
    bpy.data.objects["Lamp"].rotation_quaternion = [CAMERA_ROTATION[0],
                                                      CAMERA_ROTATION[1],
                                                      CAMERA_ROTATION[2],
                                                      CAMERA_ROTATION[3]]

    # let the light coordinate rotate 180 degree around X axis
    bpy.data.objects["Lamp"].rotation_mode = 'XYZ'
    bpy.data.objects["Lamp"].rotation_euler[0] = bpy.data.objects["Lamp"].rotation_euler[0] + math.pi

def generate_rgb_images(rgb_path):
    # set the materials
    Mat = []
    objects = bpy.data.objects
    for obj in objects:
        if obj.name in ['Camera', 'Lamp']:
            continue
        if obj.name == 'bin':
            COLOR = (255/255,60/255,0/255)	  
            #COLOR = (139/255,69/255,19/255)
        else:
            COLOR = (0.7,0.7,0.7)

        obj.data.materials.clear() 
        mat = bpy.data.materials.new('MaterialName')
        mat.diffuse_shader = 'LAMBERT'
        mat.diffuse_intensity = 1
        mat.diffuse_color = COLOR
        Mat.append(mat)
        bpy.context.scene.objects.active = obj
        mat = bpy.data.materials['Material']
        if len(obj.data.materials):
            obj.data.materials[0] = Mat[-1]
        else:
            obj.data.materials.append(Mat[-1])

    # define the compositing node
    scene = bpy.context.scene
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    for node in nodes:
        nodes.remove(node)

    render_layers2 = nodes.new("CompositorNodeRLayers")
    # divide = nodes.new("CompositorNodeMath")
    # divide.operation = "DIVIDE"
    # divide.inputs[1].default_value = DEPTH_DIVIDE

    # less_than = nodes.new("CompositorNodeMath")
    # less_than.operation = "LESS_THAN"
    # less_than.inputs[1].default_value = DEPTH_LESS

    # multiply = nodes.new("CompositorNodeMath")
    # multiply.operation = "MULTIPLY"

    # one output is for depth, the other is for rgb
    # output_file_depth = nodes.new("CompositorNodeOutputFile")
    # output_file_depth.base_path = depth_path
    # output_file_depth.format.file_format = "PNG"
    # output_file_depth.format.color_mode = "BW"
    # output_file_depth.format.color_depth = '16'

    isExist_rgb = os.path.exists(rgb_path)
    if isExist_rgb:
        shutil.rmtree(rgb_path)

    output_file_rgb = nodes.new("CompositorNodeOutputFile")
    output_file_rgb.base_path = rgb_path
    output_file_rgb.format.file_format = "JPEG"
    output_file_rgb.format.color_mode = "RGB"
    output_file_rgb.format.color_depth = '8'

    # composite = nodes.new("CompositorNodeComposite")
    # viewer = nodes.new("CompositorNodeViewer")

    # links.new(render_layers.outputs['Image'], composite.inputs['Image'])
    # links.new(render_layers.outputs['Depth'], less_than.inputs[0])
    # links.new(render_layers.outputs['Depth'], multiply.inputs[0])

    # links.new(less_than.outputs[0], multiply.inputs[1])
    # links.new(multiply.outputs[0], divide.inputs[0])

    # links.new(divide.outputs[0], output_file_depth.inputs['Image'] )
    # links.new(divide.outputs[0], viewer.inputs['Image'])
    links.new(render_layers2.outputs['Image'], output_file_rgb.inputs['Image'])

if __name__ == "__main__":
    cycle_names = os.listdir(os.path.join(PHYSICS_PATH))
    # for cycle_id in range(0,100):
    # for cycle_id in range(100,200):
    for cycle_id in range(0,50):
        cycle_name = 'cycle_{:0>4}'.format(cycle_id)
        camera_set()
        # get the name list and pose numpy array(x, y, z, qw, qx, qy, qz)
        for scene_id in range(0, 40):
            scene_id = "{:0>3}".format(scene_id)
            print(cycle_name)
            print(scene_id)
            csv_path = '/opt/data/private/data/dataset/realgrasp/physics_result/{}/{}.csv'.format(cycle_name, scene_id)
            obj_name, pose = read_csv(csv_path)
            # obj_name = "{}".format(obj_name[0]) + "_{:0>2d}".format(obj_id)
            # print(1111111111111111)
            print(obj_name)

            # exit()
            instance_index = [[] for i in range(len(CLASS_NAME))]
            # [[ ], [ ],[ ] ]
            # Convert dictionary to list
            CLASS_NAME_list = CLASS_NAME

            for m in range(len(CLASS_NAME_list)):
                instance_index[m] = [i for i, e in enumerate(obj_name) if e == CLASS_NAME_list[m]]
                # instance_index[m] = [i for i, e in enumerate(pose)]
            print(instance_index)
            # print(pose.shape)
            import_obj(pose, instance_index)
            depth_path = os.path.join(DEPTH_PATH, cycle_name)
            segment_path = os.path.join(SEGMENT_PATH, cycle_name)
            # rgb_path = os.path.join(RGB_PATH, cycle_name)
            # print(rgb_path)
            if not os.path.exists(depth_path):
                os.mkdir(depth_path)
            if not os.path.exists(segment_path):
                os.mkdir(segment_path)
            # if not os.path.exists(rgb_path):
                # os.makedirs(rgb_path)
            depth_scene_path = os.path.join(DEPTH_PATH, cycle_name, "{:0>3}".format(scene_id))
            segment_scene_path = os.path.join(SEGMENT_PATH, cycle_name, "{:0>3}".format(scene_id))
            # rgb_scene_path = os.path.join(RGB_PATH, cycle_name, "{}_{:0>3}".format(obj_id ,scene_id))
            # print(11111111111111111111111111111111)
            # print(depth_scene_path)
            # print(segment_scene_path)
            if not os.path.exists(depth_scene_path):
                os.mkdir(depth_scene_path)
            if not os.path.exists(segment_scene_path):
                os.mkdir(segment_scene_path)
            # if not os.path.exists(rgb_scene_path):
                # os.mkdir(rgb_scene_path)
            depth_graph(depth_scene_path, segment_scene_path)
            # exit()
            label_graph(len(obj_name) - 1)
            bpy.ops.render.render()
            # light_set()
            # generate_rgb_images(rgb_scene_path)
            # bpy.ops.render.render()
