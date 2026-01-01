import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import show3d_balls
import math

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
    show3d_balls.showpoints(all_points, c_gt=all_color, ballradius=radius)


def score2rgb(score):
    rgb = np.zeros([score.shape[0], 3])
    for i in range(score.shape[0]):
        rgb[i,1] = math.ceil(255 - 255 * score[i])
        rgb[i,0] = math.ceil(255 - 255 * score[i])
        rgb[i,2] = 255

    # print(np.unique(rgb, axis=0))
    return rgb


def extract_vertexes_from_obj(file_name):
    with open(file_name, 'r') as f:
        vertexes = []
        for line in f.readlines():
            line = line.strip()
            if line.startswith('v'):
                words = line.split()[1:]
                xyz = [float(w) for w in words]
                vertexes.append(xyz)
    return np.array(vertexes)


def read_label_csv(file_name):
    lable_list=[]
    with open(file_name, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split(',')
            id = int(words[0])
            if id > 0:
                lable_list.append( list(map(float, words[2:])) )
    label_array = np.array(lable_list)
    assert label_array.shape == (id,14)
    return label_array



if __name__ == '__main__':
    h5_data_path = '/opt/data/private/data/dataset/realgrasp/h5_dataset/val/cycle_0049/013.h5'
    f = h5py.File(h5_data_path)
    pc = f['data'][:] * 1000.0
    print(pc.shape)
    #print('Max XYZ', np.max(pc, axis=0))
    #print('Min XYZ', np.min(pc, axis=0))
    all_noise = np.random.standard_normal(pc.shape) * 1.5
    show_points([pc], [[255,255,255]], radius=5)   # show points
    show_points([pc+all_noise], [[255,255,255]], radius=5)   # show points
    if 1:
        labels = f['labels'][:]
        vs_label=labels[:,12]

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

        transformed_models_pc = []
        label_array=np.array(list(set([tuple(t) for t in labels[:,:15]])))
        print(label_array[:, -3:])

        for l in label_array:
            model_id = int(l[-1])
            model = model_dict[model_id]
            t = l[:3]*1000.0
            R = l[3:12].reshape((3,3))
            transformed_models_pc.append( np.dot(model, R.T) + t )
        
        transformed_models_pc= np.concatenate(transformed_models_pc, axis=0)  
        show_points([transformed_models_pc,pc], [[255,0,0],[255,255,255]], radius=5) #
        color_per_point = score2rgb(vs_label)
        show3d_balls.showpoints(pc, c_gt=color_per_point, ballradius=5, normalizecolor=False) # evaluate vs labels
        # show_points([pc], color_array=[color_per_point], radius=5) # evaluate vs labels
