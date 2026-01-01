import numpy as np
import os
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



for scene_idx in range(0,1):
    h5_data_path = '/data1/lvweijie/Parts/number16_dataset/sample/Num16_dataset/h5_dataset_ppr/train/cycle_0019/4_001.h5'
    f = h5py.File(h5_data_path)
    pc = f['data'][:] * 1000.0
    #print(pc)
    #pc = (pc-np.array([100,0,1500]))*np.array([1,-1,-1])
    print(pc)
    #print(pc.shape)
    print('Max XYZ', np.max(pc, axis=0))
    print('Min XYZ', np.min(pc, axis=0))
    x_max = np.max(pc, axis=0)[0]
    x_max = np.max(pc, axis=0)[0]





    print(x_max)
    show_points([pc], [[0,250,200]], radius=5)   # show points
    try:
        labels = f['labels'][:]
        vs_label=labels[:,12]
        #kp_label=labels[:,14:23].reshape([-1,3])* 1000.0
        #print(kp_label.shape)
        model_path = '/data1/lvweijie/Parts/number16_dataset/Num16_revised/Num16_4.obj'
        transformed_models_pc = []
        model = extract_vertexes_from_obj(model_path)
        print(model)
        label_array=np.array(list(set([tuple(t) for t in labels[:,:12]])))
        #print(label_array)
        
        for l in label_array:
            t = l[:3]*1000.0
            #print(model + t)
            R = l[3:12].reshape((3,3))
            #print(R)
            #R = l[3:12].reshape((3,3))
            transformed_models_pc.append( np.dot(model, R.T)+t)
        print(transformed_models_pc)
        transformed_models_pc= np.concatenate(transformed_models_pc, axis=0) 
        #show_points([transformed_models_pc1,pc], [[255,255,255],[255,0,0]], radius=6)# evaluate pose labels
        show_points([transformed_models_pc,pc], [[255,255,255],[255,0,0]], radius=6)# evaluate pose labels
        #show_points([pc,kp_label], [[255,255,255],[255,0,0]], radius=5) # evaluate keypoints labels
        #color_per_point=score2rgb(vs_label)
        #show3d_balls.showpoints(pc, c_gt=color_per_point, ballradius=3) # evaluate vs labels
    except:
        print('There is something wrong!')
        pass
