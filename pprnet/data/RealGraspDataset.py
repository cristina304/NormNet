import os
import sys
from glob import glob
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
# print(FILE_PATH)
# print(ROOT_DIR)
# exit()
sys.path.append(ROOT_DIR)

import pprnet.utils.dataset_util as dataset_util
import numpy as np
import torch
import torch.utils.data as data
import h5py

class RealGraspDataset(data.Dataset):
    def __init__(self,
                data_dir,
                transforms=None,
                collect_names=False,
                collect_error_names=False,
                scale=1000.0,
                is_convert2mm=True,
                num_point_in_h5=16384,
                noise_scale_range=[0, 0],
                ):
        self.num_point = num_point_in_h5
        self.transforms = transforms
        self.collect_names = collect_names
        self.scale = scale
        self.is_convert2mm = is_convert2mm
        self.data_path = glob(os.path.join(data_dir, '*', '*'))
        self.noise_scale_range = noise_scale_range
        # print(os.path.join(data_dir, '*', '*'))
        # print(self.data_path)
        
        # self.dataset = dataset_util.load_dataset_by_cycle( \
        #         data_dir, range(cycle_range[0], cycle_range[1]), range(scene_range[0], scene_range[1]),\
        #         mode, collect_names, collect_error_names)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        h5_file_name = self.data_path[idx]

        f = h5py.File(h5_file_name)
        point_clouds = f['data'][:].reshape(self.num_point, 3)
        label = f['labels'][:]
        rot_label = label[:,3:12].reshape(self.num_point, 3, 3)
        trans_label = label[:,:3].reshape(self.num_point, 3)
        cls_label = label[:,-1].reshape(self.num_point)
        vis_label = label[:, 12].reshape(self.num_point)

        # convert to mm
        if self.is_convert2mm:
            point_clouds *= self.scale
            trans_label *= self.scale

        # add_noise
        noise_scale = np.random.uniform(*self.noise_scale_range)
        all_noise = np.random.standard_normal(point_clouds.shape) * noise_scale
        point_clouds = point_clouds + all_noise

        sample = {
            'point_clouds': point_clouds.astype(np.float32),
            'rot_label': rot_label.astype(np.float32),
            'trans_label': trans_label.astype(np.float32),
            'cls_label': cls_label.astype(np.int64),
            'vis_label': vis_label.astype(np.float32)
        }

        # if self.collect_names:
        #     sample['name'] = self.dataset['name'][idx]

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample
