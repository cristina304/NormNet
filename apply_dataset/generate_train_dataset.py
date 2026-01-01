import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
#from H5DataGenerator import *
from H5DataGenerator import *

# output dirs
OUT_ROOT_DIR = '/opt/data/private/data/dataset/realgrasp'
if not os.path.exists( OUT_ROOT_DIR ):
    os.mkdir(OUT_ROOT_DIR)

# OUT_ROOT_DIR = os.path.join(OUT_ROOT_DIR, 'Cylinder')
# if not os.path.exists( OUT_ROOT_DIR ):
#     os.mkdir(OUT_ROOT_DIR)

OUT_ROOT_DIR = os.path.join(OUT_ROOT_DIR, 'h5_dataset')
if not os.path.exists( OUT_ROOT_DIR ):
    os.mkdir(OUT_ROOT_DIR)


# input dirs
IN_ROOT_DIR = '/opt/data/private/data/dataset/realgrasp'
GT_DIR = os.path.join(IN_ROOT_DIR, 'gt')
SEGMENT_DIR = os.path.join(IN_ROOT_DIR, 'segment_images')
DEPTH_DIR = os.path.join(IN_ROOT_DIR, 'depth_images')


if __name__ == "__main__":
    cycle_idx_list = range(0, 50)
    g = H5DataGenerator('./parameter.json')
    for cycle_id in cycle_idx_list:
        out_cycle_dir = os.path.join(OUT_ROOT_DIR, 'cycle_{:0>4}'.format(cycle_id))
        if not os.path.exists(out_cycle_dir):
            os.makedirs(out_cycle_dir)
        scene_path = os.path.join(GT_DIR,'cycle_{:0>4}'.format(cycle_id))
        scene_names = os.listdir(scene_path)
        for scene_id, s_name in enumerate(scene_names):
            try:
                # load inputs
                depth_image_path = os.path.join(DEPTH_DIR, 'cycle_{:0>4}'.format(cycle_id), s_name.split('.')[0], 'Image0001.png')
                # print('depth_image_path:', depth_image_path)

                depth_image = cv2.imread(depth_image_path,cv2.IMREAD_UNCHANGED)
                seg_img_path = os.path.join(SEGMENT_DIR, 'cycle_{:0>4}'.format(cycle_id), s_name.split('.')[0],'Image0001.exr')
                segment_image = cv2.imread(seg_img_path,cv2.IMREAD_UNCHANGED)
                gt_file_path = os.path.join(GT_DIR, 'cycle_{:0>4}'.format(cycle_id), s_name)
                output_h5_path = os.path.join(out_cycle_dir,  s_name.split('.')[0]+'.h5')
                g.process_train_set(depth_image, segment_image, gt_file_path, output_h5_path)
            except:
                continue

                



