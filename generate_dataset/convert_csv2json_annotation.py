"""
Conversion from "Fraunhofer IPA Bin-Picking dataset" gt format (csv) to Sileane
dataset gt format (json)
Run script via:
    python <file> --path=/<path>/<to>/<workpiece_folder>/ --projectionType='o'|'p'

@author: Christian Landgraf


"""

import glob
import os
import pandas as pd
import numpy as np

def convert_csv2json_gt(path, out_path):
    """
    Converts our ground truth CSV format to the format of [1]
    """
    if not os.path.exists(out_path): os.mkdir(out_path)

    gt_path = os.path.join(path, "gt", "*", "*", "*.csv")
    
    for file in glob.iglob(gt_path, recursive=True):
        cycle_id = int(file.split('/')[-2].split('_')[-1])
        scale_file = file.split('/')[-3]

        scene_name = file.split('/')[-1].split('.')[0]
        # scene_name = '11_024'

        output_file_name = os.path.join(out_path, scale_file + '_cycle_{:0>4}_'.format(cycle_id) + 'scene_' + scene_name + '.json')
        df = pd.read_csv(file)
        if not df.empty:
            df = df.drop(columns=['ID','class_name'])
        
        # save rotation axes
        tmp_df = pd.DataFrame()
        tmp_df['rot_x'] = df.iloc[:,3:6].values.tolist()
        tmp_df['rot_y'] = df.iloc[:,6:9].values.tolist()
        tmp_df['rot_z'] = df.iloc[:,9:12].values.tolist()    
        
        # combine single values in one column
        new_df = pd.DataFrame()
        new_df['R'] = tmp_df[:].values.tolist()
        new_df['segmentation_id'] = range(len(df))
        new_df['occlusion_rate'] = 1 - np.minimum(df['vs'], [1.0]*len(df))
        new_df['t'] = df.iloc[:,0:3].values.tolist()
        
        # do not write bin position
        new_df.iloc[:].to_json(output_file_name,'records')
        # exit()



if __name__ == '__main__':    
    # import argparse
    
    # parser = argparse.ArgumentParser(description="This script converts our ground truth CSV format to the format of [1].")
    # parser.add_argument('--path', required=True,
    #                     help="Path to workpiece")
    # parser.add_argument('--projectionType', required=True, choices=['o','p'],
    #                     help = "orthogonal (o) or perspective (p) projection")
    
    # args = parser.parse_args()

    input_path ='/opt/data/private/data/dataset/realgrasp'
    output_path = './gt_json'
    convert_csv2json_gt(input_path, output_path)
