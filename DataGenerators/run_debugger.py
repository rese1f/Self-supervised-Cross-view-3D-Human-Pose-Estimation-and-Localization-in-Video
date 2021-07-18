# please run 'prepare_dataset.py' before this
# this file aims to generate the multi-person and cross-view dataset mapping the 3D and 2D
# better run it in terminal

import os
import shutil
import numpy as np
from tqdm import tqdm
from operator import itemgetter
from arguments import parse_args

from camera_utils import *
from utils.random_function import *
from utils.seq_collision_eli import sequential_collision_elimination as col_eli
from utils.camera_debugger import *

args = parse_args()
print(args)

print('Loading data...')

dataset = np.load('data/data_3d_' + args.dataset + '.npz', allow_pickle=True)['positions_3d'].item()
dict_keys = dataset.keys()

if not os.path.exists('output4debug'):
    print('Creating output path...')
    os.makedirs('output4debug')
else:
    print('Cleaning output path...')
    shutil.rmtree('output4debug')
    os.makedirs('output4debug')


print('Loading camera...')
# a dictionary to store the information of camera
camera_metadata = [suggest_metadata(i) for i in args.camera]


print('Generating data...')

output_filename = 'data_multi_' + args.dataset + '_debug.npz'
dataset_zip = dict()

for count in tqdm(range(args.number)):
    
    dataset_zip[count] = dict()
    dataset_zip[count]["world-1"] = dict()
    # randomly get data from dataset
    keys = extract(dict_keys, args.min, args.max)
    sub_dataset = itemgetter(*keys)(dataset)
    # for each data do randomly transforming
    temp, frame_list = list(), list()
    for data in sub_dataset:
        data, f = pre_process(data, args.shift, args.translation, args.rotation)
        frame_list.append(f)
        temp.append(data)
    # all the single data should be in one frame
    frame = np.min(frame_list)
    data_3d_std = list()
    for data in temp:
        data = data[:frame]
        data_3d_std.append(data)
    data_3d_std = np.array(data_3d_std, dtype=np.float32)
    
    dataset_zip[count]["world-1"]["orig_3d"] = data_3d_std 
   
    # SCAFFOLD. eliminate collisions
    # note that data_3d_std now is of type "numpy.ndarray", which can be created by np.array([...])
    # 1. create the object from class col_eli
    col_eli_object = col_eli(data_3d_std)
    # 2. give back the value
    data_3d_std = col_eli_object.sequential_collision_eliminate_routine()

    
    dataset_zip[count]["world-1"]["colision_3d"] = data_3d_std 
    dataset_zip[count]["world-1"]["camera"] = dict()
    for i in range(args.view):
        camera_info = camera_metadata[i]
        data_c_std, camera_vec = w2c(data_3d_std, camera_info, frame)
        
        inmat = camera_info['inmat']
        data_2d_std = c2s(data_c_std, inmat)
        view_id = camera_info['layout_name'] + '_' + str(i)
        dataset_zip[count][view_id]=dict()
        dataset_zip[count]["world-1"]["camera"][view_id] = camera_vec
        cx = inmat[0,2]
        cy = inmat[1,2]
        fx = inmat[0,0]
        fy = inmat[1,1]
        dataset_zip[count][view_id]['camera']=[cx,cy,fx,fy]
        dataset_zip[count][view_id]['pose_c']=data_c_std
        dataset_zip[count][view_id]['pose_2d']=data_2d_std


print('Saving data...')
'''
dataset{
        'sample_1': {
            'view_1': {
                'camera': [cx,cy,fx,fy],
                'pose_c': ndarray(n,x,17,3),
                'pose_2d': ndarray(n,x,17,2),
            }
            'view_2': {
                'camera': [cx,cy,fx,fy],
                'pose_c': ndarray(n,x,17,3),
                'pose_2d': ndarray(n,x,17,2),
            }
        'sample_2': {

        }
    }
'''
np.savez_compressed('output4debug/'+output_filename, dataset=dataset_zip)

print('Done.')