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
from utils.camera import *

args = parse_args()
print(args)

print('Loading data...')

dataset = np.load('data/data_3d_' + args.dataset + '.npz', allow_pickle=True)['positions_3d'].item()
dict_keys = dataset.keys()

if not os.path.exists('output'):
    print('Creating output path...')
    os.makedirs('output')
else:
    print('Cleaning output path...')
    shutil.rmtree('output')
    os.makedirs('output')


print('Loading camera...')
# a dictionary to store the information of camera
camera_metadata = [suggest_metadata(i) for i in args.camera]


print('Generating data...')

output_filename = 'data_multi_' + args.dataset + '.npz'
dataset_zip = dict()

for count in tqdm(range(args.number)):
    
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

    # SCAFFOLD. eliminate collisions
    # note that data_3d_std now is of type "numpy.ndarray", which can be created by np.array([...])
    # 1. create the object from class col_eli
    col_eli_object = col_eli(data_3d_std)
    # 2. give back the value
    data_3d_std = col_eli_object.sequential_collision_eliminate_routine()

    
    dataset_zip[count] = dict()
    for i in range(args.view):
        view_id = 'view_' + str(i)
        camera_info = camera_metadata[i]
        data_c_std = w2c(data_3d_std, camera_info, frame)
        inmat = camera_info['inmat']
        data_2d_std = c2s(data_c_std, inmat)
        dataset_zip[count][view_id]=dict()
        dataset_zip[count][view_id]['camera']=inmat
        dataset_zip[count][view_id]['pose_c']=data_c_std
        dataset_zip[count][view_id]['pose_2d']=data_2d_std


print('Saving data...')
'''
dataset{
        'sample_1': {
            'view_1': {
                'camera': array(3,3),
                'pose_c': ndarray(n,x,17,3),
                'pose_2d': ndarray(n,x,17,2),
            }
            'view_2': {
                'camera': array(3,3),
                'pose_c': ndarray(n,x,17,3),
                'pose_2d': ndarray(n,x,17,2),
            }
        'sample_2': {

        }
    }
'''
np.savez_compressed('output/'+output_filename, dataset=dataset)

print('Done.')