# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import zipfile
import numpy as np
import h5py
from glob import glob
from shutil import rmtree
import random
from random import choice
import collections
from scipy.io import loadmat
import tqdm

import sys

sys.path.append('../')
from common.h36m_dataset import Human36mDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from common.utils import wrap

output_filename = "test_3d" #'data_3d_sh36m'
output_filename_2d = "test_2d" #'data_2d_sh36m_gt'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']


##### THE ONLY CHANGES #####

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_info(flat_dict, item):
    return 'S0default'
    # return list(flat_dict.keys())[list(flat_dict.values()).index(item)]


def regroup(cascading_dict, min_batch, max_batch, num_samples):
    # flatten the cascading dictionary into a list and shuffle it
    flat_dict = flatten(cascading_dict)
    flat_list = list(flat_dict.values())
    expanded_flat_list = []
    for i in range(max_batch * num_samples):
        expanded_flat_list.append(choice(flat_list))
    random.shuffle(expanded_flat_list)

    # reconstruct the flattened data into size-specified groups
    list_chunk = lambda test_list, batch: [test_list[i:i + batch] for i in
                                                range(0, len(test_list), batch)]
    grouped_list = list_chunk(expanded_flat_list, max_batch)

    # add keys and values to the list to generate the dictionary
    target_dict = {}
    counter = 0
    for item in grouped_list:
        id_dict = {}
        i = 0
        for id in item[:round(random.uniform(min_batch, max_batch))]:
            id_dict["id" + str(i) + ":" + get_info(flat_dict, item[i]) ] = item[i]
            i += 1
        target_dict['sample' + str(counter)] = id_dict
        counter += 1

    return target_dict


##### THE ONLY CHANGES END #####


if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)

    parser = argparse.ArgumentParser(description='Human3.6M dataset downloader/converter')

    # Default: convert dataset preprocessed by Martinez et al. in https://github.com/una-dinosauria/3d-pose-baseline
    parser.add_argument('--from-archive', default='', type=str, metavar='PATH', help='convert preprocessed dataset')

    # Alternatively, convert dataset from original source (the Human3.6M dataset path must be specified manually)
    parser.add_argument('--from-source', default='/mnt/sdb/h36m', type=str, metavar='PATH', help='convert original dataset')

    args = parser.parse_args()

    if args.from_archive and args.from_source:
        print('Please specify only one argument')
        exit(0)

    if os.path.exists(output_filename + '.npz'):
        print('The dataset already exists at', output_filename + '.npz')
        exit(0)

    if args.from_archive:
        print('Extracting Human3.6M dataset from', args.from_archive)
        with zipfile.ZipFile(args.from_archive, 'r') as archive:
            archive.extractall()

        print('Converting...')
        output = {}
        for subject in subjects:
            output[subject] = {}
            file_list = glob('h36m/' + subject + '/MyPoses/3D_positions/*.h5')
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
            for f in file_list:
                action = os.path.splitext(os.path.basename(f))[0]

                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video

                with h5py.File(f) as hf:
                    positions = hf['3D_positions'].value.reshape(32, 3, -1).transpose(2, 0, 1)
                    positions /= 1000  # Meters instead of millimeters
                    output[subject][action] = positions.astype('float32')

        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=output)

        print('Cleaning up...')
        rmtree('h36m')

        print('Done.')

    elif args.from_source:
        print('Converting original Human3.6M dataset from', args.from_source)
        output = {}
        
        from scipy.io import loadmat
        
        for subject in subjects:
            output[subject] = {}
            file_list = glob(args.from_source + '/' + subject + '/MyPoseFeatures/D3_Positions/*.mat')
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
            for f in file_list:
                action = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
                
                if subject == 'S11' and action == 'Directions':
                    continue # Discard corrupted video
                    
                # Use consistent naming convention
                canonical_name = action.replace('TakingPhoto', 'Photo') \
                                       .replace('WalkingDog', 'WalkDog')
                
                hf = loadmat(f)
                
                # DEBUG CHANGE
                # positions = hf['data'][0, 0].reshape(-1, 32, 3)
                positions = hf['data'].reshape(-1, 32, 3)
                
                positions /= 1000 # Meters instead of millimeters
                output[subject][canonical_name] = positions.astype('float32')
        
        output = regroup(output, 2, 4, 1) # the only changed line in the original code
        
        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=output)

        print('Done.')

    else:
        print('Please specify the dataset source')
        exit(0)

    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = Human36mDataset(output_filename + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_2d = []

            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d

    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)

    print('Done.')
