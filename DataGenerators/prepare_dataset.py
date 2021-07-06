# this file is used to convert '.mat' files to one '.npz' file
# for human3.6m dataset, it also select the 17 joints from the whole 32
# input:  some '.mat' files in './data/NAME/'
# output: one '.npz' file in same path

import numpy as np
import os
from scipy.io import loadmat
from glob import glob

from arguments import parse_args
from data_utils import *


args = parse_args()


print('Loading dataset...')

metadata = suggest_metadata(args.dataset)
num_joints = metadata['num_joints']
keypoints = metadata['keypoints']

dataset_path = 'data/' + args.dataset
file_list = glob(dataset_path + '/*.mat')
output_filename = 'data/data_3d_' + args.dataset


print('Preparing dataset...')

output = {}
for f in file_list:
    action = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
    hf = loadmat(f)
    positions = hf['data'].reshape(-1,num_joints,3)[:,keypoints,:]
    output[action] = positions.astype('float32')


print('Saving...')
np.savez_compressed(output_filename, positions_3d=output)


print('Done.')