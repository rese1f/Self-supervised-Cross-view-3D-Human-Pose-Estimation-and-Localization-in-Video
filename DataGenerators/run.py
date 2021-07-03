# please run 'prepare_dataset.py' before this
# this file aims to generate the multi-person and cross-view dataset mapping the 3D and 2D

import torch
import numpy as np
from tqdm import tqdm
from operator import itemgetter

from arguments import parse_args
from utils.random_function import *


args = parse_args()
print(args)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print('Loading data...')

dataset = np.load('data_3d_' + args.dataset + '.npz', allow_pickle=True)['positions_3d'].item()
dict_keys = dataset.keys()


print('Generating data...')

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
    #data_3d_std = torch.tensor(data_3d_std, device=device, dtype=torch.float32)
