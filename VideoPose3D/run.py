# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Code refactorer: Wenhao Chai 7/5/2021

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import os
import errno

from common.arguments import parse_args
from common.model import *
from common.regressor import *
from common.generators import ChunkedGenerator

args = parse_args()
print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)


print('Loading dataset...')
dataset_path = 'data/data_multi_' + args.dataset + '.npz'
dataset = np.load(dataset_path, allow_pickle=True)['dataset']


print('Loading Model...')
filter_widths = [int(x) for x in args.architecture.split(',')]
model_pos = TemporalModel(args.keypoints_number, 2, args.keypoints_number, filter_widths, 
            args.causal, args.dropout, args.channels, args.dense)

chk_filename = os.path.join(args.checkpoint, args.resume)
print('- Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
print('- This model was trained for {} epochs'.format(checkpoint['epoch']))
model_pos.load_state_dict(checkpoint['model_pos'])
if torch.cuda.is_available():
    model_pos = model_pos.cuda()

receptive_field = model_pos.receptive_field()
lr = args.learning_rate
lr_decay = args.lr_decay  
initial_momentum = 0.1
final_momentum = 0.001
optimizer = optim.Adam(model_pos.parameters(), lr=lr, amsgrad=True)
if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])


print('Preparing data...')
dataset = ChunkedGenerator(dataset)
data_iter = DataLoader(dataset, shuffle=True)

for epoch in tqdm(range(args.epochs)):
    for cameras, pose_cs, pose_2ds in data_iter:
        if args.multi_view:
            raise KeyError('sorry, multi_view is not in beta test')
        pose_2d_m = pose_2ds[0].squeeze(0)
        # N - number of people
        N = pose_2d_m.shape[0]
        # for each person
        for i in range(N):
            pose_2d = pose_2d_m[i].unsqueeze(0).type(torch.float32)
            pose_c = model_pos(pose_2d)
            pose_2d_test = pose_2d[:,receptive_field-1:]
            T, loss = regressor(pose_c, pose_2d_test)

            break
        break