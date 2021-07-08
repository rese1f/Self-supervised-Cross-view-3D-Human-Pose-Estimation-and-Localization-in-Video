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
from common.loss import *
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
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
dataset = np.load(dataset_path)


print('Loading Model...')
filter_widths = [int(x) for x in args.architecture.split(',')]
model = TemporalModel(args.keypoints_number, 2, args.keypoints_number, filter_widths, 
            args.causal, args.dropout, args.channels, args.dense)

chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
print('- Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
print('- This model was trained for {} epochs'.format(checkpoint['epoch']))
model.load_state_dict(checkpoint['model_pos'])

lr = args.learning_rate
lr_decay = args.lr_decay
initial_momentum = 0.1
final_momentum = 0.001
optimizer = optim.Adam(model.parameters(), Ir=lr, amsgrad=True)
enable_cuda = torch.cuda.is_available()

print('Preparing data...')
dataset = ChunkedGenerator(dataset)
if dataset.__len__ < args.epochs:
    raise KeyError('Dataset overload')
train_iter = DataLoader(dataset, shuffle=True)

for epoch in tqdm(range(args.epochs)):
    for sample in train_iter:
        # up-to-date, we use single view dataset
        # [sample]
        # list(
        # list1(pose_3d, [[camera1, pose_2d_1],[camera2, pose_2d_2], ...),
        # list2(pose_3d, [[camera1, pose_2d_1],[camera2, pose_2d_2], ...),
        # ...
        # )
        for person in sample:
            for view in person:
                camera = view[0]
                pose_2d = torch.from_numpy(view[1])
                if enable_cuda:
                    pose_2d = pose_2d.cuda()

                optimizer.zero_grad()

                # Predict 3D poses
                predicted_3d_pos = model(pose_2d)

                # loss.backward()

                optimizer.step()

            # pose_3d is used to evaluate model, if just for use, we set None
            pose_3d = person[0]
            if pose_3d:
                pose_3d = torch.from_numpy(pose_3d)
                if enable_cuda:
                    pose_3d = pose_3d.cuda()