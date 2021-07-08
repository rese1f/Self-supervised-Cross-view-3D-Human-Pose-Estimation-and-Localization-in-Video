# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.utils.data import Dataset

class ChunkedGenerator(Dataset):
    """
    Batched data generator, used for training video.
    The sequences are split into equal-length chunks and padded as necessary.

    Notes:
    dataset: a dict to store data, which contains many video sample, for each sample-
        batch_size -- number of objects (e.g. batch_size = 3 for 3 person in the video)
        cameras -- intrinsics matrix with shape (3,3) for each view
        pose_2d -- 2d pose for each person of each view
        pose_3d -- 3d pose for each person

    Structure:

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

    """
    def __init__(self, dataset):
        self.dataset = dataset.tolist()
        self.sample_keys_list = list(self.dataset)

    def __getitem__(self, index):

        cameras, pose_cs, pose_2ds = list(), list(), list()

        sample_key = self.sample_keys_list[index]
        sample = self.dataset[sample_key]
        view_keys_list = list(sample)
        for view_key in view_keys_list:
            view = sample[view_key]
            camera = torch.from_numpy(view['camera'])
            pose_c = torch.from_numpy(view['pose_c'])
            pose_2d = torch.from_numpy(view['pose_2d'])[...,:2]
            if torch.cuda.is_available():
                camera = camera.cuda()
                pose_c = pose_c.cuda()
                pose_2d = pose_2d.cuda()
            cameras.append(camera)
            pose_cs.append(pose_c)
            pose_2ds.append(pose_2d)

        return cameras, pose_cs, pose_2ds

    def __len__(self):
        return len(self.sample_keys_list)
