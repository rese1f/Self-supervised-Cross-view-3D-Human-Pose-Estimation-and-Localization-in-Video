# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
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
            'person_1': {
                'view_1': {
                    'camera': array(3,3),
                    'pose_c': ndarray(x,17,3),
                    'pose_2d': ndarray(x,17,2),
                }
                'view_2': {
                    'camera': array(3,3),
                    'pose_c': ndarray(x,17,3),
                    'pose_2d': ndarray(x,17,2),
                }
            }
            'person_2': {

            }
        }
        'sample_2': {

        }
    }

    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.sample_keys_list = list(self.dataset)

    def __getitem__(self, index):

        output = list()

        sample_key = self.sample_keys_list[index]
        sample = self.dataset[sample_key]
        person_keys_list = list(sample)
        for person_key in person_keys_list:

            person_list = list()

            person = sample[person_key]
            pose_3d = person['pose_3d']

            person_list.append(pose_3d)

            view_list = list()

            view_keys_list = list(person)[1:]
            for view_key in view_keys_list:
                view = person[view_key]
                camera = view['camera']
                pose_2d = view['pose_2d']
                info = list([camera, pose_2d])
                view_list.append(info)
            
            person_list.append(view_list)
        
            output.append(person_list)

        # index is the key for each sample
        # return list(
        # list1(pose_3d, [[camera1, pose_2d_1],[camera2, pose_2d_2], ...),
        # list2(pose_3d, [[camera1, pose_2d_1],[camera2, pose_2d_2], ...),
        # ...
        # )

        return output

        
    def __len__(self):
        return len(self.sample_keys_list)
