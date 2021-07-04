# this file is to build the dataset in Pytorch

import torch
from glob import glob
import numpy as np
from torch.utils.data import Dataset

class MultiDataset(Dataset):
    def __init__(self, path):
        self.file_list = glob(path+'/*')
        self.len = len(self.file_list)
    
    def __getitem__(self, index):
        single_file_path = self.file_list[index]
        data = np.load(single_file_path)
        data_3d_std = torch.tensor(data['data_3d_std'], device='cuda')
        data_2d_std = torch.tensor(data['data_2d_std'], device='cuda')

        return data_2d_std, data_3d_std

    def __len__(self):
        return self.len


    