import torch
from auto_dataset import MultiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import parse_args
from auto_dataset import *

args = parse_args()
print(args)

path = r'../DataGenerators/output'
dataset = MultiDataset(path)
train_dataset = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

for epoch in tqdm(range(args.epochs)):
    for data_3d_std, data_2d_std in train_dataset:
        data_3d_std = torch.from_numpy(data_3d_std)
        data_2d_std = torch.from_numpy(data_2d_std)
        if torch.cuda.is_available():
            data_3d_std = data_3d_std.cuda()
            data_2d_std = data_2d_std.cuda()