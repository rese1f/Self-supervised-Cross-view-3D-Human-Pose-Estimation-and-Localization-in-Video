import torch
from torch.utils.data import DataLoader

import numpy as np
import tqdm

from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator


pose_checkpoint = torch.load('checkpoint/pretrained_h36m_cpn.bin', map_location=lambda storage, loc: storage)
traj_checkpoint = torch.load('checkpoint/epoch_80.bin', map_location=lambda storage, loc: storage)

model_pos = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024,dense=False)
model_traj = TemporalModel(17, 2, 1, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024,dense=False)

model_pos.load_state_dict(pose_checkpoint['model_pos'])
model_traj.load_state_dict(traj_checkpoint['model_pos'])
if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_traj = model_traj.cuda()
model_pos.eval()
model_traj.eval()

dataset_path = 'data/data_multi_h36m.npz'
dataset_zip = np.load(dataset_path, allow_pickle=True)['dataset']
dataset = ChunkedGenerator(dataset_zip)
data_iter = DataLoader(dataset, shuffle=True)
loss = list()

with torch.no_grad():
    for camera, pose, pose_2d, count in data_iter:
        camera, pose, pose_2d = camera.squeeze(0), pose.squeeze(0), pose_2d.squeeze(0)
        shape = pose_2d.shape
        v, n, f, j = shape[0], shape[1], shape[2], shape[3]
        # normolization  
        # make a assignment x=(x-c)/f, y=(y-c)/f
        cameras = camera[:,None,None,None,:]
        pose_2d[...,0].add_(-cameras[...,0]).mul_(1/cameras[...,2])
        pose_2d[...,1].add_(-cameras[...,1]).mul_(1/cameras[...,3])
        # pose_2ds -> reshape to [view*number,frame,joint,2]
        pose_2d = pose_2d.reshape(-1,f,j,2)
        pose_pred = model_pos(pose_2d).reshape(v, n, f-242, j, 3)
        traj_pred = model_traj(pose_2d).reshape(v, n, f-242, 1, 3)
        pose_pred += traj_pred
        pose = pose[:,:,121:-121]
        loss.append(multi_n_mpjpe(pose_pred, pose))

print(torch.mean(torch.stack(loss)))